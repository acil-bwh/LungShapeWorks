import os
import sys
import vtk
import math
import logging
import subprocess
import lungpca
import numpy as np
from datetime import datetime
from optparse import OptionParser

# This script reads binary lung segmentation lable map NRRD files as
# input, create distance map NRRD files, create surface model STL files, run
# a bootstrapping algorithm to build correspondence incrementally starting
# from an existing ASM pre-built with smaller number of samples, finally build
# the full ASM model using the obtained correspondence.
# 
# usage: %prog [options] <input_file_list>
# 
# Argument
# --------
# input_file_list: text file that contains the absolute path of a binary input
#                  label map (nrrd) for each line
# Options
# -------
# -d, --asm_dir: input ASM directory
# -r, --ref_name: base name of one of the files in the input list to be used
#                 for the first phase when no ASM directory is given
# -o, --out_dir: output directory (sub-directory 'output' will be created if
#                not given
# -n, --num_phases: number of phases to use (configuration on each step is
#                   hard-coded in function: setup_phase_parameters)
# -p, --prep: preprocess the label map NRRD files to create surface model
# 
# Example usage 1 (without using pre-built ASM):
# vtkpython bootstrap-asm.py left-insp-list.txt -r 00001F_INSP_STD_FIC_COPD_left -n 4 -o output1
#
# Example usage 2 (using a pre-built ASM for first phase):
# vtkpython bootstrap-asm.py left-insp-list.txt -d /Users/jinho/Work/Brigham/ShapeWorks/lung-data-process/corr-4096-param-set5/output -n 4 -o output2
# 
# Lastly (but important)... use -p option to run the script directly from the 
#                           label map files, which will create the surface model
#                           STL files automatically. Those STL files will be 
#                           created in the same directory as the input NRRD files
#                           Also, change the program path for unu and ComputeDistanceMap
#                           below appropriately based on their locations
# data pre-processing commands
unu = '/Users/jinho/Github/CIP-build/teem-build/bin/unu'
dmap = '/Users/jinho/Github/CIP-build/CIP-build/bin/ComputeDistanceMap'


# setup logger
logger = logging.getLogger('bootstrap-asm')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('bootstrap-asm-{0}.log'.format(datetime.now())) # file handler
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
ch = logging.StreamHandler() # console handler
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)

# num_modes = 5 # number of modes to use in PCA-ICP fitting
max_iter = 5  # maximum number of ICP fitting using PCA
num_icp_iter = 200  # maximum number of iteration for a single ICP fitting
max_dist = 0.01  # maximum mean distance (to stop ICP iteration)


def setup_phase_parameters(num_phases):
    if num_phases == 2:
        phase_select_ratios = [0.5, 0.95]
        phase_cutoff_var_ratios = [0.9, 0.99]
    elif num_phases == 3:
        phase_select_ratios = [0.05, 0.1, 0.95]
        phase_cutoff_var_ratios = [0.95, 0.95, 0.99]
    elif num_phases == 4:
        phase_select_ratios = [0.05, 0.1, 0.3, 0.95]
        phase_cutoff_var_ratios = [0.95, 0.95, 0.95, 0.99]
    else:  # maximum number of phases would be 5
        phase_select_ratios = [0.05, 0.1, 0.3, 0.5, 0.95]
        phase_cutoff_var_ratios = [0.95, 0.95, 0.95, 0.97, 0.99]
    return [phase_select_ratios, phase_cutoff_var_ratios]


def create_surface_model(ifile, prep):
    dirname, basename = os.path.split(ifile)
    fname, ext = os.path.splitext(basename)
    stl_file = os.path.join(dirname, fname + '.stl')
    if not prep: # surface model STL files were already created
        return stl_file

    # padding using UNU
    pad = True
    padded = os.path.join(dirname, fname + '_PADDED' + ext)
    if pad:
        pad_command = '{0} pad -i {1} -b pad -v 0 -min -10 -10 -10 -max M+10 M+10 M+10 -o {2}'.format(
            unu, ifile, padded)
        logger.info(pad_command)
        subprocess.call(pad_command, shell=True)

    # distance transform (it seems to create gzip-encoded NRRD always)
    DT = True
    dt_file = os.path.join(dirname, fname + '_DT' + ext)
    if DT:
        dt_command = '{0} -p -d {1} -l {2}'.format(dmap, dt_file, padded)
        logger.info(dt_command)
        subprocess.call(dt_command, shell=True)
        # uncompress using UNU (since vtkNrrdReader does not support gzip
        # encoding)
        unzip_command = '{0} save -f nrrd -i {1} -e raw -o {1}'.format(
            unu, dt_file)
        subprocess.call(unzip_command, shell=True)

    # create mesh from distance transform using iso-surfacing
    iso_surface = True
    if iso_surface:
        readerVolume = vtk.vtkNrrdReader()
        readerVolume.SetFileName(dt_file)

        # Generate an isosurface
        logger.info('iso-surfacing: {0}'.format(dt_file))
        # UNCOMMENT THE FOLLOWING LINE FOR CONTOUR FILTER
        # contourLung = vtk.vtkContourFilter()
        contourLung = vtk.vtkMarchingCubes()
        contourLung.SetInputConnection(readerVolume.GetOutputPort())
        contourLung.ComputeNormalsOn()
        contourLung.SetValue(0, 0)  # Lung isovalue
        contourLung.Update()
        polydata = contourLung.GetOutput()

        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetInputData(polydata)
        stlWriter.SetFileName(stl_file)
        stlWriter.Write()

    return stl_file


def write_score_file(scores, result_file):
    with open(result_file, 'w') as f:
        for score in scores:
            f.write('{0}\n'.format(','.join(map(str, score))))


def select_pca_data_files(scores, ratio):
    """ pick the top 'ratio' files and return the list of file names
    input: 'scores' list of tuple (filename, score)
    """
    total = len(scores)
    pnt_files = []
    cutoff_count = int(round(total * ratio))
    for i in range(cutoff_count):
        pnt_files.append(scores[i][0])
    logger.info('cut off score: {0}'.format(scores[cutoff_count - 1][1]))
    logger.info('max score: {0}'.format(scores[total - 1][1]))
    return pnt_files


def determine_num_modes_to_use(pca, cutoff_var_ratio):
    num_modes = 0  # num modes to use

    if pca:  # replace source to average
        A, E, U = pca
        # determine number of modes to use
        max_modes = E.shape[0]
        sum_E = np.sum(E)
        cum_E = 0
        for i in range(max_modes):
            cum_E += E[i]
            if cum_E / sum_E >= cutoff_var_ratio:
                num_modes = i + 1  # considering zero-based index
                break

    logger.info('Number of PCA modes to use: {0}'.format(num_modes))
    return num_modes


def build_corr(stlFile, refPolyData, pca, num_modes, outDir):
    fit = True
    fit_mode = 'affine'  # or similarity or rigid (affine is best)

    dirname, basename = os.path.split(stlFile)
    fname, ext = os.path.splitext(basename)
    stl_file_out = os.path.join(outDir, fname + '_REG_' + fit_mode + '.stl')
    pnt_file = os.path.join(outDir, fname + '_REG.pnt')

    if fit:
        stl_reader = vtk.vtkSTLReader()
        stl_reader.SetFileName(stlFile)
        stl_reader.Update()
        target = stl_reader.GetOutput()
        source = refPolyData

        if pca:  # replace source to average
            A = pca[0]
            for i in range(source.GetNumberOfPoints()):
                source.GetPoints().SetPoint(i, A[3 * i:3 * i + 3])

        logger.info('ICP: ' + stlFile)
        iter = 0
        prev_rms = 1000
        prev_num_iter = 0

        while iter < max_iter:  # loop for PCA + ICP fitting

            icp = vtk.vtkIterativeClosestPointTransform()
            icp.SetSource(source)
            icp.SetTarget(target)
            if fit_mode == 'rigid':
                icp.GetLandmarkTransform().SetModeToRigidBody()  # for comparison to affine
            elif fit_mode == 'similarity':
                icp.GetLandmarkTransform().SetModeToSimilarity()  # for comparison to affine
            else:
                icp.GetLandmarkTransform().SetModeToAffine()  # better than similarity
            icp.SetMaximumNumberOfIterations(num_icp_iter)
            icp.SetMaximumMeanDistance(max_dist)
            icp.SetMaximumNumberOfLandmarks(5000)
            icp.StartByMatchingCentroidsOn()
            icp.CheckMeanDistanceOn()
            icp.Modified()
            icp.Update()
            NS = icp.GetLandmarkTransform().GetSourceLandmarks().GetNumberOfPoints()
            NT = icp.GetLandmarkTransform().GetTargetLandmarks().GetNumberOfPoints()
            num_iter = icp.GetNumberOfIterations()

            logger.debug(icp.GetMeanDistance())
            logger.debug('{0} {1} {2}'.format(NS, NT, num_iter))

            source_points = source.GetPoints()
            target_sampled = vtk.vtkPolyData()
            target_sampled.DeepCopy(source)
            target_points = vtk.vtkPoints()
            target_points.DeepCopy(
                icp.GetLandmarkTransform().GetTargetLandmarks())
            target_sampled.SetPoints(target_points)

            # icp.Inverse() # to register target to reference
            rigid_inverse = vtk.vtkLandmarkTransform()
            rigid_inverse.SetSourceLandmarks(target_points)
            rigid_inverse.SetTargetLandmarks(source_points)
            rigid_inverse.SetModeToRigidBody()
            rigid_inverse.Update()

            registered = vtk.vtkTransformPolyDataFilter()
            registered.SetTransform(rigid_inverse)
            registered.SetInputData(target_sampled)
            registered.Update()
            reg_polydata = registered.GetOutput()

            icp_source = vtk.vtkTransformPolyDataFilter()
            icp_source.SetTransform(icp)
            icp_source.SetInputData(source)
            icp_source.Update()
            icp_polydata = icp_source.GetOutput()

            # calculate the distance from icp-ed reference to target
            icp_points = icp_polydata.GetPoints()  # target space
            tgt_points = target_sampled.GetPoints()  # target space
            reg_points = reg_polydata.GetPoints()  # source space

            sum = 0
            reg_np = np.array([])  # for pca projection
            N = icp_points.GetNumberOfPoints()

            for i in range(N):
                icp_p = icp_points.GetPoint(i)
                tgt_p = tgt_points.GetPoint(i)
                reg_p = reg_points.GetPoint(i)
                d2 = vtk.vtkMath.Distance2BetweenPoints(icp_p, tgt_p)
                sum += d2
                reg_np = np.append(reg_np, reg_p)

            rms = math.sqrt(sum / N)
            logger.info('RMS = {0}'.format(rms))

            if not pca or prev_rms < rms:
                if pca:
                    logger.info('RMS is increasing... stopping...')
                    reg_polydata = prev_reg_polydata  # revert to result of previous iteration
                    reg_points = reg_polydata.GetPoints()
                    rms = prev_rms
                    num_iter = prev_num_iter
                break
            else:
                A, E, U = pca
                logger.debug('data shape: {0}'.format(reg_np.shape))
                # project and get coefficient
                C = np.dot(U.T, np.subtract(reg_np, A))
                N = num_modes  # number of modes to use
                for i in range(N):
                    C[i] = max(C[i], -3 * math.sqrt(E[i]))
                    C[i] = min(C[i], 3 * math.sqrt(E[i]))
                logger.debug(C)
                # use only first 5 modes to reconstruct
                P = np.add(A, np.dot(U[:, :N], C[:N]))
                for i in range(source_points.GetNumberOfPoints()):
                    source_points.SetPoint(i, P[3 * i:3 * i + 3])
                # now start next iteration with changed source
                iter += 1
                prev_rms = rms
                prev_num_iter = num_iter

            prev_reg_polydata = vtk.vtkPolyData()  # get a copy to revert to previous result
            prev_reg_polydata.DeepCopy(reg_polydata)

        # note that there may be the redundant points selected in the target landmark points
        # which will cause some zero-area triangles, which will be removed in the
        # STL file
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetInputData(reg_polydata)
        stlWriter.SetFileName(stl_file_out)
        stlWriter.Write()

        logger.info('Creating point file: ' + pnt_file)
        fp = open(pnt_file, 'w')
        for i in range(reg_points.GetNumberOfPoints()):
            p = reg_points.GetPoint(i)
            fp.write('{0} {1} {2}\n'.format(p[0], p[1], p[2]))
        fp.close()

        score = (pnt_file, rms)
        return score

    # else: # fitting is already done (PNT file exists)
    #   if os.path.exists(pnt_file):
    #
    #   else:
    #     return None


def save_obj(fname, v, f):
    np.savetxt(fname, v, fmt='v %g %g %g')
    with open(fname, 'a') as fp:
        np.savetxt(fp, f, fmt='f %d %d %d')


def convert_stl_to_obj(stl_file, obj_file):
    """ convert mesh file in STL format to OBJ format (vtk lacks OBJ writer)"""
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file)
    stl_reader.Update()
    polydata = stl_reader.GetOutput()

    flist = np.array([])
    vlist = np.array([])
    points = polydata.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        vlist = np.append(vlist, points.GetPoint(i))

    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        point_ids = cell.GetPointIds()
        for j in range(point_ids.GetNumberOfIds()):
            flist = np.append(flist, point_ids.GetId(j) + 1)

    vertices = np.reshape(vlist, (-1, 3))
    faces = np.reshape(flist, (-1, 3))
    save_obj(obj_file, vertices, faces)


def create_ref_stl(inputSTL, outputSTL):
    """ take the original NRRD-converted STL, smooth and decimate """
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(inputSTL)
    stl_reader.Update()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(stl_reader.GetOutputPort())
    smooth.SetNumberOfIterations(40)
    smooth.SetRelaxationFactor(0.2)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()
    smooth.Update()

    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputConnection(smooth.GetOutputPort())
    decimate.SetTargetReduction(0.95)
    decimate.Update()
    polydata = decimate.GetOutput()

    logger.info('Number of points in the reference output: {0}'.format(polydata.GetNumberOfPoints()))

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(outputSTL)
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

    return polydata


def create_ref_model(stlFiles, refName):
    for stl_file in stlFiles:
        dirname, basename = os.path.split(stl_file)
        fname, ext = os.path.splitext(basename)
        if fname == refName:
            stl_file_out = os.path.join(dirname, fname + '_REF.stl')
            obj_file_out = os.path.join(dirname, fname + '_REF.obj')
            ref_polydata = create_ref_stl(stl_file, stl_file_out)
            convert_stl_to_obj(stl_file_out, obj_file_out)
            return [stl_file_out, obj_file_out, ref_polydata]


def create_ref_model_from_pca_mean(asm_dir):
    obj_file_out = os.path.join(asm_dir, 'mean-mesh.obj')
    stl_file_out = os.path.join(asm_dir, 'mean-mesh.stl')
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(obj_file_out)
    obj_reader.Update()
    ref_polydata = obj_reader.GetOutput()
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(stl_file_out)
    stl_writer.SetInputData(ref_polydata)
    stl_writer.Write()
    return [stl_file_out, obj_file_out, ref_polydata]


def load_pca(pca_dir, text=False):
    logger.info('Reading PCA data...')
    if text:
        mean_file = os.path.join(pca_dir, 'pca-mean.txt')
        A = np.loadtxt(mean_file)
        eigval_file = os.path.join(pca_dir, 'pca-eigvals.txt')
        E = np.loadtxt(eigval_file)
        modes_file = os.path.join(pca_dir, 'pca-modes.txt')
        U = np.loadtxt(modes_file)
    else:
        pca = np.load(os.path.join(pca_dir, 'lung-asm.npz'))
        A = pca['mean']
        E = pca['variance']
        U = pca['modes']
    
    logger.debug('mean shape: {0}'.format(A.shape))
    logger.debug('eigval shape: {0}'.format(E.shape))
    logger.debug('modes shape: {0}'.format(U.shape))
    
    return [A, E, U]


def main():
    desc = """This script reads binary lung segmentation lable map NRRD files as
    input, create distance map NRRD files, create surface model STL files, run
    a bootstrapping algorithm to build correspondence incrementally starting
    from an existing ASM pre-built with smaller number of samples, finally build
    the full ASM model using the obtained correspondence."""

    usage = "usage: %prog [options] <input_file_list>"
    parser = OptionParser(description=desc, usage=usage)
    parser.add_option('-d', '--asm_dir', type='string', dest='asm_dir', default=None,
                      help='input ASM directory')
    parser.add_option('-r', '--ref_name', type='string', dest='ref_name', default=None,
                      help='base name of one of the files in the input list to \
                      be used for the first phase when no ASM directory is given')
    parser.add_option('-o', '--out_dir', type='string', dest='out_dir', default='output',
                      help="output directory (sub-directory 'output' \
                      will be created if not given)")
    parser.add_option('-n', '--num_phases', type='int', dest='num_phases', default=4,
                      help="number of phases to use")
    parser.add_option('-p', '--prep', action='store_true', dest='prep', default=False,
                      help='preprocess the label map NRRD files to create surface model')

    (op, args) = parser.parse_args()
    
    if len(args) != 1:
        logger.error('input file list is required.')
        parser.print_help()
        exit()
        
    if not op.asm_dir and not op.ref_name:
        logger.error('either -d ASM_DIR or -r REF_NAME should be provided.')
        parser.print_help()
        exit()
    
    output_dir = op.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create log file pointer
    input_file_list = args[0]
    input_files = []
    stl_files = []
    with open(input_file_list) as f:
        for line in f:
            input_files.append(line.strip())

    logger.info('------------------------------------------')
    logger.info('Creating surface models from label maps...')
    logger.info('------------------------------------------')
    for ifile in input_files:
        stl_file = create_surface_model(ifile, op.prep)
        stl_files.append(stl_file)

    logger.info('creating reference surface model...')
    if op.asm_dir:
        ref_stl, ref_obj, ref_polydata = create_ref_model_from_pca_mean(op.asm_dir)
    else:
        ref_stl, ref_obj, ref_polydata = create_ref_model(stl_files, op.ref_name)

    logger.info('-----------------------------------------')
    logger.info('Building point-to-point correspondence...')
    logger.info('-----------------------------------------')
    pca = None  # start first phase without pca data
    if op.asm_dir:  # use external pca for the first phase
        pca = load_pca(op.asm_dir)
    num_files = len(stl_files)

    phase_select_ratios, phase_cutoff_var_ratios = setup_phase_parameters(
        op.num_phases)

    for phase, ratio in enumerate(phase_select_ratios):
        logger.info('============= Phase {0} ============='.format(phase))
        phase_output_dir = os.path.join(output_dir, 'Phase-{0}'.format(phase))
        if not os.path.exists(phase_output_dir):
            os.makedirs(phase_output_dir)
        cutoff_var_ratio = phase_cutoff_var_ratios[phase]

        num_modes = determine_num_modes_to_use(pca, cutoff_var_ratio)

        scores = []
        for i, stl_file in enumerate(stl_files):
            logger.info('----- {0} / {1} -----'.format(i, num_files))
            score = build_corr(stl_file, ref_polydata, pca,
                               num_modes, phase_output_dir)
            scores.append(score)

        scores.sort(key=lambda x: x[1])  # sort by rms
        score_file = os.path.join(
            output_dir, 'Phase-{0}-scores.txt'.format(phase))
        write_score_file(scores, score_file)

        logger.info('selecting pca data files with top {0}% data'.format(ratio * 100))
        pnt_files = select_pca_data_files(scores, ratio)

        phase_pca_output_dir = os.path.join(
            output_dir, 'Phase-{0}-PCA'.format(phase))
        if not os.path.exists(phase_pca_output_dir):
            os.makedirs(phase_pca_output_dir)

        logger.info('saving the correspondence point file list for input of PCA...')
        point_file_list = os.path.join(
            phase_pca_output_dir, 'point_file_list.txt')
        with open(point_file_list, 'w') as pf:
            for pnt_file in pnt_files:
                pf.write(pnt_file + '\n')

        logger.info('performing pca...')
        pca = lungpca.perform_pca(point_files=pnt_files, in_mesh=ref_obj,
                                  out_dir=phase_pca_output_dir, text=False,
                                  num_modes=0.99)

if __name__ == "__main__":
    args = ' '.join(sys.argv)
    logger.info(args)
    logger.info('vtk version: {0}'.format(vtk.vtkVersion.GetVTKSourceVersion()))
    
    main()
