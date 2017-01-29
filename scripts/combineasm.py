import os
import re
import vtk
import math
import lungpca
import numpy as np

#
# these input directories follow the convention used in bootstrapasm.py
#
left_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Bootstrap-Output/output-left-insp-from-asm49'
right_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Bootstrap-Output/output-right-insp-from-none'

left_phase = 3 # point files created during left lung boothstrap phase
right_phase = 2 # point files created during right lung boothstrap phase

# directory that contains the original STL files (i.e. iso-surfaced mesh created from the distance map file)
left_org_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Left-INSP'
right_org_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Right-INSP'

ref_name = '00001F_INSP_STD_FIC_COPD'
#ref_name = '10292T_INSP_STD_TXS_COPD'

combined_point_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Bootstrap-Output/combined-points'
combined_pca_output_dir = '/Users/jinho/Work/Brigham/ShapeWorks/Data-from-Raul/Bootstrap-Output/combined-points/output-pca'

def get_points_org_space(pnt_file_reg_space, org_stl_file):
    A = np.loadtxt(pnt_file_reg_space)
    #print '{0}: {1}'.format(pfile, A.shape)
    src_points = vtk.vtkPoints()
    src_points.SetNumberOfPoints(A.shape[0])
    for i in range(A.shape[0]):
        src_points.SetPoint(i, A[i,:])
    src_polydata = vtk.vtkPolyData()
    src_polydata.SetPoints(src_points)
        
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(org_stl_file)
    stl_reader.Update()
    tgt_polydata = stl_reader.GetOutput()
    
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(src_polydata)
    icp.SetTarget(tgt_polydata)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.SetMaximumMeanDistance(0.01)
    icp.SetMaximumNumberOfLandmarks(5000)
    icp.StartByMatchingCentroidsOn()
    icp.CheckMeanDistanceOn()
    icp.Modified()
    icp.Update()
    #print icp.GetMeanDistance()
    #print icp.GetNumberOfIterations()
    
    icp_source = vtk.vtkTransformPolyDataFilter()
    icp_source.SetTransform(icp)
    icp_source.SetInputData(src_polydata)
    icp_source.Update()
    icp_polydata = icp_source.GetOutput()
    icp_points = icp_polydata.GetPoints()  # target space
    N = icp_points.GetNumberOfPoints()

    tgt_points = icp.GetLandmarkTransform().GetTargetLandmarks()
    sum = 0
    for i in range(N):
        icp_p = icp_points.GetPoint(i)
        tgt_p = tgt_points.GetPoint(i)
        d2 = vtk.vtkMath.Distance2BetweenPoints(icp_p, tgt_p)
        sum += d2

    rms = math.sqrt(sum / N)
    print 'RMS = {0}'.format(rms)
    
    if rms > 1.0:
        print 'FAIL!!!'
        return False
        
    return tgt_points
    

def save_obj(fname, v, f):
    np.savetxt(fname, v, fmt='v %g %g %g')
    with open(fname, 'a') as fp:
        np.savetxt(fp, f, fmt='f %d %d %d')

def write_obj(polydata, obj_file):
    """ write vtk polydata to OBJ format (vtk lacks OBJ writer)"""
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

def main():
    k = 0
    left_point_files = os.path.join(left_dir, 'point_file_list.txt')
    left_point_dir = os.path.join(left_dir, 'Phase-{0}'.format(left_phase))
    left_pca_dir = os.path.join(left_dir, 'Phase-{0}-PCA'.format(left_phase))

    right_point_files = os.path.join(right_dir, 'point_file_list.txt')
    right_point_dir = os.path.join(right_dir, 'Phase-{0}'.format(right_phase))
    right_pca_dir = os.path.join(right_dir, 'Phase-{0}-PCA'.format(right_phase))

    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(os.path.join(left_pca_dir, 'mean-mesh.obj'))
    obj_reader.Update()
    left_polydata = obj_reader.GetOutput()
    
    obj_reader2 = vtk.vtkOBJReader()
    obj_reader2.SetFileName(os.path.join(right_pca_dir, 'mean-mesh.obj'))
    obj_reader2.Update()
    right_polydata = obj_reader2.GetOutput()
    
    append = vtk.vtkAppendPolyData()
    append.AddInputData(left_polydata)
    append.AddInputData(right_polydata)
    append.Update()
    combined_polydata = append.GetOutput()

    # create directory to contain the combined point files
    if not os.path.exists(combined_point_dir):
        os.makedirs(combined_point_dir)
    
    # create combined PCA output directory if needed
    if not os.path.exists(combined_pca_output_dir):
        os.makedirs(combined_pca_output_dir)
    
    # it's OK if the left and right lung is NOT aligned in this mesh
    # this mesh is used to set the connectivity to generate the mean mesh OBJ
    # so the mesh connectivity matters and the actual point position does not matter
    ref_obj = os.path.join(combined_point_dir, 'ref-mesh.obj')
    write_obj(combined_polydata, ref_obj)

    combined_points = []
    
    with open(left_point_files) as pflist:
        for line in pflist:
            pfile = line.strip()
            #print pfile
            
            basename = re.search('(.*)_left_REG.pnt', pfile).group(1)
            print '{0}: {1}'.format(k, basename)
            
            if basename == ref_name:
                ref_index = k
            
            left_pfile = os.path.join(left_point_dir, basename + '_left_REG.pnt')
            right_pfile = os.path.join(right_point_dir, basename + '_right_REG.pnt')
            
            left_stl_file = os.path.join(left_org_dir, basename + '_left.stl')
            right_stl_file = os.path.join(right_org_dir, basename + '_right.stl')
            
            left_points = get_points_org_space(left_pfile, left_stl_file)
            right_points = get_points_org_space(right_pfile, right_stl_file)
            
            all_points = vtk.vtkPoints()
            left_cnt = left_points.GetNumberOfPoints()
            right_cnt = right_points.GetNumberOfPoints()
            for i in range(left_cnt):
                all_points.InsertNextPoint(left_points.GetPoint(i))
            for i in range(right_cnt):
                all_points.InsertNextPoint(right_points.GetPoint(i))
            
            #save STL for visual verification
            # src_polydata.SetPoints(tgt_points)
            # tgt_stl = os.path.join(left_org_dir, basename + '_REG.stl')
            # print tgt_stl
            # stl_writer = vtk.vtkSTLWriter()
            # stl_writer.SetFileName(tgt_stl)
            # stl_writer.SetInputData(src_polydata)
            # stl_writer.Write()
            
            combined_points.append((basename, all_points))
            
            k = k+1
            #if k > 20:
            #    break

    print 'reference index: {0}'.format(ref_index)
    ref_points = combined_points[ref_index][1]
    ref_polydata = vtk.vtkPolyData()
    ref_polydata.SetPoints(ref_points)
    com = vtk.vtkCenterOfMass()
    com.SetInputData(ref_polydata)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    center = com.GetCenter()
    
    # mean-centering of reference point set
    for i in range(ref_points.GetNumberOfPoints()):
        p = ref_points.GetPoint(i)
        pc = [p[0]-center[0], p[1]-center[1], p[2]-center[2]]
        ref_points.SetPoint(i, pc)
        
    pnt_files = []
    with open(os.path.join(combined_point_dir, 'point_file_list.txt'), 'w') as list_fp:
        for k, basename_points_pair in enumerate(combined_points):
            basename = basename_points_pair[0]
            points = basename_points_pair[1]
            
            # rigid alignment of rest of point sets to the (mean-centered) reference point set
            landmarkTransform = vtk.vtkLandmarkTransform()
            landmarkTransform.SetSourceLandmarks(points)
            landmarkTransform.SetTargetLandmarks(ref_points)
            landmarkTransform.SetModeToRigidBody()
            landmarkTransform.Update()
            
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            
            xform_filter = vtk.vtkTransformPolyDataFilter()
            xform_filter.SetInputData(polydata)
            xform_filter.SetTransform(landmarkTransform)
            xform_filter.Update()
            
            reg_polydata = xform_filter.GetOutput()
            reg_points = reg_polydata.GetPoints()
            
            fname = basename + '_COMBINED.pnt'
            new_pnt_file = os.path.join(combined_point_dir, fname)
            list_fp.write(fname + '\n')
            
            print '{0}: {1}'.format(k, new_pnt_file)
            
            with open(new_pnt_file, 'w') as fp:
                for i in range(reg_points.GetNumberOfPoints()):
                    p = reg_points.GetPoint(i)
                    fp.write('{0} {1} {2}\n'.format(p[0], p[1], p[2]))
            
            pnt_files.append(new_pnt_file)
                    
    print 'performing pca...'
    pca = lungpca.perform_pca(point_files=pnt_files, in_mesh=ref_obj,
                              out_dir=combined_pca_output_dir, text=False,
                              num_modes=0.99)
                              
    left_cnt_file = os.path.join(combined_pca_output_dir, 'left-point-count.txt')
    with open(left_cnt_file, 'w') as lcf:
        lcf.write(str(left_cnt))
        
if __name__ == "__main__":
    main()