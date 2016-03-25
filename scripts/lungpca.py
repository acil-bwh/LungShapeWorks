import os
import numpy as np
from optparse import OptionParser

def pca(data):
    """
    input: data as 2D NumPy array (cols: observations, rows: variables)
    output: principal components, eigenvalues, mean as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    mean = data.mean(axis=1)
    # print mean.shape
    cdata = data - np.tile(mean, (n,1)).T
    # calculate the covariance matrix
    C = np.dot(cdata.T, cdata) / (n - 1)
    print 'dimension of covariance matrix: {0}'.format(C.shape)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eig(C)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # carry out the transformation on the data using eigenvectors
    # and return the principal components (U), eigenvalues, and mean data
    U = np.dot(cdata, evecs)
    # normalize U 
    # discard the last column (numerically unstable, no meaning since mean was subtracted)
    for i in range(n-1):
        U[:,i] /= np.linalg.norm(U[:,i])
    return U[:,:-1], evals[:-1], mean

def save_obj(fname, v, f):
    np.savetxt(fname, v, fmt='v %g %g %g')
    with open(fname, 'a') as fp:
        np.savetxt(fp, f, fmt='f %d %d %d')
        
def main():
    desc = """This script reads correspondent point sets, computes PCA, save
    resulting principal components, eigenvalues (variance), and mean data. 
    Optionally it creates triangle mesh files of first N mode variations, mean data, 
    and original data samples in OBJ format (if the mesh connectivity is given).
    
    Parameters
    ----------
    in_dir : str
        input data directory
        
    in_file : str
        text file that contains list of point files to process (can be relative to in_dir)
        the content should have only file names (relative to in_dir)
    
    in_mesh : str, optional
        OBJ file that contains the common mesh connectivity of all point sets
    
    out_dir : str, optional
        output directory (sub-directory 'output' of in_dir if not given)
        
    num_modes: int, optional
        number of modes to save (full modes if not specified)
        
    text : boolean, optional
        save output files in ascii text format (default is numpy binary)
    """

    parser = OptionParser(description=desc)
    parser.add_option('-i', '--in_dir', type='string', dest='in_dir', default=None,
                      help='input data directory')
    parser.add_option('-f', '--in_file', type='string', dest='in_file', default=None,
                      help='text file that contains list of point \
                      files to process (can be relative to in_dir). \
                      The content should have only file names (relative to in_dir)')
    parser.add_option('-m', '--in_mesh', type='string', dest='in_mesh', default=None,
                      help='OBJ file that contains the common mesh \
                      connectivity of all point sets')
    parser.add_option('-o', '--out_dir', type='string', dest='out_dir', default=None,
                      help="output directory (sub-directory 'output' \
                      of in_dir if not given)")
    parser.add_option('-n', '--num_modes', type='int', dest='num_modes', default=0,
                      help="number of modes to save (full modes if not specified)")
    parser.add_option('-t', '--text', action='store_true', dest='text', default=False,
                      help='save output files in ascii text format (default is numpy binary)')

    (op, args) = parser.parse_args()
    if not op.in_dir or not op.in_file:
        parser.error('-i and -f is required.')
    
    # check input directory
    if not os.path.isdir(op.in_dir):
        parser.error('Input directory {0} does not exist.'.format(op.in_dir))
        
    # check output directory
    if not op.out_dir: # empty or not given
        op.out_dir = os.path.join(op.in_dir, 'output')
        if not os.path.isdir(op.out_dir): # create one
            os.makedirs(op.out_dir)
    
    if not os.path.isdir(op.out_dir):
        parser.error('Output directory {0} does not exist.'.format(op.out_dir))
    
    print 'Output directory: {0}'.format(op.out_dir)
    
    # check mesh object file
    if op.in_mesh and not os.path.isfile(op.in_mesh):
        in_mesh = os.path.join(op.in_dir, op.in_mesh)
        if not os.path.isfile(in_mesh):
            parser.error('Input mesh file does not exist: {0}'.format(op.in_mesh))
        op.in_mesh = in_mesh
        
    point_files = []
    
    if os.path.isfile(op.in_file): # check if absolute path is given
        in_file = op.in_file
    else:
        in_file = os.path.join(op.in_dir, op.in_file)
    
    with open(in_file) as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                point_files.append(os.path.join(op.in_dir, line))
    
    #print point_files
    
    num_samples = len(point_files)
    print 'Number of point files to process: {0} in file: {1}'.format(num_samples, in_file)
    
    if num_samples < 2:
        print 'Process requires at least 2 point set files.'
        exit()
    
    # to apply correction matrix (RAS to IJK) to align back to original CT image
    # turned out that this is unnecessary
    # ras2ijk = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    data = np.array([])
    for fname in point_files:
        s = np.loadtxt(fname)
        # s = np.dot(s, ras2ijk)
        v = np.reshape(s, -1) # make vector
        if not data.any():
            data = v
        else:
            data = np.vstack([data, v])
    
    data = data.T # to make columns observations
    print 'dimension of data matrix: {0}'.format(data.shape)
    U, E, A = pca(data)
    print U.shape
    
    full_num_modes = U.shape[1]
    
    # print variances per mode
    sumE = np.sum(E)
    cumE = 0
    print 'mode, variance, percent variance, sum percent'
    for i in range(E.size):
        cumE += E[i]
        print 'mode {0}: {1}, {2} %, {3} %'.format(i, E[i], E[i]/sumE*100, cumE/sumE*100)
    
    # double-check ortho-normality of U (principal components in columns)
    I = np.dot(U.T, U)
    #print I
    print 'determinent of U.T*U: {0}'.format(np.linalg.det(I)) 
    print 'trace of U.T*U: {0}'.format(I.trace())
    
    # truncate U to keep first N modes
    if op.num_modes > 0 and op.num_modes < full_num_modes:
        print 'keeping only first {0} modes in the output data...'.format(op.num_modes)
        U = np.delete(U, np.s_[op.num_modes:full_num_modes], axis=1)
        print 'dimension of U after truncation: {0}'.format(U.shape)
    
    if op.text:
        # save serialized mean
        fname = os.path.join(op.out_dir, 'pca-mean.txt')
        np.savetxt(fname, A)
        
        # save eigenvalues
        fname = os.path.join(op.out_dir, 'pca-eigvals.txt')
        np.savetxt(fname, E)
        
        # save principal components
        fname = os.path.join(op.out_dir, 'pca-modes.txt')
        np.savetxt(fname, U)
    else: 
        # save mean, variance, and modes in one zipped numpy binary (uncompressed)
        # with 4-byte float data type
        fname = os.path.join(op.out_dir, 'lung-asm.npz')
        np.savez(fname, mean=A.astype(np.dtype('f4')), 
                        variance=E.astype(np.dtype('f4')), 
                        modes=U.astype(np.dtype('f4')))

    # save average asc & obj
    mean = np.reshape(A, (-1,3))
    print 'center of mean shape: {0}'.format(mean.mean(axis=0))

    fname = os.path.join(op.out_dir, 'data-mean.asc') # for reading from MeshLab
    np.savetxt(fname, mean)
    
    flist = []
    faces = None
    if op.in_mesh:
        # extract faces from OBJ file
        with open(op.in_mesh) as mf:
            for line in mf:
                line = line.strip()
                if line and line[0] == 'f':
                    fstr = line[1:].strip()
                    vlist = [int(v) for v in fstr.split()]
                    flist.append(vlist)
                    
        faces = np.array(flist) # make numpy array
        fname = '{0}/data-mean.obj'.format(op.out_dir)
        if os.path.exists(fname): # change name if exists
            b = os.path.basename(fname) + "-1"
            fname = '{0}/{1}.obj'.format(op.out_dir, b)
        
        save_obj(fname, mean, faces)

    #
    # create additional mesh objects for visualization and verification
    #
    if not faces is None:
        # save first 5 mode variations into obj format
        num_modes = 5
        for i in range(num_modes):
            k_str = ['pos', 'neg']
            for k in range(2):
                dev = 3 * np.sqrt(E[i]) * U[:,i]
                if k == 0:
                    dev = -dev
                m = A + dev
                m = np.reshape(m, (-1,3))
                fname = '{0}/pca-mode-{1}-{2}.obj'.format(op.out_dir, i, k_str[k])
                save_obj(fname, m, faces)

        # reconstruction test using first N modes
        # change the flag to enable or disable (TODO: add this to option)
        reconstunction_test = False
        num_modes = 20
        if reconstunction_test:
            print 'Creating mesh files of reconstructed data samples...'
            RU = U[:,range(num_modes)]
            for i in range(data.shape[1]):
                dv = data[:,i] - A
                x, e, rank, val = np.linalg.lstsq(RU, dv)
                r = A + np.dot(RU, x)
                r = np.reshape(r, (-1,3))
                print 'sample {0} reconstruction error: {1}'.format(i, np.sqrt(e[0]))
                fname = '{0}/data-recons-{1}.obj'.format(op.out_dir, i)
                save_obj(fname, r, faces)

        # create original data obj files with the mesh connectivity of mean data
        # change the flag to enable or disable (TODO: add this to option)
        create_original_data_mesh = False
        if create_original_data_mesh:
            print 'Creating mesh files of original data samples...'
            obj_list_file = os.path.join(op.out_dir, 'data_obj_list.txt')
            with open(obj_list_file, 'w') as of:
                k = 0
                for p in point_files:
                    fname = os.path.basename(p)
                    pre, ext = os.path.splitext(fname)
                    obj_path = '{0}/{1}.obj'.format(op.out_dir, pre)
                    d = np.reshape(data[:,k], (-1,3))
                    save_obj(obj_path, d, faces)
                    of.write(obj_path+'\n')
                    k += 1

if __name__ == "__main__":
    main()
