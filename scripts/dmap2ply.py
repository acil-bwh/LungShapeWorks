# create mesh (PLY format) files from gray scale distance map (NRRD)
# use python interactor console in Slicer and run by
# execfile('path/to/this/file/dmap2ply.py')

dataPath = '/Users/jinho/Work/Brigham/ShapeWorks/lung-data-process'

def createGrayModel(fileName):
    slicer.util.loadVolume(dataPath + '/' + fileName + '.nrrd')
    print dataPath + '/' + fileName + '.nrrd'
    volumeNode = getNode(fileName)
    parameters = {}
    parameters["InputVolume"] = volumeNode.GetID()
    outModel = slicer.vtkMRMLModelNode()
    slicer.mrmlScene.AddNode( outModel )
    parameters["OutputGeometry"] = outModel.GetID()
    parameters["Threshold"] = 0
    grayMaker = slicer.modules.grayscalemodelmaker
    # important to set wait_for_completion=True (otherwise it returns immediately)
    slicer.cli.run(grayMaker, None, parameters, wait_for_completion=True)
    print 'Saving ' + fileName + ' ...'
    slicer.util.saveNode(outModel, dataPath + '/' + fileName + '.ply')


if __name__ == '__main__':
    # dmapFiles.txt should contain the list of file names only (without extension or path)
    dataFile = dataPath + '/dmapFiles.txt'
    with open(dataFile) as f:
        for line in f:
            fname = line.strip()
            createGrayModel(fname)
