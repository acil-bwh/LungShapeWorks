
import time,subprocess,os
import argparse


class ShapeWorks():
  def __init__(self,workingdir):
    
    self.workingdir=workingdir
    self.paramsfile=None
    self.params=dict()

  def create_params_file(self):
  
    file_name='shapeworksrun.'+time.strftime("%Y-%m-%d-%H:%M:%S")+'.xml'
    self.paramsfile=os.path.join(self.workingdir,file_name)
    file=open(self.paramsfile,"w")
    file.write('<?xml version="1.0" encoding="UTF-8"?>')
    for key in self.params:
      if self.params[key] == None:
        continue
      
      file.write('<%s>\n'%key)
      if hasattr(self.params[key],'__iter__'):
        for value in self.params[key]:
          file.write('%s\n'%value)
      else:
        file.write('%s\n'%self.params[key])
      
      file.write('</%s>\n'%key)

    file.close()

  def run(self,options=''):
    self.create_params_file()
    exec_name=type(self).__name__
    cmd='%s %s %s'%(exec_name,self.paramsfile,options)
    subprocess.call(cmd,shell=True)


class ShapeWorksRun(ShapeWorks):

  def __init__(self,workingdir,inputs,output_prefix='shapeworksrun-output',point_files=None):

    #super(ShapeWorksRun,self).__init__(workingdir)
    ShapeWorks.__init__(self,workingdir)
    self.params['inputs']=inputs
    self.params['point_files']=point_files
    self.params['number_of_particles']=256
    self.params['iteractions_per_split']=200
    self.params['starting_regularization']=10.0
    self.params['ending_regularization']=0.1
    self.params['optimization_iterations']=1000
    self.params['checkpointing_interval']=50
    self.params['relative_weighting']=1

    self.params['output_points_prefix']=output_prefix
    self.params['verbose']=0


class ShapeWorksGroom(ShapeWorks):
  def __init__(self,workingdir,inputs,outputs):

    super(ShapeWorksGroom, self).__init__(workingdir)

    self.params['inputs']=inputs
    self.params['outputs']=outputs
    self.params['background']=0
    self.params['foreground']=1
    self.params['pad']=10
    self.params['transform_file']=None
    self.params['antialias_iterations']=20
    self.params['blur_sigma']=2.0
    self.params['fastmarching_isovalue']=0.0
    self.params['verbose']=0


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Utility to run shape works over a list of cases')

  parser.add_argument("-c",dest="caseList",required=True)
  parser.add_argument("--dataDir",dest="dataDir",required=True)
  parser.add_argument("--suffix",dest="suffix",required=True)
  parser.add_argument("--output_prefix",dest="output_prefix",required=True)
  op=parser.parse_args()

  caseList=list()
  if os.path.isfile(op.caseList):
    file=open(op.caseList,"r")
    for line in file.readlines():
      caseList.append(line)
  else:
    caseList.append(op.caseList)

  #Create input file list and check that is valid
  input=list()
  for cc in caseList:
    in_file=os.path.join(op.dataDir,cc+op.suffix)
    if os.path.isfile(in_file):
      input.append(in_file)

  swr=ShapeWorksRun(op.dataDir,input,op.output_prefix)
  swr.params['verbose']=1
  swr.run()


