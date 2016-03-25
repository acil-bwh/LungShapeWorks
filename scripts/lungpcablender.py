# this file has a symbolic link on the following location
# /Users/jinho/Library/Application Support/Blender/2.72/scripts/
# to use as a module which can be loaded in Blender python console using sys.path
# or put this file directly in any place that is in sys.path
#
# this module is to visualize the output OBJ files that were created by lungpca.py 
# and expect certain input file name convention used by lungpca.py

# ======================================
# blender python only supports Python3.x
# ======================================

import bpy # main blender python module
import math
import os

data_path = None

def hello(): # for test
    print("hello lung pca blender module.")

def set_data_path(path):
    global data_path
    data_path = path

def check_data_path():
    if not data_path or not os.path.exists(data_path):
        print('Set data path first using set_data_path()')
        return False
    return True
    
def show_mean():
    if not check_data_path():
        return
    fname = '{0}/data-mean.obj'.format(data_path)
    bpy.ops.import_scene.obj(filepath=fname)
    obj_objects = bpy.context.selected_objects[:]
    for obj in obj_objects:
        print(obj.name)
        obj.rotation_euler = [0, 0, math.radians(90)]

def show_pca(num_modes = 5):
    if not check_data_path():
        return
    for i in range(num_modes):
        k_str = ['pos', 'neg']
        for k in range(2):
            fname = '{0}/pca-mode-{1}-{2}.obj'.format(data_path, i, k_str[k])
            bpy.ops.import_scene.obj(filepath=fname)
            obj_objects = bpy.context.selected_objects[:]
            for obj in obj_objects:
                print(obj.name)
                obj.rotation_euler = [0, 0, math.radians(90)]
                obj.location.x = i * 320
                obj.location.z = k * 400
                obj.location.y = 0

def show_data():
    if not check_data_path():
        return
    num_cols = 7
    obj_list_file = os.path.join(data_path, 'data_obj_list.txt')
    k = 0
    with open(obj_list_file) as f:
        for line in f:
            i = int(k / num_cols) # row index
            j = k % num_cols # column index
            print('i={0}, j={1}'.format(i,j))
            fname = line.strip()
            bpy.ops.import_scene.obj(filepath=fname)
            obj_objects = bpy.context.selected_objects[:]
            for obj in obj_objects:
                print(obj.name)
                obj.rotation_euler = [0, 0, math.radians(90)]
                obj.location.x = j * 320
                obj.location.z = i * 400
                obj.location.y = 0
            k += 1

def clear():
    # gather list of items of interest.
    candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
    # select them only.
    for object_name in candidate_list:
        bpy.data.objects[object_name].select = True
    # remove all selected.
    bpy.ops.object.delete()
    # remove the meshes, they have no users anymore.
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
