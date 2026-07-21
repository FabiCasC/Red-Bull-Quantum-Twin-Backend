import bpy
import mathutils
import os

bpy.ops.wm.read_factory_settings(use_empty=True)
obj_path = os.path.abspath("blender/tinker.obj")
bpy.ops.wm.obj_import(filepath=obj_path)

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        print(f"Objeto: {obj.name}")
        print(f"Location: {obj.location}")
        print(f"Rotation: {obj.rotation_euler}")
        bb = [obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box]
        xs = [v.x for v in bb]
        ys = [v.y for v in bb]
        zs = [v.z for v in bb]
        print(f"X: {min(xs):.2f} a {max(xs):.2f} ancho:{max(xs)-min(xs):.2f}")
        print(f"Y: {min(ys):.2f} a {max(ys):.2f} prof:{max(ys)-min(ys):.2f}")
        print(f"Z: {min(zs):.2f} a {max(zs):.2f} alto:{max(zs)-min(zs):.2f}")