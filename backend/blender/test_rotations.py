import bpy
import mathutils
import os
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
obj_path = os.path.abspath("blender/tinker.obj")
bpy.ops.wm.obj_import(filepath=obj_path)

rotations = [
    (0, 0, 0),
    (1.5708, 0, 0),
    (0, 1.5708, 0),
    (0, 0, 1.5708),
]

for idx, rot in enumerate(rotations):
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.rotation_euler = rot

    bpy.context.view_layer.update()

    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for v in obj.bound_box:
                wv = obj.matrix_world @ mathutils.Vector(v)
                for i in range(3):
                    min_co[i] = min(min_co[i], wv[i])
                    max_co[i] = max(max_co[i], wv[i])

    dims = [max_co[i] - min_co[i] for i in range(3)]
    center = mathutils.Vector([(min_co[i] + max_co[i]) / 2 for i in range(3)])
    size = max(dims)

    cam = mathutils.Vector((center.x, center.y - size * 2, center.z + size * 0.3))

    bpy.ops.object.camera_add(location=cam)
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    direction = center - cam
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 50
    camera.data.clip_end = 99999

    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.04, 1)

    bpy.ops.object.light_add(type='SUN', location=(center.x, center.y - size, center.z + size))
    bpy.context.object.data.energy = 5

    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 400
    bpy.context.scene.render.resolution_y = 300
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    out = os.path.abspath(f"blender/rot_{idx}_{'_'.join(str(round(r,2)) for r in rot)}.png")
    bpy.context.scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {out}")

    bpy.ops.object.camera_add(location=(0,0,0))  # reset cam slot