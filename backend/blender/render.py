import bpy
import sys
import os
import mathutils

argv = sys.argv
argv = argv[argv.index("--") + 1:]
ers_status = argv[0] if len(argv) > 0 else "NORMAL"
output_path = argv[1] if len(argv) > 1 else "blender/render_output.png"

bpy.ops.wm.read_factory_settings(use_empty=True)

obj_path = os.path.abspath("blender/tinker.obj")
bpy.ops.wm.obj_import(filepath=obj_path)

color_map = {
    "NORMAL": (0.0, 1.0, 0.53, 1.0),
    "ALERTA": (1.0, 0.8, 0.0, 1.0),
    "CRITICO": (0.8, 0.0, 0.0, 1.0),
    "PELIGRO": (0.8, 0.0, 0.0, 1.0),
}
color = color_map.get(ers_status, color_map["NORMAL"])

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        mat = bpy.data.materials.new(name="ERS_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = 0.3
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.2
        obj.data.materials.clear()
        obj.data.materials.append(mat)

# Rotación correcta encontrada: (0, 0, 1.5708)
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.rotation_euler = (0, 0, 1.5708)

bpy.context.view_layer.update()

# Recalcular bbox tras rotación
min_co = [float('inf')] * 3
max_co = [float('-inf')] * 3
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        for v in obj.bound_box:
            wv = obj.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                min_co[i] = min(min_co[i], wv[i])
                max_co[i] = max(max_co[i], wv[i])

center = mathutils.Vector([(min_co[i] + max_co[i]) / 2 for i in range(3)])
size = max(max_co[i] - min_co[i] for i in range(3))

# Cámara lateral con ángulo 3/4 vista clásica F1
cam = mathutils.Vector((
    center.x + size * 0.3,
    center.y - size * 2.0,
    center.z + size * 0.25
))

bpy.ops.object.camera_add(location=cam)
camera = bpy.context.object
bpy.context.scene.camera = camera
direction = center - cam
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = 60
camera.data.clip_end = 99999

# Mundo oscuro
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
world = bpy.context.scene.world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.02, 0.04, 1)

# Iluminación 3 puntos
bpy.ops.object.light_add(type='SUN', location=(center.x, center.y - size, center.z + size))
bpy.context.object.data.energy = 5

bpy.ops.object.light_add(type='AREA', location=(center.x - size, center.y + size * 0.5, center.z + size * 0.5))
fill = bpy.context.object
fill.data.energy = size * 15
fill.data.size = size

bpy.ops.object.light_add(type='AREA', location=(center.x, center.y + size, center.z - size * 0.2))
rim = bpy.context.object
rim.data.energy = size * 8
rim.data.size = size * 0.6

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 500
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = os.path.abspath(output_path)

bpy.ops.render.render(write_still=True)
print(f"Rendered: {output_path} | ERS: {ers_status}")