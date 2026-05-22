"""Unified Blender rendering script.

Replaces the previously separated:
    - render.py                    -> --task views / --task normalize
    - blender_scripts_obj_ori.py   -> --task ortho_ref
    - blender_scripts_obj_edit.py  -> --task ortho_edit
    - blender_obj_with_mask.py     -> --task mask

Usage:
    blender -b -P render.py -- --task <task> --object <path> --output_folder <dir> [task-specific args]
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np  # noqa: E402


# ============================================================================
# 1. 通用常量 / 通用工具函数
# ============================================================================

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import") else bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
    "vrm": bpy.ops.import_scene.vrm if hasattr(bpy.ops.import_scene, "vrm") else None,
}

EXT = {
    "PNG": "png", "JPEG": "jpg", "OPEN_EXR": "exr",
    "TIFF": "tiff", "BMP": "bmp", "HDR": "hdr", "TARGA": "tga",
}


def scene_meshes() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if isinstance(getattr(obj, "data", None), bpy.types.Mesh):
            yield obj


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def reset_scene(keep_camera_light: bool = False) -> None:
    """Resets the scene to a clean state (optionally keeping Camera / Light)."""
    for obj in bpy.data.objects:
        if keep_camera_light and obj.type in {"CAMERA", "LIGHT"}:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Load a 3D model (glb/gltf/obj/fbx/ply/stl/vrm/blend/...) into the current scene."""
    ext = object_path.split(".")[-1].lower()
    if ext == "glb" or ext == "gltf":
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif ext == "blend":
        bpy.ops.wm.append(directory=object_path, link=False)
    elif ext == "obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=object_path)
        else:
            bpy.ops.import_scene.obj(filepath=object_path)
    elif ext == "ply":
        bpy.ops.import_mesh.ply(filepath=object_path)
    elif ext == "stl":
        bpy.ops.import_mesh.stl(filepath=object_path)
    elif ext == "vrm":
        bpy.ops.import_scene.vrm(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj: Optional[bpy.types.Object] = None,
               ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    objs = [single_obj] if single_obj is not None else list(scene_meshes())
    for obj in objs:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no mesh objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene(shrink: float = 1.0, unparent_camera: bool = False) -> Tuple[float, Vector]:
    """Normalize the scene so the longest bbox side becomes `shrink`, centered at origin."""
    roots = list(get_scene_root_objects())
    if len(roots) > 1:
        parent = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent)
        for obj in roots:
            obj.parent = parent
    else:
        parent = roots[0]

    bbox_min, bbox_max = scene_bbox()
    longest = max(bbox_max - bbox_min)
    if longest <= 0:
        raise RuntimeError("degenerate bbox")
    scale = (1.0 / longest) * float(shrink)
    parent.scale = parent.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    parent.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")
    if unparent_camera and "Camera" in bpy.data.objects:
        bpy.data.objects["Camera"].parent = None
    return scale, offset


def apply_norm_to_objects(object_set, scale: float, offset) -> None:
    """Apply a pre-computed (scale, offset) to a specific set of objects (for mask alignment)."""
    roots = [obj for obj in object_set if not obj.parent]
    parent = bpy.data.objects.new("ParentEmpty_Apply", None)
    bpy.context.scene.collection.objects.link(parent)
    for obj in roots:
        obj.parent = parent
    parent.scale = parent.scale * scale
    bpy.context.view_layer.update()
    offset_vec = Vector((float(offset[0]), float(offset[1]), float(offset[2])))
    parent.matrix_world.translation += offset_vec
    bpy.ops.object.select_all(action="DESELECT")


def strip_bsdf_normal_links() -> None:
    """Remove `Normal` input links on all Principled BSDF nodes (so textures don't perturb normals)."""
    for obj in scene_meshes():
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    if node.inputs["Normal"].is_linked:
                        mat.node_tree.links.remove(node.inputs["Normal"].links[0])
                if node.type == "TRANSPARENT_BSDF":
                    node.inputs[0].default_value = (1, 1, 1, 1)


def convert_to_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        return
    bpy.context.view_layer.objects.active = meshes[0]
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.convert(target="MESH")


def triangulate_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        return
    bpy.context.view_layer.objects.active = meshes[0]
    for o in meshes:
        o.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def unhide_all_objects() -> None:
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)


def export_glb(filepath: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.context.scene.objects:
        if o.type in {"MESH", "EMPTY", "ARMATURE", "LIGHT", "CAMERA"}:
            o.select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects[0]
    bpy.ops.export_scene.gltf(
        filepath=filepath, export_format="GLB", export_apply=True,
        export_yup=True, export_materials="EXPORT",
        export_cameras=False, export_lights=False,
    )


# ============================================================================
# 2. 渲染引擎 / 相机 / 灯光 / compositor 节点树
# ============================================================================

def init_render_engine(engine: str, resolution: int, geo_mode: bool = False,
                       threads: int = 32) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128 if not geo_mode else 1
    scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    scene.cycles.use_denoising = True
    scene.cycles.adaptive_threshold = 0
    try:
        scene.cycles.tile_size = 8192
    except Exception:
        pass

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"


def init_persp_camera() -> bpy.types.Object:
    """Create a perspective camera (for --task views)."""
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cc = cam.constraints.new(type="TRACK_TO")
    cc.track_axis = "TRACK_NEGATIVE_Z"
    cc.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(empty)
    cc.target = empty
    return cam


def configure_default_ortho_camera() -> Tuple[bpy.types.Object, Any]:
    """Configure the default Blender Camera as an ORTHO camera with TRACK_TO constraint."""
    cam = bpy.context.scene.objects["Camera"]
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.0
    cam.data.lens = 35
    cam.data.sensor_height = 32
    cam.data.sensor_width = 32
    cc = cam.constraints.new(type="TRACK_TO")
    cc.track_axis = "TRACK_NEGATIVE_Z"
    cc.up_axis = "UP_Y"
    return cam, cc


def init_environment_light(env_light: float = 1.0, strength: float = 1.3) -> None:
    """Remove all lights, use world ambient background at given strength."""
    for light in [o for o in bpy.context.scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(light, do_unlink=True)
    back = bpy.context.scene.world.node_tree.nodes["Background"]
    back.inputs["Color"].default_value = Vector([env_light, env_light, env_light, 1.0])
    back.inputs["Strength"].default_value = strength


def build_normal_compositor(output_base: str):
    """Build a compositor node-tree that writes RGB-encoded normals to EXR files.
    Returns the file-output node, caller sets file_slots[0].path per view.
    """
    scene = bpy.context.scene
    # 用当前激活的 view_layer，避免不同 Blender 版本/语言下默认名不是 "View Layer"
    view_layer = bpy.context.view_layer if bpy.context.view_layer is not None else scene.view_layers[0]
    view_layer.use_pass_normal = True
    view_layer.use_pass_z = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for n in list(nodes):
        nodes.remove(n)

    rl = nodes.new("CompositorNodeRLayers")
    scale_n = nodes.new("CompositorNodeMixRGB")
    scale_n.blend_type = "MULTIPLY"
    scale_n.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(rl.outputs["Normal"], scale_n.inputs[1])
    bias_n = nodes.new("CompositorNodeMixRGB")
    bias_n.blend_type = "ADD"
    bias_n.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_n.outputs[0], bias_n.inputs[1])

    out = nodes.new("CompositorNodeOutputFile")
    out.label = "Normal Output"
    out.format.file_format = "OPEN_EXR"
    out.format.color_mode = "RGB"
    out.base_path = output_base
    links.new(bias_n.outputs[0], out.inputs[0])
    return out


# ============================================================================
# 3. 相机运动 / 相机内外参
# ============================================================================

def set_camera_mvdream(azimuth_deg: float, elevation_deg: float, distance: float) -> bpy.types.Object:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    point = (distance * math.cos(az) * math.cos(el),
             distance * math.sin(az) * math.cos(el),
             distance * math.sin(el))
    cam = bpy.data.objects["Camera"]
    cam.location = point
    direction = -cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rot, _ = obj.matrix_world.decompose()
    rt = rot.to_matrix()
    m = [[rt[i][j] for j in range(3)] + [pos[i]] for i in range(3)]
    m.append([0, 0, 0, 1])
    return m


def get_K_persp(camd) -> Matrix:
    scene = bpy.context.scene
    f = camd.lens
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    pct = scene.render.resolution_percentage / 100
    par = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    sw, sh = camd.sensor_width, camd.sensor_height
    if camd.sensor_fit == "VERTICAL":
        s_u = rx * pct / sw / par
        s_v = ry * pct / sh
    else:
        s_u = rx * pct / sw
        s_v = ry * pct * par / sh
    return Matrix(((f * s_u, 0, rx * pct / 2),
                   (0, f * s_v, ry * pct / 2),
                   (0, 0, 1)))


def get_K_ortho(camd, ortho_scale: float) -> Matrix:
    scene = bpy.context.scene
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    par = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    fx = rx / ortho_scale
    fy = ry / ortho_scale / par
    return Matrix(((fx, 0, rx / 2), (0, fy, ry / 2), (0, 0, 1)))


def get_RT_3x4(cam) -> np.ndarray:
    bpy.context.view_layer.update()
    loc, rot = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rot.to_matrix())
    t = np.asarray(loc)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:, None]], 1)


# ============================================================================
# 4. 五种 Task 的具体实现
# ============================================================================

def _common_init_and_load(args):
    """Reset + load object + kill lights + set ambient light + clean up BSDF normals."""
    init_render_engine(args.engine, args.resolution)
    reset_scene(keep_camera_light=False)
    load_object(args.object)
    init_environment_light()
    try:
        strip_bsdf_normal_links()
    except Exception as e:
        print(f"[WARN] strip_bsdf_normal_links failed: {e}")


def task_views(args):
    """Preprocess step 1: render N views from JSON, save mesh.ply + normalized.glb + transforms.json.
    If --transform_path is provided, read scale/offset from it (used for mask normalization).
    """
    os.makedirs(args.output_folder, exist_ok=True)

    _common_init_and_load(args)

    # ---- mask alignment mode: just apply the stored normalization ----
    if args.transform_path:
        with open(args.transform_path, "r") as f:
            meta = json.load(f)
        scale = float(meta["scale"])
        offset = np.array(meta["offset"], dtype=np.float32)
        apply_norm_to_objects(list(bpy.context.scene.objects), scale, offset)
        if args.save_mesh:
            unhide_all_objects(); convert_to_meshes(); triangulate_meshes()
            name = os.path.splitext(os.path.basename(args.object))[0]
            bpy.ops.wm.ply_export(filepath=os.path.join(args.output_folder, name + ".ply"))
        return

    # ---- normal path: do normalization, render all views, save mesh.ply / normalized.glb ----
    scale, offset = normalize_scene(shrink=1.0, unparent_camera=False)

    cam = init_persp_camera()

    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": [],
    }

    views = json.loads(args.views) if args.views else []
    for i, view in enumerate(views):
        r = view["radius"] * args.radius_scale
        cam.location = (
            r * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            r * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            r * np.sin(view["pitch"]),
        )
        cam.data.lens = 16 / np.tan(view["fov"] / 2)
        bpy.context.scene.render.filepath = os.path.join(args.output_folder, f"{i:03d}.png")
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append({
            "file_path": f"{i:03d}.png",
            "camera_angle_x": view["fov"],
            "transform_matrix": get_transform_matrix(cam),
        })

    with open(os.path.join(args.output_folder, "transforms.json"), "w") as f:
        json.dump(to_export, f, indent=4)

    if args.save_mesh:
        unhide_all_objects(); convert_to_meshes(); triangulate_meshes()
        bpy.ops.wm.ply_export(filepath=os.path.join(args.output_folder, "mesh.ply"))

    if args.export_normalized_glb:
        export_glb(os.path.join(args.output_folder, "normalized.glb"))


def _ortho_render_loop(args, do_normalize: bool, image_subdir: str,
                       normal_subdir: Optional[str] = None,
                       save_camera_npy: bool = False):
    """Common ORTHO-16-views loop used by ortho_ref / ortho_edit / mask tasks.
    Returns (scale, offset) applied to scene (None if not normalized)."""
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, image_subdir), exist_ok=True)
    if save_camera_npy:
        os.makedirs(os.path.join(args.output_folder, "camera"), exist_ok=True)

    # 1) 先把渲染引擎设成 CYCLES + GPU（和 task_views 保持一致），
    #    否则默认引擎可能触发 GL / EGL 路径导致 headless 崩溃。
    init_render_engine(args.engine, args.resolution)

    # 2) 清场景（此时默认 Camera/Light 也一并清掉，避免残留状态）
    reset_scene(keep_camera_light=False)

    # 3) 导入模型 + 环境光
    load_object(args.object)
    init_environment_light()

    try:
        strip_bsdf_normal_links()
    except Exception as e:
        print(f"[WARN] strip_bsdf_normal_links failed: {e}")

    # 4) 归一化（先于相机创建，保证 empty/相机位置对齐归一化后的坐标系）
    scale_offset = None
    if do_normalize:
        scale, offset = normalize_scene(shrink=1.0, unparent_camera=False)
        scale_offset = (scale, offset)

    # 5) 现在再创建正交相机（此时场景里一定没有老 "Camera"，新建一个干净的）
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.0
    cam.data.lens = 35
    cam.data.sensor_height = 32
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    # optional mask loading (only for task=mask)
    if getattr(args, "mask_path", None) and scale_offset is not None:
        init_objs = set(bpy.context.scene.objects)
        load_object(args.mask_path)
        new_objs = set(bpy.context.scene.objects) - init_objs
        apply_norm_to_objects(new_objs, scale_offset[0], scale_offset[1])

    # empty target
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # optional normal compositor
    normal_out = None
    if normal_subdir:
        normal_out = build_normal_compositor(os.path.join(args.output_folder, normal_subdir))

    scene = bpy.context.scene
    subject_width = 1
    distance = 2

    for i in range(args.num_images):
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = subject_width
        azimuth = i * 360.0 / args.num_images
        bpy.context.view_layer.update()
        set_camera_mvdream(azimuth, 0, distance)

        render_path = os.path.join(args.output_folder, image_subdir, f"{i:03d}.png")
        scene.render.filepath = render_path
        if normal_out is not None:
            normal_out.file_slots[0].path = f"{i:03d}"

        bpy.ops.render.render(write_still=True)

        # convert normal EXR to PNG uint16
        if normal_subdir:
            exr = os.path.join(args.output_folder, normal_subdir, f"{i:03d}0001.exr")
            if os.path.exists(exr):
                try:
                    import cv2  # lazy
                    normal = cv2.imread(exr, cv2.IMREAD_UNCHANGED)
                    normal_u16 = (normal * 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(args.output_folder, normal_subdir, f"{i:03d}.png"), normal_u16)
                    os.remove(exr)
                except Exception as e:
                    print(f"[WARN] normal EXR->PNG failed for view {i}: {e}")

        # camera intrinsic/extrinsic npy
        if save_camera_npy:
            K = get_K_ortho(cam.data, ortho_scale=cam.data.ortho_scale)
            RT = get_RT_3x4(cam)
            para_path = os.path.join(args.output_folder, "camera", f"{i:03d}.npy")
            paras = {
                "intrinsic": np.array(K, np.float32),
                "extrinsic": np.array(RT, np.float32),
                "fov": cam.data.angle,
                "azimuth": azimuth,
                "elevation": 0.0,
                "distance": distance,
                "focal": cam.data.lens,
                "sensor_width": cam.data.sensor_width,
                "near": distance - 1,
                "far": distance + 1,
                "camera": "ortho",
            }
            np.save(para_path, paras)

    return scale_offset


def task_ortho_ref(args):
    """Preprocess step 2 (blender_scripts_obj_ori.py): normalize + 16 ortho views + camera/*.npy."""
    _ortho_render_loop(
        args, do_normalize=True,
        image_subdir="image_ori",
        normal_subdir=None,   # 暂不渲染法线；后续需要时改回 "normal_ori"
        save_camera_npy=True,
    )


def task_ortho_edit(args):
    """After-edit rendering (blender_scripts_obj_edit.py): 16 ortho views, NO normalization."""
    _ortho_render_loop(
        args, do_normalize=False,
        image_subdir="image",
        normal_subdir=None,   # 暂不渲染法线；后续需要时改回 "normal"
        save_camera_npy=False,
    )


def task_mask(args):
    """Mask rendering (blender_obj_with_mask.py): normalize main mesh + overlay mask + render 16 ortho views, then binarize."""
    _ortho_render_loop(
        args, do_normalize=True,
        image_subdir="mask_image", normal_subdir=None,
        save_camera_npy=False,
    )

    imgs_path = os.path.join(args.output_folder, "mask_image")
    masks_path = os.path.join(args.output_folder, "mask")
    os.makedirs(masks_path, exist_ok=True)

    try:
        import cv2  # lazy
        for img_id in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_id)
            mask_path = os.path.join(masks_path, img_id)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[-1] != 4:
                continue
            b, g, r, a = cv2.split(img)
            mask_img = np.full(img.shape[:2], 255, dtype=np.uint8)
            transparent = a < 255
            pure_black = (b <= 5) & (g <= 5) & (r <= 5) & (a == 255)
            mask_img[transparent | pure_black] = 0
            cv2.imwrite(mask_path, mask_img)
    except Exception as e:
        print(f"[ERROR] img2mask failed: {e}")

    if os.path.exists(imgs_path):
        shutil.rmtree(imgs_path, ignore_errors=True)


# ============================================================================
# 5. 命令行入口
# ============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Unified Blender rendering script.")
    p.add_argument("--task", type=str, default="views",
                   choices=["views", "ortho_ref", "ortho_edit", "mask"],
                   help="渲染任务类型")

    # 通用
    p.add_argument("--object", type=str, required=True, help="主模型路径")
    p.add_argument("--output_folder", type=str, required=True, help="输出目录")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--engine", type=str, default="CYCLES",
                   choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])

    # --task views
    p.add_argument("--views", type=str, default="",
                   help="JSON string of views, each entry: {yaw,pitch,radius,fov}")
    p.add_argument("--save_mesh", action="store_true", help="保存 mesh.ply（仅 --task views）")
    p.add_argument("--export_normalized_glb", action="store_true",
                   help="保存 normalized.glb（仅 --task views）")
    p.add_argument("--radius_scale", type=float, default=0.76)
    p.add_argument("--transform_path", type=str, default=None,
                   help="若提供，从该 transforms.json 读 scale/offset 对 mask 做归一化（仅 --task views）")

    # --task ortho_*
    p.add_argument("--num_images", type=int, default=16, help="ortho 视角数（仅 ortho_* / mask）")

    # --task mask
    p.add_argument("--mask_path", type=str, default=None, help="mask 模型路径（仅 --task mask）")

    return p


def main():
    parser = build_argparser()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)

    dispatch = {
        "views": task_views,
        "ortho_ref": task_ortho_ref,
        "ortho_edit": task_ortho_edit,
        "mask": task_mask,
    }
    dispatch[args.task](args)


if __name__ == "__main__":
    main()
