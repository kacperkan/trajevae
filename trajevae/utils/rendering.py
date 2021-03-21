import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from trajevae.utils.skeleton import Skeleton

DEFAULT_JOINT_SIZE = 0.025
EMPHASIZED_JOINT_SIZE = 0.04

CONE_PATH = Path(__file__).parent.parent.parent / "external" / "cone2.obj"

xml_head = """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="1000"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{},{},{}" target="{},{},{}" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            <sampler type="ldsampler">
                <integer name="sampleCount" value="16"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="{}"/>
                <integer name="height" value="{}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <!-- default 0.5 -->
            <rgb name="diffuseReflectance" value="1,1,1"/>
        </bsdf>
    """


xml_limb_segment = """
        <shape type="cylinder">
            <float name="radius" value="0.025"/>
            <point name="p0" x="{}" y="{}" z="{}"/>
            <point name="p1" x="{}" y="{}" z="{}"/>
            <bsdf type="twosided">
                <bsdf type="roughplastic">
                    <rgb name="diffuseReflectance" value="{},{},{}"/>
                </bsdf>
            </bsdf>
        </shape>
    """

xml_joint_segment = """
        <shape type="sphere">
            <float name="radius" value="{}"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="roughplastic">
                <rgb name="diffuseReflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_trajectory_segment = """
        <shape type="cylinder">
            <float name="radius" value="0.0025"/>
            <point name="p0" x="{}" y="{}" z="{}"/>
            <point name="p1" x="{}" y="{}" z="{}"/>
            <bsdf type="twosided">
                <bsdf type="roughplastic">
                    <rgb name="diffuseReflectance" value="{},{},{}"/>
                </bsdf>
            </bsdf>
        </shape>
    """

# xml_trajectory_joint_segment = """
#         <shape type="sphere">
#             <float name="radius" value="{}"/>
#             <transform name="toWorld">
#                 <translate x="{}" y="{}" z="{}"/>
#             </transform>
#             <bsdf type="roughplastic">
#                 <rgb name="diffuseReflectance" value="{},{},{}"/>
#             </bsdf>
#         </shape>
#     """

xml_trajectory_joint_segment = """
        <shape type="obj">
            <string name="filename" value="cone2.obj"/>
            <transform name="toWorld">
                <scale value="0.05"/>
                <lookat origin="{},{},{}" target="{},{},{}" up="0,0,1" />
            </transform>
            <bsdf type="coating">
                <float name="intIOR" value="1.7"/>
                <bsdf type="plastic">
                    <rgb name="diffuseReflectance" value="{},{},{}"/>
                </bsdf>
            </bsdf>
        </shape>
    """

xml_tail = """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <translate x="0" y="0" z="{}"/>
            </transform>
        </shape>
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="30" y="30" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="1.7,1.7,1.7"/>
            </emitter>
        </shape>
    </scene>
    """


PALETTE = [
    (25, 95, 235),
    # (8, 30, 74) # original ,
    (255, 102, 99),
    (25, 95, 74),
    (230, 194, 41),
    (241, 113, 5),
]


def decode_image(byte_data: List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def colormap(x: float, y: float, z: float) -> List[float]:
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    if np.any(np.isnan(vec)):
        print("Vec")
        exit(0)
    return [vec[0], vec[1], vec[2]]


def build_humanoid(
    human: np.ndarray,
    skeleton: Skeleton,
    trajectory: Optional[np.ndarray],
    human_color: Union[List[float], Tuple[float, ...]] = colormap(
        8 / 255, 30 / 255, 74 / 255
    ),
    trajectory_color: Union[List[float], Tuple[float, ...]] = [
        30 / 255,
        30 / 255,
        8 / 255,
    ],
    trajectory_joint_size: float = 0.01,
    joints_to_emphasize: Optional[Union[Tuple[int, ...], List[int]]] = None,
    color_joints_to_emphasize: Optional[
        Union[List[float], Tuple[float, ...]]
    ] = None,
) -> List[str]:
    xml_components = []
    for j, j_parent in enumerate(skeleton.parents()):
        if j_parent == -1:
            continue

        xml_components.append(
            xml_limb_segment.format(*human[j], *human[j_parent], *human_color)
        )

        if (
            joints_to_emphasize is not None
            and color_joints_to_emphasize is not None
            and j in joints_to_emphasize
        ):
            xml_components.append(
                xml_joint_segment.format(
                    EMPHASIZED_JOINT_SIZE,
                    *human[j],
                    *color_joints_to_emphasize,
                )
            )
        else:
            xml_components.append(
                xml_joint_segment.format(
                    DEFAULT_JOINT_SIZE, *human[j], *human_color
                )
            )
        xml_components.append(
            xml_joint_segment.format(
                DEFAULT_JOINT_SIZE, *human[j_parent], *human_color
            )
        )

    if trajectory is not None:
        xml_components.extend(
            build_trajectory(
                trajectory, trajectory_color, trajectory_joint_size
            )
        )
    return xml_components


def build_trajectory(
    trajectory: np.ndarray,
    trajectory_color: Union[List[float], Tuple[float, ...]] = [
        30 / 255,
        30 / 255,
        8 / 255,
    ],
    trajectory_joint_size: float = 0.01,
) -> List[str]:
    direction_color = (120 / 255, 180 / 255, 120 / 255)
    xml_components = []

    trajectory_for_arrows = filter_trajectory(trajectory, 0.2)

    for joint in trajectory_for_arrows:
        for joint_1, joint_2 in zip(joint[:-1], joint[1:]):
            if np.all(joint_1 == joint_2):
                joint_1 -= 1e-3
            xml_components.append(
                xml_trajectory_joint_segment.format(
                    *joint_1, *joint_2, *direction_color
                )
            )
    for traj_1, traj_2 in zip(trajectory[:-2], trajectory[1:-1]):
        for joint_1, joint_2 in zip(traj_1, traj_2):
            if np.all(joint_1 == joint_2):
                joint_1 -= 1e-3
            xml_components.append(
                xml_trajectory_segment.format(
                    *joint_1, *joint_2, *trajectory_color
                )
            )
    return xml_components


def filter_trajectory(trajectory: np.ndarray, epsilon: float) -> np.ndarray:
    trajectory = trajectory[::-1]
    new_points = [[joint] for joint in trajectory[0]]
    for i, point in enumerate(trajectory[1:]):
        for j in range(point.shape[0]):
            distance = np.linalg.norm(new_points[j][-1] - point[j])
            if i == 0 or distance > epsilon:
                new_points[j].append(point[j])
    output = [joint[::-1] for joint in new_points]
    return output


def wrap_in_scene(
    components: List[str],
    camera_params: List[float],
    floor_z_coordinate: float,
    resolution: Tuple[int, int] = (640, 480),
    camera_target: Optional[List[float]] = None,
) -> List[str]:
    if camera_target is None:
        camera_target = [0, 0, 0]
    components.insert(
        0, xml_head.format(*camera_params, *camera_target, *resolution)
    )
    components.append(xml_tail.format(floor_z_coordinate))
    return components


def render(components: List[str], renderer_port: int) -> np.ndarray:
    xml_content = str.join("", components)

    result = requests.post(
        f"http://localhost:{renderer_port}/render", data=xml_content
    )
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img


def render_with_trajectories(
    components: List[str], renderer_port: int
) -> np.ndarray:
    xml_content = str.join("", components)

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        folder_path = directory_path / "scene"
        folder_path.mkdir()
        with open(folder_path / "scene.xml", "w") as f:
            f.write(xml_content)

        shutil.copy2(CONE_PATH.as_posix(), folder_path / "cone2.obj")

        with zipfile.ZipFile(directory_path / "scene.zip", "w") as zip_file:
            zip_file.write(
                (folder_path / "cone2.obj").as_posix(), "scene/cone2.obj"
            )
            zip_file.write(
                (folder_path / "scene.xml").as_posix(), "scene/scene.xml"
            )

        with open(directory_path / "scene.zip", "rb") as f:
            result = requests.post(
                f"http://localhost:{renderer_port}/render_zip",
                files={"zip": f},
            )
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img
