import xml.etree.ElementTree as ET
from dataclasses import dataclass
from config import URDF_FILE, JOINT_COUNT

@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: list
    rpy: list
    axis: list
    lower: float | None
    upper: float | None

def nums(text, default):
    return default if text is None else [float(x) for x in text.split()]

def load_robot_model(urdf_file= "jetcobot.urdf"):
    root = ET.parse(urdf_file).getroot()
    joints = []

    for j in root.findall("joint"):
        joint_type = j.attrib.get("type")

        # Use only the 6 main arm revolute joints.
        if joint_type != "revolute":
            continue

        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")

        joints.append(Joint(
            name=j.attrib["name"],
            joint_type=joint_type,
            parent=j.find("parent").attrib["link"],
            child=j.find("child").attrib["link"],
            xyz=nums(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0]),
            rpy=nums(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0]),
            axis=nums(axis.attrib.get("xyz") if axis is not None else None, [0, 0, 1]),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        ))

        if len(joints) == JOINT_COUNT:
            break

    return joints

if __name__ == "__main__":
    joints = load_robot_model()

    for joint in joints:
        print(joint.name)
        print(joint)
        print("-" * 40)

    print("Total arm joints:", len(joints))
