import pybullet as p


class RoverPhysics:
    def __init__(self, rover_id):
        self.rover_id = rover_id

        self.wheel_joints = []
        num_joints = p.getNumJoints(rover_id)

        for i in range(num_joints):
            info = p.getJointInfo(rover_id, i)
            name = info[1].decode("utf-8")

            if "wheel_joint" in name:
                self.wheel_joints.append(i)

    def apply_action(self, left_vel, right_vel):
        # first 3 wheels = left, last 3 = right
        for i, joint in enumerate(self.wheel_joints):
            vel = left_vel if i < 3 else right_vel

            p.setJointMotorControl2(
                bodyUniqueId=self.rover_id,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=20
            )