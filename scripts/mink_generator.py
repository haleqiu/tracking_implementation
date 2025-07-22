import mink
import mujoco
import mujoco.viewer
import numpy as np
from mink import SE3, SO3
from robot_descriptions import g1_mj_description

N_STEPS = 15000
FPS = 50.0

COM_POS = (0.05, 0.0, 0.5)
LEFT_HAND_POS = (0.3, 0.05, 0.75)
RIGT_HAND_POS = LEFT_HAND_POS[0], -LEFT_HAND_POS[1], LEFT_HAND_POS[2]
HAND_RANGE = (0.2, 0.3, 0.75)

RESAMPLE_TIME_RANGE = (0.5, 1.5)
VELOCITY_SAFETY_FACTOR = 0.02

FEET_SITES = ["left_foot", "right_foot"]
HAND_SITES = ["left_ee", "right_ee"]


def construrct_spec():
    spec = mujoco.MjSpec.from_file(g1_mj_description.MJCF_PATH)

    for geom in spec.geoms:
        if geom.classname.name in ["collision", "foot"]:
            spec.delete(geom)

    def add_capsule(spec, body_name: str, geom_name: str, **overrides):
        COMMON = dict(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=(0.07,) * 3,
            contype=1,
            conaffinity=1,
            group=3,
            rgba=(0.2, 0.6, 0.2, 0.3),
        )
        spec.body(body_name).add_geom(**(COMMON | {"name": geom_name} | overrides))

    CAPSULES = [
        ("right_wrist_yaw_link", "right_wrist_collider",
         dict(fromto=(0.05, 0, 0, 0.18, 0, 0))),
        ("left_wrist_yaw_link", "left_wrist_collider",
         dict(fromto=(0.05, 0, 0, 0.18, 0, 0))),
        ("left_hip_roll_link", "left_thigh_collision",
         dict(fromto=(0.02, 0, 0, 0.02, 0, -0.2))),
        ("right_hip_roll_link", "right_thigh_collision",
         dict(fromto=(0.02, 0, 0, 0.02, 0, -0.2))),
        ("torso_link", "torso_collision",
         dict(size=(0.11,) * 3, fromto=(0, 0, 0.1, 0, 0, 0.38))),
        ("left_shoulder_yaw_link", "left_shoulder_yaw_collision",
         dict(size=(0.045, 0.07, 0), pos=(0.0, 0.0, -0.015), quat=(0, 1, 0, 0))),
        ("right_shoulder_yaw_link", "right_shoulder_yaw_collision",
         dict(size=(0.045, 0.07, 0), pos=(0.0, 0.0, -0.015), quat=(0, 1, 0, 0))),
    ]

    for body, name, extra in CAPSULES:
        add_capsule(spec, body, name, **extra)

    # Add hand sites.
    for side in ["left", "right"]:
        body = spec.body(f"{side}_wrist_yaw_link")
        site = body.add_site()
        site.name = f"{side}_ee"
        cls = spec.find_default("g1")
        site.classname = cls
        site.pos[0] = 0.08

    # Add mocap bodies.
    sites = FEET_SITES + HAND_SITES
    for site in sites:
        mocap_body = spec.worldbody.add_body()
        mocap_body.name = site + "_target"
        mocap_body.mocap = True
        mocap_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.05, 0.05, 0.05),
            rgba=(0.6, 0.2, 0.2, 0.6),
            contype=0,
            conaffinity=0,
        )

    key0 = spec.add_key()
    key0.name = "teleop"
    key0.qpos = [
        0, 0, 0.755,
        1, 0, 0, 0,
        -0.312, 0, 0, 0.669, -0.363, 0,
        -0.312, 0, 0, 0.669, -0.363, 0,
        0, 0, 0,
        0.2, 0.2, 0, 0, 0, 0, 0,
        0.2, -0.2, 0, 0, 0, 0, 0
    ]

    def workspace_site(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]):
        site = spec.worldbody.add_site()
        site.name = name
        site.pos = pos
        site.size = size
        site.group = 4
        site.rgba = (0.2, 0.6, 0.2, 0.2)
        site.type = mujoco.mjtGeom.mjGEOM_BOX

    workspace_site("left_workspace", LEFT_HAND_POS, HAND_RANGE)
    workspace_site("right_workspace", RIGT_HAND_POS, HAND_RANGE)

    return spec


if __name__ == "__main__":
    spec = construrct_spec()
    model = spec.compile()

    configuration = mink.Configuration(model)

    tasks = [
        com_task := mink.ComTask(cost=[1e2, 1e2, 1.0]),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=0.25,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
    ]
    feet_tasks = []
    for foot in FEET_SITES:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1e2,
            orientation_cost=1e2,
            lm_damping=0.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in HAND_SITES:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    max_velocities = {
        # --- legs ---
        "left_ankle_pitch_joint": 37.0,
        "left_ankle_roll_joint": 37.0,
        "left_hip_pitch_joint": 32.0,
        "left_hip_roll_joint": 20.0,
        "left_hip_yaw_joint": 32.0,
        "left_knee_joint": 20.0,

        "right_ankle_pitch_joint": 37.0,
        "right_ankle_roll_joint": 37.0,
        "right_hip_pitch_joint": 32.0,
        "right_hip_roll_joint": 20.0,
        "right_hip_yaw_joint": 32.0,
        "right_knee_joint": 20.0,

        # --- waist ---
        "waist_pitch_joint": 37.0,
        "waist_roll_joint": 37.0,
        "waist_yaw_joint": 32.0,

        # --- left arm ---
        "left_shoulder_pitch_joint": 37.0,
        "left_shoulder_roll_joint": 37.0,
        "left_shoulder_yaw_joint": 37.0,
        "left_elbow_joint": 37.0,
        "left_wrist_roll_joint": 37.0,
        "left_wrist_pitch_joint": 22.0,
        "left_wrist_yaw_joint": 22.0,

        # --- right arm ---
        "right_shoulder_pitch_joint": 37.0,
        "right_shoulder_roll_joint": 37.0,
        "right_shoulder_yaw_joint": 37.0,
        "right_elbow_joint": 37.0,
        "right_wrist_roll_joint": 37.0,
        "right_wrist_pitch_joint": 22.0,
        "right_wrist_yaw_joint": 22.0,
    }
    for name, vel in max_velocities.items():
        max_velocities[name] = vel * VELOCITY_SAFETY_FACTOR

    velocity_limit = mink.VelocityLimit(model, max_velocities)

    # Collision avoidance.
    l_geoms = [model.geom("left_wrist_collider").id]
    r_geoms = [model.geom("right_wrist_collider").id]
    l_hip_pitch_geoms = mink.get_body_geom_ids(model, model.body("left_hip_roll_link").id)
    r_hip_pitch_geoms = mink.get_body_geom_ids(model, model.body("right_hip_roll_link").id)
    l_hip_yaw_geoms = mink.get_body_geom_ids(model, model.body("left_hip_yaw_link").id)
    r_hip_yaw_geoms = mink.get_body_geom_ids(model, model.body("right_hip_yaw_link").id)
    torso_geom = model.geom("torso_collision").id
    left_shoulder_yaw_geom = model.geom("left_shoulder_yaw_collision").id
    right_shoulder_yaw_geom = model.geom("right_shoulder_yaw_collision").id

    # from ipdb import set_trace; set_trace()
    collision_pairs = [
        (l_geoms, l_hip_pitch_geoms + l_hip_yaw_geoms),
        (r_geoms, r_hip_pitch_geoms + r_hip_yaw_geoms),
        (l_geoms, r_hip_pitch_geoms + r_hip_yaw_geoms),
        (r_geoms, l_hip_pitch_geoms + l_hip_yaw_geoms),
        (l_geoms, r_geoms),
        ([torso_geom], [right_shoulder_yaw_geom, left_shoulder_yaw_geom]),
        ([torso_geom], l_geoms + r_geoms),
    ]

    limits = [
        mink.ConfigurationLimit(model),
        velocity_limit,
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.5,
        )
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in FEET_SITES]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in HAND_SITES]


    class PoseCommand:
        def __init__(self, name, center, range, resample_time_range=RESAMPLE_TIME_RANGE):
            self.name = name
            self.start_pose = SE3.from_rotation_and_translation(
                rotation=SO3.from_matrix(data.site(self.name).xmat.reshape(3, 3)),
                translation=data.site(self.name).xpos)
            self.end_pose = self.start_pose
            self.center = center
            self.range = range
            self.resample_time = resample_time_range
            self.start_time = 0.0
            self.end_time = 0.0

            self.resample()

        def resample(self):
            self.start_time = self.end_time
            self.end_time = self.start_time + np.random.uniform(*self.resample_time)
            self.start_pose = self.end_pose
            self.end_pose = SE3.from_rotation_and_translation(
                rotation=SO3.identity(),
                translation=np.array([
                    self.center[0] + np.random.uniform(-self.range[0], self.range[0]),
                    self.center[1] + np.random.uniform(-self.range[1], self.range[1]),
                    self.center[2] + np.random.uniform(-self.range[2], self.range[2]),
                ])
            )

        @staticmethod
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def __call__(self, t):
            if t > self.end_time:
                self.resample()
            alpha = (t - self.start_time) / (self.end_time - self.start_time)
            shaped_alpha = self.sigmoid(10 * (alpha - 0.5))
            return self.start_pose.interpolate(self.end_pose, shaped_alpha)


    left_hand_command = PoseCommand("left_ee", LEFT_HAND_POS, HAND_RANGE, RESAMPLE_TIME_RANGE)
    right_hand_command = PoseCommand("right_ee", RIGT_HAND_POS, HAND_RANGE, RESAMPLE_TIME_RANGE)


    def update_mocap_pose(d: mujoco.MjData, t: float) -> None:
        d.mocap_pos[hands_mid[0]] = left_hand_command(t).translation()
        d.mocap_pos[hands_mid[1]] = right_hand_command(t).translation()


    with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(HAND_SITES, FEET_SITES):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        posture_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        com_task.set_target(COM_POS)

        qpos_log = np.empty((N_STEPS, model.nq))

        dt = 0.005

        t = 0.0
        while viewer.is_running() and int(t * FPS) < N_STEPS - 1:
            update_mocap_pose(data, t)

            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            vel = mink.solve_ik(
                configuration, tasks, dt, solver, 1e-4, limits=limits
            )
            configuration.integrate_inplace(vel, dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            t += dt

            qpos = data.qpos.copy()
            qpos[3:7] = np.roll(qpos[3:7], -1)  # Re-order quat (wxyz) to (xyzw).
            qpos_log[int(t * FPS)] = qpos

        np.savetxt("reach.csv", qpos_log, delimiter=",")
