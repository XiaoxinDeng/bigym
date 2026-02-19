
import mujoco
import numpy as np
from mojo.elements import Geom

def make_state_buffer(physics):
    m = physics.model.ptr
    n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return np.zeros(n, dtype=np.float64)


def _snapshot(physics, buf):
    m = physics.model.ptr
    d = physics.data.ptr
    mujoco.mj_getState(m, d, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)

def _restore(physics, buf):
    m = physics.model.ptr
    d = physics.data.ptr
    mujoco.mj_setState(m, d, buf, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m, d)

def copy_state(env_src, env_dst, buf):
    # MuJoCo arrays
    _snapshot(env_src.mojo.physics, buf)
    _restore(env_dst.mojo.physics, buf)

    # Human script state
    hs = env_src.humanarms[0]
    hd = env_dst.humanarms[0]
    hd._CURRENT_TIME = hs._CURRENT_TIME
    hd._qpos_target[:] = hs._qpos_target
    hd._ctrl_target[:] = hs._ctrl_target

    # Floating-base controller buffers (BiGym-specific)
    fbs = env_src.robot.floating_base
    fbd = env_dst.robot.floating_base
    if fbs is not None and fbd is not None:
        fbd._accumulated_actions[:] = fbs._accumulated_actions
        fbd._last_action[:] = fbs._last_action


def get_robot_colliders(env):
    geoms = []
    for geom_mjcf in env.robot._body.mjcf.find_all("geom"):
        g = Geom(env.mojo, geom_mjcf)
        if g.is_collidable():
            geoms.append(g)
    return geoms

def _has_penetration(physics, colliders_1, colliders_2, pen_eps=0.0):
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)
    for c in physics.data.contact:
        if c.dist > pen_eps:
            continue
        if ((c.geom1 in ids_1 and c.geom2 in ids_2) or
            (c.geom2 in ids_1 and c.geom1 in ids_2)):
            return True
    return False

def get_hold_action_from_state(env):
    """
    Construct an action that holds the robot at its *current* state.
    This assumes the action is absolute joint positions (+ floating dofs if enabled)
    in the same order as env.action_mode expects.
    """
    # Most BiGym action modes expose a method to get the current action or target.
    # If yours has it, use it (preferred).
    if hasattr(env.action_mode, "get_action"):
        return env.action_mode.get_action()

    # Fallback: use qpos slices. You MUST match action ordering used by JointPositionActionMode.
    # If your action_mode exposes indices, use them.
    m = env.mojo.physics.model.ptr
    d = env.mojo.physics.data.ptr

    # Example: if action is exactly actuator joint positions, you can map joint names.
    # Replace with your env's joint list / mapping if available.
    joint_names = env.robot.joint_names  # if exists
    q = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = int(m.jnt_qposadr[jid])
        q.append(float(d.qpos[qadr]))
    return np.array(q, dtype=np.float32)


def freeze_human_if_contact(env, robot_colliders):
    human = env.humanarms[0]
    if _has_penetration(env.mojo.physics, human.colliders, robot_colliders, pen_eps=0.0):
        human._CURRENT_TIME -= env.get_dt()
        return True
    return False

def geom_ids_from_colliders(physics, colliders):
    # element_id for geoms in dm_control bind
    ids = np.array(physics.bind([c.mjcf for c in colliders]).element_id, dtype=np.int32)
    return np.unique(ids)


def will_collide_within(env_pred, horizon_s, action, human_geom_ids_pred, robot_geom_ids_pred, hit_thresh=0.01, step_dt=None):
    physics = env_pred.mojo.physics
    m = physics.model.ptr
    d = physics.data.ptr
    human = env_pred.humanarms[0]
    collision_t = None

    step_dt = float(step_dt or env_pred.get_dt())
    sub_steps = env_pred._sub_steps_count
    steps = int(np.ceil(horizon_s / step_dt))

    # early check
    dist0 = min_geom_distance(m, d, human_geom_ids_pred, robot_geom_ids_pred, distmax=hit_thresh)
    if dist0 < hit_thresh:
        collision_t = 0.0
        return True, 0.0

    for i in range(steps):
        human._on_step(step_dt)
        mujoco.mj_forward(m, d)

        env_pred.action_mode.step(action)
        for _ in range(sub_steps - 1):
            env_pred.mojo.step()

        dist = min_geom_distance(m, d, human_geom_ids_pred, robot_geom_ids_pred, distmax=hit_thresh)
        if dist < hit_thresh:
            collision_t = (i + 1) * step_dt
            return True, collision_t

    return False, collision_t


def get_dof_ids(env):
    m = env.mojo.physics.model.ptr
    pelvis = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_x"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_y"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_z"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/pelvis_rz"),
    ]
    pelvis_dofs = [int(m.jnt_dofadr[j]) for j in pelvis]

    fb_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "h1/h1_floating_base/h1_floating_base")
    fb_dof = int(m.jnt_dofadr[fb_j])

    return pelvis_dofs, [fb_dof]

def set_resistance(env, dof_ids, damping=None, frictionloss=None):
    m = env.mojo.physics.model.ptr
    if damping is not None:
        m.dof_damping[dof_ids] = damping
    if frictionloss is not None:
        m.dof_frictionloss[dof_ids] = frictionloss

def min_contact_dist(physics, colliders_1, colliders_2):
    ids_1 = set(physics.bind([c.mjcf for c in colliders_1]).element_id)
    ids_2 = set(physics.bind([c.mjcf for c in colliders_2]).element_id)
    md = None
    for c in physics.data.contact:
        if ((c.geom1 in ids_1 and c.geom2 in ids_2) or
            (c.geom2 in ids_1 and c.geom1 in ids_2)):
            md = c.dist if md is None else min(md, c.dist)
    return md


def has_contact_ids(physics, ids_a, ids_b, dist_margin=0.0):
    for c in physics.data.contact:
        if c.dist > dist_margin:
            continue
        if (c.geom1 in ids_a and c.geom2 in ids_b) or (c.geom2 in ids_a and c.geom1 in ids_b):
            return True, float(c.dist)
    return False, None

def min_rbound_separation(physics, ids_a, ids_b):
    """
    Conservative min separation using MuJoCo geom bounding spheres.
    Returns: (sep, ga, gb)
      sep = ||xa - xb|| - (ra + rb)
      sep > 0  : separated by ~sep meters (lower bound)
      sep <= 0 : bounding spheres overlap (very close / likely visual intersection)
    """
    m = physics.model.ptr
    d = physics.data.ptr

    best_sep = 1e9
    best_pair = (-1, -1)

    # Access arrays once
    xpos = d.geom_xpos          # (ngeom, 3)
    rbd  = m.geom_rbound        # (ngeom,)

    for ga in ids_a:
        xa = xpos[ga]
        ra = rbd[ga]
        for gb in ids_b:
            dx = xa - xpos[gb]
            sep = float(np.linalg.norm(dx) - (ra + rbd[gb]))
            if sep < best_sep:
                best_sep = sep
                best_pair = (ga, gb)

    return best_sep, best_pair[0], best_pair[1]


def min_geom_distance(model, data, ids_a, ids_b, distmax=0.2):
    # distmax: early-exit threshold (m). Set to your gate/hit threshold.
    frompos = np.zeros(3, dtype=np.float64)
    topos   = np.zeros(3, dtype=np.float64)
    best = distmax
    for ga in ids_a:
        for gb in ids_b:
            # MuJoCo C API: mj_geomDistance(m,d,ga,gb,distmax,frompos,topos) -> distance
            dist = mujoco.mj_geomDistance(model, data, int(ga), int(gb), best, frompos, topos)
            if dist < best:
                best = float(dist)
                if best <= 0.0:   # penetration
                    return best
    return best


def geom_ids_with_prefix(model, prefix: str):
    ids = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith(prefix):
            ids.add(gid)
    return ids

def collidable_ids_with_prefix(model, prefix):
    ids = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith(prefix):
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct != 0 and ca != 0:
            ids.add(gid)
    return ids



def collision_enabled_pairs(model, ids_a, ids_b):
    enabled = []
    for ga in ids_a:
        ct_a = int(model.geom_contype[ga])
        ca_a = int(model.geom_conaffinity[ga])
        if ct_a == 0 and ca_a == 0:
            continue

        for gb in ids_b:
            ct_b = int(model.geom_contype[gb])
            ca_b = int(model.geom_conaffinity[gb])
            if ct_b == 0 and ca_b == 0:
                continue

            # MuJoCo mask rule
            ok = (ct_a & ca_b) != 0 and (ct_b & ca_a) != 0
            if ok:
                enabled.append((ga, gb))
    return enabled

def get_arm_geo_ids(model):
    ids = set()
    names = ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        ids.add(gid)
    return ids

def zero_floating_base_velocity(env, jnames = ["h1/pelvis_x", "h1/pelvis_y", "h1/pelvis_z", "h1/pelvis_rz"]):
    """Hard-stop the floating base by zeroing its qvel DOFs."""
    phys = env.mojo.physics
    m = phys.model.ptr
    d = phys.data.ptr

    # DOFs for pelvis_x/y/z/rz joints (as you configured in action_mode)
    dofs = []
    for jn in jnames:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {jn}")
        dofs.append(int(m.jnt_dofadr[jid]))
    
    d.qvel[dofs] = 0.0
    d.qacc[dofs] = 0.0
    
    fb = getattr(env.robot, "floating_base", None)
    if fb is None:
        return
    if hasattr(fb, "_accumulated_actions"):
        fb._accumulated_actions[:] = 0.0
    if hasattr(fb, "_last_action"):
        fb._last_action[:] = 0.0

    # Make sure derived quantities are consistent
    mujoco.mj_forward(m, d)

def get_pelvis_qpos(env, jnames = ["h1/pelvis_x", "h1/pelvis_y", "h1/pelvis_z", "h1/pelvis_rz"]):
    phys = env.mojo.physics
    m = phys.model.ptr
    d = phys.data.ptr
    vals = []
    for jn in jnames:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = int(m.jnt_qposadr[jid])
        vals.append(float(d.qpos[qadr]))
    return np.array(vals, dtype=np.float64)


def has_contact_ids(physics, ids_a, ids_b, dist_margin=0.0):
    """True if MuJoCo generated a contact between sets (optionally allow small positive margin)."""
    d = physics.data.ptr
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > dist_margin:
            continue
        if (c.geom1 in ids_a and c.geom2 in ids_b) or (c.geom2 in ids_a and c.geom1 in ids_b):
            return True, float(c.dist), int(c.geom1), int(c.geom2)
    return False, None, -1, -1


def should_pause(phys_pred, human_ids, robot_ids,
                 broad_gate=-0.01,   # rbound overlap gate (negative)
                 contact_margin=0.002 # treat within 2mm as "contact"
                ):
    # Broad-phase: cheap, may false positive
    sep, ga, gb = min_rbound_separation(phys_pred, human_ids, robot_ids)

    if sep > broad_gate:
        return False, sep, None, ga, gb  # far enough => no pause

    # Narrow-phase: authoritative (contacts)
    hit, cdist, cg1, cg2 = has_contact_ids(phys_pred, human_ids, robot_ids, dist_margin=contact_margin)
    return hit, sep, cdist, cg1, cg2

def clamp_to_action_space(env, a):
    return np.clip(a, env.action_space.low, env.action_space.high)

def make_pause_hold_action(env, last_sent_action):
    """
    Bigym floating pelvis DOFs are action[:4] = [pelvis_x, pelvis_y, pelvis_z, pelvis_rz].
    These are typically *relative* commands (bounded small), so during pause set them to 0.

    For the remaining dims, keep the last command you sent (so the joint controller holds).
    """
    a = np.array(last_sent_action, dtype=np.float32).copy()
    a[:4] = 0.0
    return clamp_to_action_space(env, a)

def set_margins_for_sets(physics, gids, margin=0.06):
    m = physics.model.ptr
    for gid in gids:
        m.geom_margin[int(gid)] = float(margin)

def pair_min_contact_dist_between_sets(physics, ids_a, ids_b, dist_max=np.inf):
    """
    Returns (found, min_dist, ga, gb) for contacts between geom-id sets.
    - min_dist can be positive (within margin), ~0 (touch), or negative (penetration).
    - dist_max lets you ignore very-far margin contacts.
    """
    ids_a = set(map(int, ids_a))
    ids_b = set(map(int, ids_b))

    d = physics.data.ptr
    if d.ncon == 0:
        return False, None, -1, -1

    best = (float("inf"), -1, -1)
    for i in range(d.ncon):
        c = d.contact[i]
        dist = float(c.dist)

        if dist > dist_max:
            continue

        g1 = int(c.geom1)
        g2 = int(c.geom2)

        hit = (g1 in ids_a and g2 in ids_b) or (g2 in ids_a and g1 in ids_b)
        if not hit:
            continue

        if dist < best[0]:
            best = (dist, g1, g2)

    if best[1] == -1:
        return False, float("inf"), -1, -1
    return True, best[0], best[1], best[2]

class GeomHighlighter:
    def __init__(self, physics, visible_group=2, env=None, env_pred=None):
        self.physics = physics
        self.m = physics.model.ptr
        self.d = physics.data.ptr
        self.visible_group = int(visible_group)

        self._orig_rgba = {}
        self._orig_group = {}
        self._active = set()

        self.env = env
        self.env_pred = env_pred

    def _save_once(self, gid):
        if gid not in self._orig_rgba:
            self._orig_rgba[gid] = self.m.geom_rgba[gid].copy()
            self._orig_group[gid] = int(self.m.geom_group[gid])

    def highlight(self, gids, rgba=(1, 0, 0, 1), force_visible=True):
        rgba = np.asarray(rgba, dtype=np.float32)
        for gid in gids:
            gid = int(gid)
            if gid < 0 or gid >= self.m.ngeom:
                continue
            self._save_once(gid)
            self.m.geom_rgba[gid] = rgba
            if force_visible:
                self.m.geom_group[gid] = self.visible_group
            self._active.add(gid)
        mujoco.mj_forward(self.m, self.d)

    def clear(self):
        for gid in list(self._active):
            if gid in self._orig_rgba:
                self.m.geom_rgba[gid] = self._orig_rgba[gid]
            if gid in self._orig_group:
                self.m.geom_group[gid] = self._orig_group[gid]
        self._active.clear()
        mujoco.mj_forward(self.m, self.d)

    def _gid_from_name(self, name: str) -> int:
        if not name:
            return -1
        try:
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)
        except Exception:
            return -1

    def _all_visual_geoms_of_body(self, body_id: int):
        # heuristic: visual meshes usually group=2 and contype=0/ca=0, but group is enough to see them
        gids = []
        for gid in range(self.m.ngeom):
            if int(self.m.geom_bodyid[gid]) == int(body_id):
                # Prefer visual group geoms if present; else include all on that body
                gids.append(gid)
        return gids

    def highlight_pred_contact_pair(self, cg1_pred, cg2_pred, rgba=(1,0,0,1),
                                   force_visible=True, highlight_body_visual=True):
        """
        cg*_pred are geom ids in env_pred. We map via name into THIS model,
        then (optionally) highlight all geoms on those bodies so you actually see it.
        """
        if self.env_pred is None:
            raise ValueError("env_pred not set; cannot map pred geom ids -> names -> env ids.")

        mp = self.env_pred.mojo.physics.model.ptr
        name1 = mujoco.mj_id2name(mp, mujoco.mjtObj.mjOBJ_GEOM, int(cg1_pred)) or ""
        name2 = mujoco.mj_id2name(mp, mujoco.mjtObj.mjOBJ_GEOM, int(cg2_pred)) or ""

        g1 = self._gid_from_name(name1)
        g2 = self._gid_from_name(name2)

        if not highlight_body_visual:
            self.highlight([g1, g2], rgba=rgba, force_visible=force_visible)
            return name1, name2, g1, g2

        gids = []
        for g in [g1, g2]:
            if g >= 0:
                bid = int(self.m.geom_bodyid[g])
                gids.extend(self._all_visual_geoms_of_body(bid))

        # de-dup
        gids = list(dict.fromkeys(gids))
        self.highlight(gids, rgba=rgba, force_visible=force_visible)
        return name1, name2, g1, g2
