
import mujoco
import numpy as np

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


def freeze_human_if_contact(env, robot_colliders):
    human = env.humanarms[0]
    if _has_penetration(env.mojo.physics, human.colliders, robot_colliders, pen_eps=0.0):
        human._CURRENT_TIME -= env.get_dt()
        return True
    return False



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