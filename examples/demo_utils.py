
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

def clamp_action(env, a):
    a = np.asarray(a, dtype=np.float32)
    lo = env.action_space.low.astype(np.float32)
    hi = env.action_space.high.astype(np.float32)
    return np.minimum(np.maximum(a, lo), hi)

def disable_arm_collisions(physics):
    m = physics.model.ptr
    # arm geom names
    arm_geom_names = ["cylinder_arm/upperarm_geom", "cylinder_arm/forearm_geom"]
    for name in arm_geom_names:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"Arm geom not found: {name}")
        m.geom_contype[gid] = 0
        m.geom_conaffinity[gid] = 0
        # optional: zero margin so it doesn't generate margin contacts either
        m.geom_margin[gid] = 0.0

def copy_state(env_src, env_dst, buf):
    _snapshot(env_src.mojo.physics, buf)
    _restore(env_dst.mojo.physics, buf)

    # Human script state
    hs = env_src.humanarms[0]
    hd = env_dst.humanarms[0]

    # ---- core time/control ----
    hd._CURRENT_TIME = hs._CURRENT_TIME
    hd._mode = hs._mode
    hd._qpos_target[:] = hs._qpos_target
    hd._ctrl_target[:] = hs._ctrl_target

    # ---- walking state (OU) ----
    if getattr(hs, "_walk_center_xy", None) is not None:
        hd._walk_center_xy = hs._walk_center_xy.copy()
    if getattr(hs, "_walk_xy", None) is not None:
        hd._walk_xy = hs._walk_xy.copy()
    if getattr(hs, "_walk_v", None) is not None:
        hd._walk_v = hs._walk_v.copy()

    # ---- joint smoothing ----
    if getattr(hs, "_qpos_filt", None) is not None:
        hd._qpos_filt = hs._qpos_filt.copy()

    # ---- noise state ----
    hd._next_resample_t = hs._next_resample_t
    if getattr(hs, "_noise_freqs", None) is not None:
        hd._noise_freqs = hs._noise_freqs.copy()
    if getattr(hs, "_noise_phases", None) is not None:
        hd._noise_phases = hs._noise_phases.copy()
    if getattr(hs, "_noise_amps", None) is not None:
        hd._noise_amps = hs._noise_amps.copy()

    # If you use blend variables:
    if getattr(hs, "_noise_old", None) is not None:
        hd._noise_old = tuple(x.copy() for x in hs._noise_old)
    if getattr(hs, "_noise_new", None) is not None:
        hd._noise_new = tuple(x.copy() for x in hs._noise_new)
    hd._noise_blend_t0 = getattr(hs, "_noise_blend_t0", 0.0)

    # ---- RNG state (critical for OU) ----
    if getattr(hs, "_rng", None) is not None:
        if getattr(hd, "_rng", None) is None:
            hd._rng = np.random.default_rng()
        hd._rng.bit_generator.state = hs._rng.bit_generator.state

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

def make_pause_hold_action_hybrid(env, action_joint_jids, last_sent_action):
    """
    Hybrid hold for your action definition:
      - dims 0..3 : floating base deltas -> 0
      - dims 4..13: absolute joint targets -> current qpos of mapped joints
      - dims 14..15: gripper -> keep last commanded (or map similarly if needed)
    """
    phys = env.mojo.physics
    m = phys.model.ptr
    d = phys.data.ptr

    a = np.array(last_sent_action, dtype=np.float32).copy()

    # floating base deltas: hold by sending zero delta
    a[0:4] = 0.0

    # absolute joint targets: set to CURRENT qpos
    for i, jid in enumerate(action_joint_jids):
        dim = 4 + i
        qadr = int(m.jnt_qposadr[int(jid)])
        a[dim] = float(d.qpos[qadr])

    # gripper dims (14,15): keep last command (stable)
    # a[14:16] already from last_sent_action

    return np.clip(a, env.action_space.low, env.action_space.high)

def build_action_joint_mapping_from_ranges(env, prefix="h1/", start_dim=4, end_dim=14, tol=5e-3):
    """
    Map action dims [start_dim:end_dim) to MuJoCo hinge/slide joints by matching (low, high)
    against joint ranges. Returns a list of joint IDs (len = end_dim-start_dim).
    """
    phys = env.mojo.physics
    m = phys.model.ptr

    lows = np.asarray(env.action_space.low, dtype=np.float64)
    highs = np.asarray(env.action_space.high, dtype=np.float64)

    # collect candidate joints with ranges
    candidates = []
    for jid in range(int(m.njnt)):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if not jname.startswith(prefix):
            continue
        jtype = int(m.jnt_type[jid])
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        r = m.jnt_range[jid].copy()
        candidates.append((jid, jname, float(r[0]), float(r[1])))

    mapping = []
    used = set()

    for dim in range(start_dim, end_dim):
        lo, hi = float(lows[dim]), float(highs[dim])

        # find best range match among unused joints
        best = None
        best_err = 1e9
        for jid, jname, rlo, rhi in candidates:
            if jid in used:
                continue
            err = abs(rlo - lo) + abs(rhi - hi)
            if err < best_err:
                best_err = err
                best = (jid, jname, rlo, rhi)

        if best is None or best_err > tol:
            raise RuntimeError(
                f"Could not reliably match action dim {dim} range [{lo},{hi}] "
                f"to any {prefix} hinge/slide joint range (best_err={best_err})."
            )

        jid, jname, rlo, rhi = best
        used.add(jid)
        mapping.append(jid)

    # helpful print
    print("[ACTION→JOINT MAP]")
    for k, jid in enumerate(mapping):
        dim = start_dim + k
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        rlo, rhi = m.jnt_range[jid]
        print(f"  dim {dim:2d} -> {jname:40s}  range=[{rlo:.4f},{rhi:.4f}]")

    return mapping

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