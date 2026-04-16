#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def find_first_success_demo(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    demos = manifest["demos"] if isinstance(manifest, dict) and "demos" in manifest else manifest

    for item in demos:
        success = item.get("success") or item.get("is_success") or item.get("task_success")
        if success:
            return item
    raise RuntimeError("No successful demo found in manifest.")


def consecutive_runs(mask: np.ndarray):
    runs = []
    run = 0
    for v in mask:
        if v:
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    return runs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--delta_threshold", type=float, default=1e-3)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    item = find_first_success_demo(manifest_path)

    print("=" * 80)
    print("Selected demo from manifest:")
    print(json.dumps(item, indent=2, ensure_ascii=False))

    demo_path = Path(item.get("target_path") or item.get("file") or item.get("path"))
    if not demo_path.is_absolute():
        demo_path = manifest_path.parent / demo_path

    print("\nResolved demo path:")
    print(demo_path)

    data = load_file(str(demo_path))
    keys = list(data.keys())

    print("\nAvailable keys:")
    print(keys)

    actions = data["info_demo_action"]   # [T, A]
    paused = data["info_paused"].reshape(-1).astype(np.int32) if "info_paused" in data else None
    mode_label = data["info_mode_label"].reshape(-1) if "info_mode_label" in data else None

    if actions.ndim == 1:
        actions = actions[:, None]
    elif actions.ndim >= 3:
        actions = actions.reshape(-1, actions.shape[-1])

    T = actions.shape[0]

    # action norm
    action_norm = np.linalg.norm(actions, axis=-1)

    # delta action
    delta = np.linalg.norm(actions[1:] - actions[:-1], axis=-1)   # [T-1]
    delta_with_zero = np.concatenate([[0.0], delta], axis=0)      # 对齐到 [T]

    print("\n" + "=" * 80)
    print("ACTION NORM STATS")
    print("=" * 80)
    print(f"shape            : {actions.shape}")
    print(f"norm min/max     : {action_norm.min():.6f} / {action_norm.max():.6f}")
    print(f"norm mean/std    : {action_norm.mean():.6f} / {action_norm.std():.6f}")
    print(f"first 20 norms   : {np.array2string(action_norm[:20], precision=4)}")

    print("\n" + "=" * 80)
    print("DELTA ACTION STATS")
    print("=" * 80)
    print(f"delta min/max    : {delta_with_zero.min():.6f} / {delta_with_zero.max():.6f}")
    print(f"delta mean/std   : {delta_with_zero.mean():.6f} / {delta_with_zero.std():.6f}")
    print(f"first 20 deltas  : {np.array2string(delta_with_zero[:20], precision=6)}")

    hold_mask = delta_with_zero < args.delta_threshold
    hold_ratio = hold_mask.mean()
    hold_runs = consecutive_runs(hold_mask)
    print(f"hold ratio       : {hold_ratio:.4f} (delta < {args.delta_threshold})")
    print(f"longest hold run : {max(hold_runs) if hold_runs else 0}")
    print(f"mean hold run    : {np.mean(hold_runs):.2f}" if hold_runs else "mean hold run    : 0.00")

    if paused is not None:
        print("\n" + "=" * 80)
        print("PAUSE STATS")
        print("=" * 80)
        paused_ratio = paused.mean()
        paused_runs = consecutive_runs(paused.astype(bool))
        print(f"paused ratio     : {paused_ratio:.4f} ({paused.sum()}/{len(paused)})")
        print(f"longest pause run: {max(paused_runs) if paused_runs else 0}")
        print(f"mean pause run   : {np.mean(paused_runs):.2f}" if paused_runs else "mean pause run   : 0.00")

        paused_delta = delta_with_zero[paused == 1]
        active_delta = delta_with_zero[paused == 0]

        if len(paused_delta) > 0:
            print(f"paused delta mean/std : {paused_delta.mean():.6f} / {paused_delta.std():.6f}")
        if len(active_delta) > 0:
            print(f"active delta mean/std : {active_delta.mean():.6f} / {active_delta.std():.6f}")

    if mode_label is not None:
        print("\n" + "=" * 80)
        print("MODE LABEL STATS")
        print("=" * 80)
        unique, counts = np.unique(mode_label, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"mode {u}: {c} ({c / len(mode_label):.4f})")


if __name__ == "__main__":
    main()