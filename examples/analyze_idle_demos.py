#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def find_first_success_demo(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # BiGym manifest 常见结构：
    # [
    #   {"file": "...", "success": true, ...},
    #   ...
    # ]

    if isinstance(manifest, dict) and "demos" in manifest:
        demos = manifest["demos"]
    else:
        demos = manifest

    for item in demos:
        success = item.get("success") or item.get("is_success") or item.get("task_success")
        if success:
            return item

    raise RuntimeError("No successful demo found in manifest.")


def load_actions(path: Path):
    data = load_file(str(path))
    keys = list(data.keys())

    action_key = None
    for k in ["action", "actions", "demo_action"]:
        if k in data:
            action_key = k
            break

    if action_key is None:
        for k in keys:
            if "action" in k.lower() and "mode" not in k.lower():
                action_key = k
                break

    if action_key is None:
        raise RuntimeError(f"No action key found. Keys: {keys}")

    actions = data[action_key]

    # flatten to [T, A]
    if actions.ndim == 1:
        actions = actions[:, None]
    elif actions.ndim >= 3:
        actions = actions.reshape(-1, actions.shape[-1])

    return actions, action_key, keys


def analyze(actions, idle_threshold=1e-2):
    norms = np.linalg.norm(actions, axis=-1)

    idle_mask = norms < idle_threshold
    idle_ratio = float(idle_mask.mean())

    # 连续 idle 段
    runs = []
    run = 0
    for v in idle_mask:
        if v:
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)

    longest_run = max(runs) if runs else 0
    mean_run = float(np.mean(runs)) if runs else 0.0

    return {
        "num_steps": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "idle_ratio": idle_ratio,
        "idle_steps": int(idle_mask.sum()),
        "longest_idle_run": int(longest_run),
        "mean_idle_run": mean_run,
        "first_20_norms": norms[:20],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--idle_threshold", type=float, default=1e-2)

    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    # 1️⃣ 找到第一个 success demo
    item = find_first_success_demo(manifest_path)

    print("=" * 80)
    print("Selected demo from manifest:")
    print(json.dumps(item, indent=2, ensure_ascii=False))

    # 2️⃣ 构建 safetensor 路径
    file_path = item.get("target_path")

    if file_path is None:
        raise RuntimeError("Manifest item has no 'file' or 'path' field.")

    demo_path = Path(file_path)

    # 如果是相对路径 → 相对于 manifest
    if not demo_path.is_absolute():
        demo_path = manifest_path.parent / demo_path

    print("\nResolved demo path:")
    print(demo_path)

    # 3️⃣ 加载 actions
    actions, action_key, keys = load_actions(demo_path)

    print("\nAvailable keys:")
    print(keys)

    print(f"\nUsing action key: {action_key}")
    print(f"Action shape: {actions.shape}")

    # 4️⃣ 分析
    stats = analyze(actions, idle_threshold=args.idle_threshold)

    print("\n" + "=" * 80)
    print("ANALYSIS RESULT")
    print("=" * 80)

    print(f"num_steps        : {stats['num_steps']}")
    print(f"action_dim       : {stats['action_dim']}")
    print(
        f"norm stats       : min={stats['norm_min']:.6f}, "
        f"max={stats['norm_max']:.6f}, "
        f"mean={stats['norm_mean']:.6f}, "
        f"std={stats['norm_std']:.6f}"
    )
    print(
        f"idle ratio       : {stats['idle_ratio']:.4f} "
        f"({stats['idle_steps']}/{stats['num_steps']})"
    )
    print(
        f"idle run         : longest={stats['longest_idle_run']}, "
        f"mean={stats['mean_idle_run']:.2f}"
    )
    print(
        f"first 20 norms   : {np.array2string(stats['first_20_norms'], precision=4)}"
    )


if __name__ == "__main__":
    main()