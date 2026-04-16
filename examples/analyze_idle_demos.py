#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from safetensors.numpy import load_file


ACTION_KEY_CANDIDATES = [
    "info_demo_action",
    "action",
    "actions",
    "demo_action",
    "expert_action",
]


def load_manifest(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["demos"] if isinstance(manifest, dict) and "demos" in manifest else manifest


def resolve_path(raw_path: str, manifest_path: Path):
    p = Path(raw_path)
    if not p.is_absolute():
        p = manifest_path.parent / p
    return p


def find_action_key(data):
    for k in ACTION_KEY_CANDIDATES:
        if k in data:
            return k
    for k in data.keys():
        lk = k.lower()
        if "action" in lk and "mode" not in lk:
            return k
    raise RuntimeError(f"No action-like key found. Keys={list(data.keys())}")


def flatten_actions(actions: np.ndarray):
    if actions.ndim == 1:
        return actions[:, None]
    if actions.ndim >= 3:
        return actions.reshape(-1, actions.shape[-1])
    return actions


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


def analyze_safetensor(path: Path, delta_threshold: float = 1e-3):
    data = load_file(str(path))
    keys = list(data.keys())

    action_key = find_action_key(data)
    actions = flatten_actions(np.asarray(data[action_key]))
    T, A = actions.shape

    action_norm = np.linalg.norm(actions, axis=-1)
    delta = np.linalg.norm(actions[1:] - actions[:-1], axis=-1)
    delta_with_zero = np.concatenate([[0.0], delta], axis=0)

    hold_mask = delta_with_zero < delta_threshold
    hold_runs = consecutive_runs(hold_mask)

    result = {
        "file": str(path),
        "action_key": action_key,
        "num_steps": int(T),
        "action_dim": int(A),
        "action_norm_mean": float(action_norm.mean()),
        "action_norm_std": float(action_norm.std()),
        "action_norm_min": float(action_norm.min()),
        "action_norm_max": float(action_norm.max()),
        "delta_mean": float(delta_with_zero.mean()),
        "delta_std": float(delta_with_zero.std()),
        "delta_min": float(delta_with_zero.min()),
        "delta_max": float(delta_with_zero.max()),
        "hold_ratio": float(hold_mask.mean()),
        "longest_hold_run": int(max(hold_runs) if hold_runs else 0),
        "mean_hold_run": float(np.mean(hold_runs) if hold_runs else 0.0),
        "first_10_norms": [float(x) for x in action_norm[:10]],
        "first_10_deltas": [float(x) for x in delta_with_zero[:10]],
        "keys": keys,
    }

    if "info_paused" in data:
        paused = np.asarray(data["info_paused"]).reshape(-1).astype(np.int32)
        paused_runs = consecutive_runs(paused.astype(bool))
        result["paused_ratio"] = float(paused.mean())
        result["paused_steps"] = int(paused.sum())
        result["longest_pause_run"] = int(max(paused_runs) if paused_runs else 0)
        paused_delta = delta_with_zero[paused == 1]
        active_delta = delta_with_zero[paused == 0]
        result["paused_delta_mean"] = float(paused_delta.mean()) if len(paused_delta) > 0 else None
        result["active_delta_mean"] = float(active_delta.mean()) if len(active_delta) > 0 else None
    else:
        result["paused_ratio"] = None
        result["paused_steps"] = None
        result["longest_pause_run"] = None
        result["paused_delta_mean"] = None
        result["active_delta_mean"] = None

    if "info_mode_label" in data:
        mode = np.asarray(data["info_mode_label"]).reshape(-1)
        uniq, cnt = np.unique(mode, return_counts=True)
        result["mode_distribution"] = {int(u): int(c) for u, c in zip(uniq, cnt)}
    else:
        result["mode_distribution"] = None

    return result


def summarize_group(rows, name):
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"group": name, "count": 0}

    def arr(k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return np.asarray(vals, dtype=np.float64) if vals else np.asarray([], dtype=np.float64)

    hold = arr("hold_ratio")
    delta = arr("delta_mean")
    paused = arr("paused_ratio")

    return {
        "group": name,
        "count": len(rows),
        "hold_ratio_mean": float(hold.mean()) if len(hold) else None,
        "hold_ratio_std": float(hold.std()) if len(hold) else None,
        "hold_ratio_min": float(hold.min()) if len(hold) else None,
        "hold_ratio_max": float(hold.max()) if len(hold) else None,
        "delta_mean_mean": float(delta.mean()) if len(delta) else None,
        "delta_mean_std": float(delta.std()) if len(delta) else None,
        "paused_ratio_mean": float(paused.mean()) if len(paused) else None,
        "num_with_pause": int(np.sum(paused > 0)) if len(paused) else 0,
    }


def print_one(title, r):
    if r is None:
        print(f"{title}: None")
        return
    print(f"{title}:")
    print(f"  file              : {r['file']}")
    print(f"  action_key        : {r['action_key']}")
    print(f"  num_steps         : {r['num_steps']}")
    print(f"  action_dim        : {r['action_dim']}")
    print(f"  action_norm mean  : {r['action_norm_mean']:.6f} ± {r['action_norm_std']:.6f}")
    print(f"  delta mean        : {r['delta_mean']:.6f} ± {r['delta_std']:.6f}")
    print(f"  hold_ratio        : {r['hold_ratio']:.4f}")
    print(f"  longest_hold_run  : {r['longest_hold_run']}")
    if r["paused_ratio"] is not None:
        print(f"  paused_ratio      : {r['paused_ratio']:.4f}")
        print(f"  paused_steps      : {r['paused_steps']}")
        print(f"  longest_pause_run : {r['longest_pause_run']}")
        print(f"  paused_delta_mean : {r['paused_delta_mean']}")
        print(f"  active_delta_mean : {r['active_delta_mean']}")
    print(f"  mode_distribution : {r['mode_distribution']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--delta_threshold", type=float, default=1e-3)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    items = load_manifest(manifest_path)

    success_items = [x for x in items if int(x.get("success", 0)) == 1]
    print(f"Total manifest entries   : {len(items)}")
    print(f"Successful manifest rows : {len(success_items)}")
    print(f"delta_threshold          : {args.delta_threshold}")

    paired_results = []
    target_rows = []
    source_rows = []

    for i, item in enumerate(success_items):
        print("\n" + "=" * 120)
        print(f"[{i+1}/{len(success_items)}] uuid={item.get('uuid')} source_uuid={item.get('source_uuid')}")

        target_result = None
        source_result = None

        target_path = item.get("target_path")
        if target_path:
            try:
                target_result = analyze_safetensor(resolve_path(target_path, manifest_path), args.delta_threshold)
                target_rows.append(target_result)
            except Exception as e:
                target_result = {"error": str(e), "file": str(target_path)}
                print(f"TARGET ERROR: {e}")

        source_path = item.get("source_path")
        if source_path:
            try:
                source_result = analyze_safetensor(resolve_path(source_path, manifest_path), args.delta_threshold)
                source_rows.append(source_result)
            except Exception as e:
                source_result = {"error": str(e), "file": str(source_path)}
                print(f"SOURCE ERROR: {e}")

        print_one("TARGET", target_result if "error" not in (target_result or {}) else None)
        if target_result and "error" in target_result:
            print(f"TARGET ERROR DETAIL: {target_result}")

        print_one("SOURCE", source_result if "error" not in (source_result or {}) else None)
        if source_result and "error" in source_result:
            print(f"SOURCE ERROR DETAIL: {source_result}")

        diff = None
        if target_result and source_result and "error" not in target_result and "error" not in source_result:
            diff = {
                "hold_ratio_diff": target_result["hold_ratio"] - source_result["hold_ratio"],
                "delta_mean_diff": target_result["delta_mean"] - source_result["delta_mean"],
                "steps_diff": target_result["num_steps"] - source_result["num_steps"],
            }
            print("DIFF:")
            print(f"  hold_ratio_diff   : {diff['hold_ratio_diff']:.6f}")
            print(f"  delta_mean_diff   : {diff['delta_mean_diff']:.6f}")
            print(f"  steps_diff        : {diff['steps_diff']}")

        paired_results.append({
            "uuid": item.get("uuid"),
            "source_uuid": item.get("source_uuid"),
            "target": target_result,
            "source": source_result,
            "diff": diff,
            "manifest_meta": {
                "success": item.get("success"),
                "failure_reason": item.get("failure_reason"),
                "final_drawer_state": item.get("final_drawer_state"),
                "total_pause_steps": item.get("total_pause_steps"),
                "resume_count": item.get("resume_count"),
            }
        })

    print("\n" + "#" * 120)
    print("GROUP SUMMARY")
    print("#" * 120)

    target_summary = summarize_group(target_rows, "target")
    source_summary = summarize_group(source_rows, "source")
    print(json.dumps(target_summary, indent=2, ensure_ascii=False))
    print(json.dumps(source_summary, indent=2, ensure_ascii=False))

    valid_pairs = [
        p for p in paired_results
        if p["diff"] is not None
    ]
    if valid_pairs:
        hold_diff = np.asarray([p["diff"]["hold_ratio_diff"] for p in valid_pairs], dtype=np.float64)
        delta_diff = np.asarray([p["diff"]["delta_mean_diff"] for p in valid_pairs], dtype=np.float64)
        print("\nPAIRWISE DIFF SUMMARY")
        print(json.dumps({
            "num_pairs": len(valid_pairs),
            "hold_ratio_diff_mean": float(hold_diff.mean()),
            "hold_ratio_diff_std": float(hold_diff.std()),
            "delta_mean_diff_mean": float(delta_diff.mean()),
            "delta_mean_diff_std": float(delta_diff.std()),
        }, indent=2, ensure_ascii=False))

        print("\nTop target hold_ratio demos:")
        for p in sorted(valid_pairs, key=lambda x: x["target"]["hold_ratio"], reverse=True)[:args.top_k]:
            print(
                f"  target_hold={p['target']['hold_ratio']:.4f} "
                f"source_hold={p['source']['hold_ratio']:.4f} "
                f"uuid={p['uuid']} file={p['target']['file']}"
            )

        print("\nDemos with non-zero pause in TARGET:")
        nonzero_pause = [p for p in valid_pairs if (p["target"].get("paused_ratio") or 0) > 0]
        if nonzero_pause:
            for p in sorted(nonzero_pause, key=lambda x: x["target"]["paused_ratio"], reverse=True):
                print(
                    f"  paused_ratio={p['target']['paused_ratio']:.4f} "
                    f"hold={p['target']['hold_ratio']:.4f} "
                    f"uuid={p['uuid']} file={p['target']['file']}"
                )
        else:
            print("  None")

    if args.save_json:
        out = {
            "target_summary": target_summary,
            "source_summary": source_summary,
            "pairs": paired_results,
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()