import json
import sys
import time
import io
import traceback
import orjson
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_PATH = "raw"
OUTPUT_DIR = "labeled_v2"
WORKERS = 8


def _load_jsonl(path: Path) -> list:
    results = []
    bad = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(orjson.loads(line))
            except Exception:
                bad += 1
    if bad:
        print(f"skipped {bad} malformed lines in {path.name}")
    return results


_SCENARIO_SIZES = {
    "wilson": 634_246,
    "wheeler": 616_161,
    "harrison": 593_948,
    "fox": 473_104,
    "santos": 130_779,
    "wardbeck": 91_257,
    "shaw": 70_782,
    "russellmitchell": 45_544,
}


def _process_scenario_worker(args: tuple):
    scenario, dataset_path, output_dir = args
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    buf = io.StringIO()

    def log(*a, **kw):
        print(*a, **kw, file=buf)

    t_start = time.time()

    try:
        from alert_labeler import AlertLabeler, parse_attack_windows

        log(f"\n{scenario}")

        windows = parse_attack_windows(str(dataset_path / "labels.csv"), scenario)
        if not windows:
            return scenario, None, buf.getvalue(), "no attack windows"

        log(f"{len(windows)} attack phases:")
        for start, end, attack_type in windows:
            duration = (end - start).total_seconds() / 60
            log(f"{attack_type}: {duration:.1f} min")

        labeler = AlertLabeler(windows)
        all_alerts = []

        wazuh_file = dataset_path / f"{scenario}_wazuh.json"
        if wazuh_file.exists():
            wazuh_alerts = _load_jsonl(wazuh_file)
            all_alerts.extend(wazuh_alerts)
            log(f"wazuh:  {len(wazuh_alerts):,}")

        aminer_file = dataset_path / f"{scenario}_aminer.json"
        if aminer_file.exists():
            aminer_alerts = _load_jsonl(aminer_file)
            all_alerts.extend(aminer_alerts)
            log(f"aminer: {len(aminer_alerts):,}")

        if not all_alerts:
            return scenario, None, buf.getvalue(), "no alert files"

        # label_scenario prints internally, redirect to our buffer
        old_stdout, sys.stdout = sys.stdout, buf
        try:
            labeled = labeler.label_scenario(all_alerts, scenario)
            labeler.export_comprehensive(
                labeled, str(output_dir / f"{scenario}_comprehensive.json")
            )
        finally:
            sys.stdout = old_stdout

        label_counts = Counter()
        conf_very_high = conf_high = conf_med = conf_low = 0
        has_ip = has_user = has_process = has_domain = 0

        for a in labeled:
            label_counts[a.label] += 1
            c = a.confidence
            if c >= 0.9:
                conf_very_high += 1
            elif c >= 0.8:
                conf_high += 1
            elif c >= 0.6:
                conf_med += 1
            else:
                conf_low += 1
            if a.src_ip or a.dst_ip:
                has_ip += 1
            if a.users:
                has_user += 1
            if a.processes:
                has_process += 1
            if a.domains:
                has_domain += 1

        n = len(labeled)
        elapsed = time.time() - t_start

        stats = {
            "total": n,
            "labels": dict(label_counts),
            "conf_very_high": conf_very_high,
            "conf_high": conf_high,
            "conf_med": conf_med,
            "conf_low": conf_low,
            "elapsed_s": round(elapsed, 1),
            "entity_extraction": {
                "with_ips": 100 * has_ip / n,
                "with_users": 100 * has_user / n,
                "with_processes": 100 * has_process / n,
                "with_domains": 100 * has_domain / n,
            },
        }

        log(f"done {scenario}: {n:,} alerts in {elapsed:.1f}s ({n / elapsed:,.0f}/s)")

        return scenario, stats, buf.getvalue(), None

    except Exception as e:
        err = traceback.format_exc()
        log(f"error in {scenario}: {e}\n{err}")
        return scenario, None, buf.getvalue(), err


def process_all_scenarios(
    dataset_path: str, output_dir: str = "labeled_v2", workers: int = 8
):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    labels_file = dataset_path / "labels.csv"
    if not labels_file.exists():
        print(f"labels.csv not found at {labels_file}")
        return

    all_scenarios = [
        "fox",
        "harrison",
        "russellmitchell",
        "santos",
        "shaw",
        "wardbeck",
        "wheeler",
        "wilson",
    ]
    scenarios = sorted(
        all_scenarios, key=lambda s: _SCENARIO_SIZES.get(s, 0), reverse=True
    )

    n_workers = min(workers, len(scenarios))
    print(f"workers: {n_workers}, scenarios: {' > '.join(scenarios)}")

    t_batch_start = time.time()

    work_args = [(sc, str(dataset_path), str(output_dir)) for sc in scenarios]

    completed_logs = {}
    all_stats = {}
    failed = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_scenario_worker, args): args[0] for args in work_args
        }

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            scenario, stats, log_text, error = future.result()
            completed_logs[scenario] = log_text

            if error or stats is None:
                failed.append(scenario)
                last_line = error.splitlines()[-1] if error else "no output"
                status = f"FAILED: {last_line}"
            else:
                all_stats[scenario] = stats
                m = stats["labels"].get("MALICIOUS", 0)
                s = stats["labels"].get("SUSPICIOUS", 0)
                n = stats["total"]
                elapsed = stats["elapsed_s"]
                status = f"{n:,} alerts MAL={m:,}({100*m/n:.1f}%) SUSP={s:,}({100*s/n:.1f}%) {elapsed:.1f}s"

            print(f"[{done_count}/{len(scenarios)}] {scenario} {status}")

    # dump per-scenario logs in scenario order, not completion order
    for sc in scenarios:
        if sc in completed_logs:
            print(completed_logs[sc], end="")

    for scenario in scenarios:
        if scenario not in all_stats:
            continue
        stats = all_stats[scenario]
        n = stats["total"]
        print(f"\n{scenario} ({stats['elapsed_s']:.1f}s): {n:,} alerts")
        for label, count in sorted(stats["labels"].items()):
            print(f"{label}: {count:,} ({100*count/n:.2f}%)")
        print(
            f"conf >=0.9: {stats['conf_very_high']:,} | 0.8-0.9: {stats['conf_high']:,} | "
            f"0.6-0.8: {stats['conf_med']:,} | <0.6: {stats['conf_low']:,}"
        )

    total_elapsed = time.time() - t_batch_start
    total_alerts = sum(s["total"] for s in all_stats.values())
    print(
        f"\ntotal: {total_alerts:,} alerts / {len(all_stats)} scenarios in "
        f"{total_elapsed:.1f}s ({total_alerts / total_elapsed:,.0f}/s)"
    )

    if failed:
        print(f"failed: {', '.join(failed)}")

    summary_out = {
        sc: {k: v for k, v in s.items() if k != "elapsed_s"}
        for sc, s in all_stats.items()
    }
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"summary: {summary_file}")
    print(f"output: {output_dir}/")

    return all_stats


if __name__ == "__main__":
    stats = process_all_scenarios(DATASET_PATH, OUTPUT_DIR, WORKERS)
    if not stats:
        print("no scenarios processed")
