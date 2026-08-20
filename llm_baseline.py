import json, os, re, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

LLM_TEST_PATH = Path("processed_v2/llm_windows.json")

# "gemini" or "ollama"
BACKEND = "gemini"
OUTPUT = "llm_scores.json"
RESUME = False
WORKERS = 10

FEW_SHOT = """LOGS: Feb  4 13:33:27 mail dovecot: imap-login: Login: user=<ryan.chambers>, method=PLAIN, rip=10.237.2.255 src=10.237.2.255 dst=10.237.2.255
LABEL: 0
---
LOGS: sig:Suricata: Alert - SURICATA TLS invalid handshake message | cat:Generic Protocol Command Decode | flow:52.94.236.45:443->10.229.0.4:48030
LABEL: 0
---
LOGS: sig:Suricata: Alert - ET SCAN Possible Nmap User-Agent Observed | cat:Web Application Attack src=10.182.193.78 dst=10.182.194.196
LABEL: 1
---
LOGS: 172.19.131.174 - - [24/Jan/2022:03:58:09 +0000] "HEAD /wp-content/plugins/image-symlinks/inc/thumb.php HTTP/1.1" 404 146 "WPScan v3.8.20" src=172.19.131.174
LABEL: 1
---"""

SYSTEM_PROMPT = f"""You are a cybersecurity analyst. Given a 5-minute window of security alerts, label it 1 (ANOMALOUS) or 0 (NORMAL). Return ONLY the digit.

### EXAMPLES ###
{FEW_SHOT}

TASK: Label the following window:"""


def make_client(backend):
    if backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        assert api_key

        return (
            OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "gemini-2.5-flash",
        )
    elif backend == "ollama":
        return (
            OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
            ),
            "llama3.1:8b",
        )
    assert False


def classify_window(client, model, text, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"[01]", raw)
            return int(match.group()) if match else -1
        except Exception as e:
            err = str(e)
            retry_match = re.search(r"retry in ([\d.]+)s", err, re.I)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = (
                    float(retry_match.group(1)) + 1
                    if retry_match
                    else 15 * (attempt + 1)
                )
                print("rate limited", wait)
                time.sleep(wait)
                continue
            print("err:", e)
            return -1
    print(f"FAILED after {max_retries} retries")
    return -1


def main():
    assert LLM_TEST_PATH.exists()

    with open(LLM_TEST_PATH) as f:
        test_windows = json.load(f)

    n_anom = sum(1 for w in test_windows if w["label"] == 1)
    assert all(k in test_windows[0] for k in ("idx", "label", "text", "scenario"))

    client, model = make_client(BACKEND)
    print(
        f"{BACKEND} ({model}) | {len(test_windows)} windows ({n_anom} anomaly) | {WORKERS} workers"
    )

    done = {}
    if RESUME and Path(OUTPUT).exists():
        with open(OUTPUT) as f:
            for entry in json.load(f):
                done[entry["idx"]] = entry
        print(f"Resuming: {len(done)} already done")

    todo = [w for w in test_windows if w["idx"] not in done]
    results = list(done.values())
    lock = threading.Lock()
    n_total = len(test_windows)
    dirty = False

    def checkpoint():
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)

    def process(w):
        pred = classify_window(client, model, w["text"])
        return {
            "idx": w["idx"],
            "scenario": w["scenario"],
            "label": w["label"],
            "score": pred,
            "phases": w.get("phases", []),
            "set": "test_anomaly" if w["label"] == 1 else "test_normal",
        }

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(process, w): w for w in todo}
            for fut in as_completed(futures):
                entry = fut.result()
                with lock:
                    results.append(entry)
                    dirty = True
                    n_done = len(results)
                    print(
                        f"[{n_done}/{n_total}] idx={entry['idx']} true={entry['label']} pred={entry['score']}"
                    )
                    if n_done % 10 == 0:
                        checkpoint()
                        dirty = False
    except KeyboardInterrupt:
        print("interrupted, saving...")
    finally:
        if dirty:
            checkpoint()
        print(f"saved {len(results)} results to {OUTPUT}")

    valid = [r for r in results if r["score"] in (0, 1)]
    if valid:
        from sklearn.metrics import classification_report, roc_auc_score

        y_true = [r["label"] for r in valid]
        y_pred = [r["score"] for r in valid]
        print(f"{len(valid)}/{len(results)} valid")
        print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"]))
        if len(set(y_true)) > 1:
            print(f"AUROC: {roc_auc_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()
