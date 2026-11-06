#!/usr/bin/env python3
"""
Standalone diagnostic for the entailment model pipeline.

Run this on the SAME machine/environment where the evaluation jobs run
(e.g. on Barnard, inside the same venv/module load as run_SQuAI_evaluation*.py).
It isolates each step so you can see exactly where things break: network
reachability to the Hub, local cache state, disk space, model/tokenizer
loading, or actual inference.

Usage:
    python3 test_entailment_model.py
"""
import os
import shutil
import sys
import traceback

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("1. Environment")
    for var in [
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
        "http_proxy", "https_proxy", "HF_TOKEN", "HUGGINGFACE_TOKEN",
    ]:
        print(f"  {var} = {os.environ.get(var, '<unset>')}")

    import torch
    print(f"  torch = {torch.__version__}, cuda_available = {torch.cuda.is_available()}")

    import transformers
    print(f"  transformers = {transformers.__version__}")

    import huggingface_hub
    print(f"  huggingface_hub = {huggingface_hub.__version__}")

    section("2. Disk space on cache filesystem")
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    try:
        usage = shutil.disk_usage(os.path.dirname(cache_dir) if not os.path.exists(cache_dir) else cache_dir)
        print(f"  cache_dir = {cache_dir}")
        print(f"  total={usage.total/1e9:.1f}GB used={usage.used/1e9:.1f}GB free={usage.free/1e9:.1f}GB")
        if usage.free / 1e9 < 2:
            print("  WARNING: less than 2GB free — this model is ~1.7GB, a download could fail mid-way.")
    except Exception as e:
        print(f"  could not stat cache dir: {e}")

    section("3. What's already cached locally")
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        found = False
        for r in info.repos:
            if MODEL_NAME in r.repo_id:
                found = True
                print(f"  {r.repo_id} ({r.repo_type}) size={r.size_on_disk_str} files={r.nb_files}")
                for rev in r.revisions:
                    print(f"    revision {rev.commit_hash[:8]} refs={rev.refs} size={rev.size_on_disk_str}")
        if not found:
            print(f"  {MODEL_NAME} not found in local cache — a fresh download is required.")
    except Exception as e:
        print(f"  scan_cache_dir failed: {e}")

    section("4. Network reachability to Hugging Face Hub")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info(MODEL_NAME, timeout=15)
        print(f"  OK — reached the Hub. sha={model_info.sha}, siblings={len(model_info.siblings)} files")
    except Exception as e:
        print(f"  FAILED to reach/query the Hub: {type(e).__name__}: {e}")
        print("  -> if this fails, it's a network/DNS/proxy/firewall issue on this node, not the model itself.")

    section("5. Loading tokenizer")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"  OK — tokenizer loaded: {type(tok).__name__}")
    except Exception:
        print("  FAILED loading tokenizer:")
        traceback.print_exc()
        sys.exit(1)

    section("6. Loading model")
    try:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print(f"  OK — model loaded: {type(model).__name__}, id2label={model.config.id2label}")
    except Exception:
        print("  FAILED loading model:")
        traceback.print_exc()
        sys.exit(1)

    section("7. Running one real inference (CPU)")
    try:
        import torch
        inputs = tok(
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower is in Paris.",
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)[0]
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        scores = {id2label[i]: probs[i].item() for i in range(len(probs))}
        print(f"  OK — scores = {scores}")
        if any(v != v for v in scores.values()):  # NaN check
            print("  WARNING: NaN in scores!")
    except Exception:
        print("  FAILED during inference:")
        traceback.print_exc()
        sys.exit(1)

    section("Result")
    print("All steps succeeded — the model pull/load/inference pipeline is healthy on this node.")
    print("If evaluation jobs are still failing with 'missing/invalid value(s) for noise',")
    print("the issue is more likely in the data (e.g. empty/unpunctuated context) than the model.")


if __name__ == "__main__":
    main()
