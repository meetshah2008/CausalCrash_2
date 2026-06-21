from openai import OpenAI
import json, time, logging, os
import numpy as np
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — only edit this block
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
JUDGE_MODEL        = "google/gemini-2.5-flash"  # ← change to any OpenRouter model
                                                 # e.g. "openai/gpt-4o"
                                                 #      "anthropic/claude-3.5-sonnet"
                                                 #      "meta-llama/llama-3.1-70b-instruct"

MODEL_TAG  = os.getenv("MODEL_TAG", "model_tag")   # ← VLM being evaluated, change per run

PRED_DIR   = Path(os.getenv("PRED_DIR", f"./outputs/model_answers/{MODEL_TAG}"))
GT_DIR     = Path(os.getenv("GT_DIR", "./__data/annotations"))
OUT_DIR    = Path(os.getenv("OUT_DIR", f"./outputs/judge_results/gemini-2.5-flash/{MODEL_TAG}"))

BATCH_SIZE     = 20
PAUSE_SECS     = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 10
COST_PER_CALL  = 0.002   # USD estimate, adjust per model

# CCS weights — must sum to 1.0
CCS_WEIGHTS = {
    "level_1_perception"    : 0.25,
    "level_2_temporal"      : 0.25,
    "level_3_causal_chain"  : 0.00,
    "level_4_preventive"    : 0.25,
    "level_5_counterfactual": 0.25,
}

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.info(f"Model    : {MODEL_TAG}")
    logging.info(f"Judge    : {JUDGE_MODEL}")
    logging.info(f"PRED_DIR : {PRED_DIR}")
    logging.info(f"GT_DIR   : {GT_DIR}")
    logging.info(f"OUT_DIR  : {OUT_DIR}")
    logging.info(f"Log      : {log_path}\n")

    # OpenRouter uses OpenAI client with custom base_url
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://anonymous.4open.science/"),
            "X-Title"     : os.getenv("OPENROUTER_TITLE", "CausalCrash NeurIPS 2026"),
        }
    )

# ─────────────────────────────────────────────────────────────────────────────
# FILE HELPERS — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_pred_files() -> list[Path]:
    files = sorted(PRED_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No predicted JSONs found in {PRED_DIR}")
    return files

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(gt: dict, pred: dict) -> str:
    return f"""You are a senior road safety researcher evaluating a Vision-Language Model's
crash analysis output for the CausalCrash benchmark (NeurIPS 2026).

Your role is a LENIENT SEMANTIC AUDITOR — not a strict fact-checker.
The ground truth was annotated by a human researcher whose visual perspective
may differ slightly from another valid interpretation. Score generously when
the core understanding is correct, even if surface details differ.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO BE LENIENT ABOUT (do NOT penalize):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Color/appearance ambiguity  → "grey" vs "black", "white" vs "silver"
- Synonym descriptions        → "collision" vs "crash", "swerve" vs "veer"
- Vehicle type near-miss      → "SUV" vs "crossover", "truck" vs "pickup"
- Speed estimates             → ±15 km/h tolerance
- Lane naming style           → "left lane" vs "lane 1", "middle" vs "center"
- Agent IDs vs descriptive    → "agent_1" vs "white_sedan" is OK if role matches
- Extra detail not in GT      → bonus context, not a penalty
- Annotation verbosity        → more/less detailed is fine
- Minor env details           → road markings, weather adjectives
- Timing                      → ±2.0 seconds is acceptable
- Empty fields in prediction  → do not penalize, treat as "not provided"
- Generic agent IDs           → match by ROLE not by name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO SCORE STRICTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- VEHICLE COUNT      → Total involved must match
- INITIATOR/VICTIM   → Who caused vs who was hit. Swap = major failure
- COLLISION TYPE     → Rear-end vs head-on vs side-swipe vs T-bone
- CAUSAL SEQUENCE    → Order of events must be directionally correct
- TIMING             → Beyond ±2.0s = penalize
- ROAD POSITION      → Left vs right side matters

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 1 — SCENE PERCEPTION (0–10):
Evaluate: agent count, initiator/victim/ego roles, vehicle types,
          environmental factors (road type, traffic, lighting, surface)
  9–10 → Correct agents, roles, collision type. Minor label diffs OK.
  7–8  → Correct scene overall, 1 non-critical detail wrong.
  5–6  → Right count and collision type but role confusion.
  3–4  → Some agents or collision type wrong.
  0–2  → Fundamentally wrong scene or role swap.

LEVEL 2 — TEMPORAL REASONING (0–10):
Evaluate: critical_point_time_sec, latent/active phase, outcome label
  9–10 → Time within ±0.5s AND sequence perfect.
  7–8  → Time within ±1.0s OR 1 minor reordering.
  5–6  → Time within ±2.0s OR directionally correct.
  3–4  → Time off >2s but direction right.
  0–2  → Wrong sequence OR cannot identify critical moment.

LEVEL 3 — CAUSAL CHAIN (0–10):
Evaluate: do predicted steps semantically cover GT steps?
          Are key actors in the right steps? Is order correct?
  9–10 → All GT steps covered, actors correct, order correct.
  7–8  → Core steps covered, 1 actor/step missing or minor gap.
  5–6  → Main cause identified, chain incomplete or partially wrong.
  3–4  → Some steps present but sequence or actors wrong.
  0–2  → Completely different chain or missing entirely.

LEVEL 4 — PREVENTIVE ACTIONS (0–10):
Evaluate: primary action, alternatives — are they correct and relevant?
  9–10 → Primary action matches GT exactly or semantically. Alternatives valid.
  7–8  → Primary correct, alternatives partially match.
  5–6  → Primary is plausible but not the GT action.
  3–4  → Generic action without specificity.
  0–2  → Wrong actor or physically infeasible action.

LEVEL 5 — COUNTERFACTUAL QUALITY (0–10):
Evaluate: actor, action, explanation — does it cover the GT counterfactual meaning?
  9–10 → Correct actor, correct action, explanation semantically matches GT.
  7–8  → Correct actor and action, explanation partially matches.
  5–6  → Correct actor but different (still valid) action.
  3–4  → Plausible counterfactual but wrong actor or action.
  0–2  → Physically impossible or completely irrelevant counterfactual.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND_TRUTH_JSON:
{json.dumps(gt, indent=2)}

PREDICTED_JSON:
{json.dumps(pred, indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON, no markdown, no explanation:
{{
  "scores": {{
    "level_1_perception":         <int 0-10>,
    "level_2_temporal":           <int 0-10>,
    "level_3_causal_chain":       <int 0-10>,
    "level_4_preventive":         <int 0-10>,
    "level_5_counterfactual":     <int 0-10>
  }},
  "metrics": {{
    "temporal_error_sec":         <float>,
    "vehicle_count_match":        <bool>,
    "role_match":                 <bool>,
    "collision_type_match":       <bool>,
    "physical_logic_valid":       <bool>,
    "causal_steps_covered":       <float 0.0-1.0>,
    "counterfactual_actor_match": <bool>
  }},
  "leniency_applied": ["<what you were lenient about and why>"],
  "critical_errors":  ["<genuinely wrong things that affected score>"],
  "justification": "<2-3 sentences: what model got right, what wrong, overall quality>"
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def validate_result(result: dict) -> dict:
    expected_scores = list(CCS_WEIGHTS.keys())
    expected_metrics = [
        "temporal_error_sec", "vehicle_count_match", "role_match",
        "collision_type_match", "physical_logic_valid",
        "causal_steps_covered", "counterfactual_actor_match"
    ]
    for key in expected_scores:
        val = result.get("scores", {}).get(key)
        if val is None:
            raise ValueError(f"Missing score: {key}")
        if not (0 <= int(val) <= 10):
            raise ValueError(f"Score out of range for {key}: {val}")
    for key in expected_metrics:
        if key not in result.get("metrics", {}):
            raise ValueError(f"Missing metric: {key}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# API CALL — only this block changes for OpenRouter
# ─────────────────────────────────────────────────────────────────────────────

def call_judge(client: OpenAI, gt: dict, pred: dict) -> dict:
    last_error = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": build_prompt(gt, pred)
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"},  # forces JSON output
            )

            raw = response.choices[0].message.content
            # strip markdown fences if any model adds them
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            return validate_result(result)

        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS - 1:
                wait = RETRY_DELAY * (attempt + 1)
                logging.warning(f"  Attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                logging.error(f"  All {RETRY_ATTEMPTS} attempts failed: {e}")

    raise RuntimeError(f"Judge failed after {RETRY_ATTEMPTS} attempts: {last_error}")

# ─────────────────────────────────────────────────────────────────────────────
# CCS + SUMMARY — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def compute_ccs(scores: dict) -> float:
    return sum(CCS_WEIGHTS[k] * scores[k] for k in CCS_WEIGHTS)

def print_summary(all_results: list, total_files: int, total_cost: float) -> None:
    ok     = [r for r in all_results if r.get("eval_status") == "ok"]
    failed = [r for r in all_results if r.get("eval_status") == "failed"]

    if not ok:
        logging.info("No successful evaluations to summarize.")
        return

    l1 = [r["scores"]["level_1_perception"]     for r in ok]
    l2 = [r["scores"]["level_2_temporal"]       for r in ok]
    l3 = [r["scores"]["level_3_causal_chain"]   for r in ok]
    l4 = [r["scores"]["level_4_preventive"]     for r in ok]
    l5 = [r["scores"]["level_5_counterfactual"] for r in ok]
    ccs_scores = [compute_ccs(r["scores"]) for r in ok]

    summary = {
        "model": MODEL_TAG, "judge": JUDGE_MODEL,
        "total_videos": total_files, "evaluated": len(ok), "failed": len(failed),
        "L1_mean": round(float(np.mean(l1)), 3), "L1_std": round(float(np.std(l1)), 3),
        "L2_mean": round(float(np.mean(l2)), 3), "L2_std": round(float(np.std(l2)), 3),
        "L3_mean": round(float(np.mean(l3)), 3), "L3_std": round(float(np.std(l3)), 3),
        "L4_mean": round(float(np.mean(l4)), 3), "L4_std": round(float(np.std(l4)), 3),
        "L5_mean": round(float(np.mean(l5)), 3), "L5_std": round(float(np.std(l5)), 3),
        "CCS_mean": round(float(np.mean(ccs_scores)), 3),
        "CCS_std" : round(float(np.std(ccs_scores)),  3),
        "estimated_cost_usd": round(total_cost, 4),
    }
    save_json(summary, OUT_DIR / "summary.json")

    if failed:
        failed_names = [r["video_id"] for r in failed]
        (OUT_DIR / "failed.txt").write_text("\n".join(failed_names))

    sep = "=" * 52
    logging.info(f"\n{sep}")
    logging.info(f"  MODEL     : {MODEL_TAG}")
    logging.info(f"  JUDGE     : {JUDGE_MODEL}")
    logging.info(f"  Evaluated : {len(ok)} / {total_files}  |  Failed: {len(failed)}")
    logging.info(sep)
    logging.info(f"  L1 Perception     : {np.mean(l1):.2f} ± {np.std(l1):.2f}")
    logging.info(f"  L2 Temporal       : {np.mean(l2):.2f} ± {np.std(l2):.2f}")
    logging.info(f"  L3 Causal Chain   : {np.mean(l3):.2f} ± {np.std(l3):.2f}")
    logging.info(f"  L4 Preventive     : {np.mean(l4):.2f} ± {np.std(l4):.2f}")
    logging.info(f"  L5 Counterfactual : {np.mean(l5):.2f} ± {np.std(l5):.2f}")
    logging.info(sep)
    logging.info(f"  CausalCrash Score : {np.mean(ccs_scores):.2f} ± {np.std(ccs_scores):.2f} / 10")
    logging.info(f"  Estimated cost    : ~${total_cost:.3f} USD")
    logging.info(f"  Results in        : {OUT_DIR}")
    logging.info(sep)
    if failed:
        logging.info(f"\n  Failed videos → {OUT_DIR / 'failed.txt'}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def run(client: OpenAI) -> None:
    pred_files  = get_pred_files()
    all_results = []
    total_cost  = 0.0

    logging.info(f"Videos found : {len(pred_files)}\n")

    for i, pred_path in enumerate(pred_files):
        gt_path    = GT_DIR  / pred_path.name
        cache_path = OUT_DIR / f"{pred_path.stem}_eval.json"

        if cache_path.exists():
            all_results.append(load_json(cache_path))
            logging.info(f"[{i+1:03d}/{len(pred_files)}] SKIP (cached) : {pred_path.name}")
            continue

        if not gt_path.exists():
            logging.info(f"[{i+1:03d}/{len(pred_files)}] SKIP (no GT)  : {pred_path.name}")
            continue

        try:
            gt   = load_json(gt_path)
            pred = load_json(pred_path)

            result = call_judge(client, gt, pred)
            result["video_id"]    = pred_path.stem
            result["model"]       = MODEL_TAG
            result["judge"]       = JUDGE_MODEL
            result["eval_status"] = "ok"

            total_cost += COST_PER_CALL
            save_json(result, cache_path)
            all_results.append(result)

            s   = result["scores"]
            ccs = compute_ccs(s)
            logging.info(
                f"[{i+1:03d}/{len(pred_files)}] OK  "
                f"L1={s['level_1_perception']} "
                f"L2={s['level_2_temporal']} "
                f"L3={s['level_3_causal_chain']} "
                f"L4={s['level_4_preventive']} "
                f"L5={s['level_5_counterfactual']} "
                f"CCS={ccs:.1f} ~${total_cost:.3f}  {pred_path.name}"
            )

        except Exception as e:
            result = {
                "video_id": pred_path.stem, "model": MODEL_TAG,
                "judge": JUDGE_MODEL, "eval_status": "failed", "error": str(e),
            }
            save_json(result, cache_path)
            all_results.append(result)
            logging.error(f"[{i+1:03d}/{len(pred_files)}] FAILED : {pred_path.name} — {e}")

        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < len(pred_files):
            logging.info(f"\n--- Batch {(i+1)//BATCH_SIZE} done. Pausing {PAUSE_SECS}s ---\n")
            time.sleep(PAUSE_SECS)

    print_summary(all_results, len(pred_files), total_cost)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = setup()
    run(client)