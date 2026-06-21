"""
Hazard detection script using the official OpenRouter Python SDK.
Processes videos from VIDEO_DIR, saves one JSON per video to OUTPUT_DIR.
"""

import base64
import json
import re
import sys
import os
from pathlib import Path

from openrouter import OpenRouter
from tqdm import tqdm

# ─────────────────────────── USER CONSTANTS ──────────────────────────────────
# MODEL_ID   = "google/gemini-2.5-flash"
MODEL_ID   = os.getenv("MODEL_ID", "qwen/qwen3.5-plus-20260420")
# MODEL_ID   = "z-ai/glm-4.6v"
VIDEO_DIR  = Path(os.getenv("VIDEO_DIR", "./__data/dataset/"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs/model_answers")) / MODEL_ID.split("/")[1]
API_KEY    = os.getenv("OPENROUTER_API_KEY", "")
REASONING_ENABLED = False
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

MIME_TYPES = {
    ".mp4":  "video/mp4",
    ".avi":  "video/x-msvideo",
    ".mov":  "video/quicktime",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
    ".flv":  "video/x-flv",
    ".wmv":  "video/x-ms-wmv",
}

HAZARD_PROMPT = """You are a traffic-safety analyst reviewing dashcam/CCTV footage.
Watch the video and return a single JSON object - no markdown, no extra text.

SCHEMA (fill every field):
{
  "video_metadata": {
    "hazard_present": <true|false>,
    "hazard_type": [],
    "severity_score": <0–5>,
    "confidence": <0.0–1.0>,
    "notes": ""
  },
  "level_1_perception": {
    "agents": [
      {
        "id": "",
        "role": <"initiator"|"victim"|"ego"|"observer"|"contributor"|"unknown">,
        "type": <"vehicle"|"heavy_vehicle"|"motorcycle"|"bicycle"|"pedestrian"|"animal"|"obstacle"|"unknown">,
        "initial_state": {
          "motion": <"stopped"|"slow"|"moving"|"fast"|"unknown">,
          "lane": <"left"|"center"|"right"|"merging"|"shoulder"|"unknown">,
          "intent": <"keeping_lane"|"changing_left"|"changing_right"|"braking"|"accelerating"|"swerving"|"unknown">
        },
        "spatial_location": <"front_left"|"front_center"|"front_right"|"left"|"right"|"rear_left"|"rear_center"|"rear_right"|"unknown">,
        "visibility": <"clear"|"obstructed"|"partial"|"unknown">,
        "confidence": <0.0–1.0>
      }
    ],
    "environmental_factors": {
      "road_type": "",
      "traffic": <"light"|"moderate"|"heavy"|"unknown">,
      "lighting": <"daylight"|"overcast"|"night"|"dawn_dusk"|"unknown">,
      "road_surface": <"dry"|"wet"|"icy"|"unknown">,
      "other": []
    },
    "confidence": <0.0–1.0>,
    "notes": ""
  },
  "level_2_predictive": {
    "temporal_window": {
      "latent_phase_sec": [<start>, <end>],
      "active_phase_sec": [<start>, <end>]
    },
    "critical_point_time_sec": <float>,
    "predicted_outcome": {
      "label": <hazard_type label or "none">,
      "summary": "",
      "confidence": <0.0–1.0>
    },
    "notes": ""
  },
  "level_3_explanation_counterfactual": {
    "causal_chain": [
      {
        "step": <int>,
        "event": "",
        "actors": [],
        "time_sec": <float|null>,
        "confidence": <0.0–1.0>
      }
    ],
    "preventive_actions": {
      "primary": <action>,
      "alternatives": [],
      "infeasible": [],
      "notes": ""
    },
    "counterfactual_analysis": [
      {
        "actor": "",
        "action": <action>,
        "risk_reduced": <true|false>,
        "explanation": "",
        "confidence": <0.0–1.0>
      }
    ],
    "notes": ""
  }
}

VOCABULARIES (use only these labels):
hazard_type: collision | near_miss | unsafe_lane_change | loss_of_control | sudden_stop | cut_in | pedestrian_conflict | cyclist_conflict | debris | wrong_way | red_light_violation | road_departure | other
preventive_action: maintain_safe_following_distance | brake_earlier | brake_hard | stay_in_lane | check_blind_spot | avoid_lane_change | reduce_speed | increase_gap | keep_right | other

RULES:
1. hazard_present=false → hazard_type=[], severity_score=0; Levels 2/3 fields set to null/"N/A"/[].
2. hazard_present=true → hazard_type≥1 label, severity_score 1–5, critical_point_time_sec within active_phase_sec.
3. Time ordering: latent_phase_sec[0] ≤ latent_phase_sec[1] ≤ active_phase_sec[0] ≤ active_phase_sec[1].
4. All values of keys ending with "_sec" should purely be in seconds (E.g. {"time_sec": 121.0} for 02:01 mins).
4. All counterfactual actions must come from preventive_action vocab.
5. All initial_state fields must use vocab tokens - no free text.
6. The schema must be followed in the exact order. Neither any extra fields should be present nor any any field from the schema must be missing. Only double-quotes (") must be used to enclose keys and values.
7. No comments are allowed in the returned JSON.

severity_score: 0=none, 1=very minor, 2=minor, 3=moderate, 4=serious, 5=severe/life-threatening.
"""


def encode_video(video_path: Path) -> str:
    """Return a base64 data-URL for the given video file."""
    mime = MIME_TYPES.get(video_path.suffix.lower(), "video/mp4")
    data = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def run_inference(video_path: Path, client: OpenRouter) -> tuple[str, int, int]:
    """
    Send a single video to OpenRouter and return:
        (raw_text_response, prompt_tokens, completion_tokens)
    """
    video_url = encode_video(video_path)

    response = client.chat.send(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "fps": 1,
                    },
                    {
                        "type": "text",
                        "text": HAZARD_PROMPT,
                    },
                ],
            }
        ],
        reasoning={"enabled": REASONING_ENABLED},
        # max_tokens=500_000,
        # temperature=1.0,
        # top_p=0.95,
        # presence_penalty=1.5,
    )

    # The official openrouter SDK returns an object with OpenAI-compatible fields.
    # Usage is always included in every response automatically.
    raw_text = response.choices[0].message.content
    if raw_text is None:
        tqdm.write(f"[WARNING] API call returned `None` in message content.")
        tqdm.write(f"Full response:\n{response}")
        raw_text = "None"

    raw_text = raw_text.strip()

    usage = getattr(response, "usage", None)
    prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    if prompt_tokens == 0:
        tqdm.write(f"[WARNING] prompt_tokens is 0 for {video_path.name} — usage data may be missing or incomplete.")
    if completion_tokens == 0:
        tqdm.write(f"[WARNING] completion_tokens is 0 for {video_path.name} — usage data may be missing or incomplete.")

    return raw_text, prompt_tokens, completion_tokens


def parse_model_output(raw: str, video_id: str) -> dict:
    """
    Extract JSON from the model's response and inject video_id into video_metadata.
    Falls back to a safe default if parsing fails.
    """
    # Strip <think>...</think> block emitted by thinking models before the JSON
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Strip optional markdown fences
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("`").strip()

    # Locate the outermost JSON object
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        json_str = cleaned[start:end]
    else:
        json_str = cleaned

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        tqdm.write(f"[WARNING] JSON parse error for {video_id}: {exc}")
        tqdm.write(f"Raw output was:\n{raw}\n")
        return None

    # Inject video_id into the model's existing video_metadata block
    payload.setdefault("video_metadata", {})
    payload["video_metadata"] = {"video_id": video_id, **payload["video_metadata"]}

    # Coerce critical types inside video_metadata for safety
    vm = payload["video_metadata"]
    vm["hazard_present"] = bool(vm.get("hazard_present", False))
    if not isinstance(vm.get("hazard_type"), list):
        vm["hazard_type"] = [vm["hazard_type"]] if vm.get("hazard_type") else []
    vm["severity_score"] = int(vm.get("severity_score", 0))

    return payload


def process_video(
    video_path: Path,
    output_dir: Path,
    client: OpenRouter,
) -> tuple[int, int]:
    """
    Run inference on one video, write the result JSON, and return
    (prompt_tokens, completion_tokens) consumed by this call.
    Returns (0, 0) when the output file already existed (skipped).
    """
    video_id    = video_path.name
    output_path = output_dir / (video_path.stem + ".json")

    if output_path.exists():
        return 0, 0  # silently skip; tqdm will still advance

    raw_output, prompt_tokens, completion_tokens = run_inference(video_path, client)
    result = parse_model_output(raw_output, video_id)
    if result is not None:
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return prompt_tokens, completion_tokens


def main() -> None:
    print("MODEL ID:", MODEL_ID, "\n")

    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)

    if not VIDEO_DIR.exists():
        print(f"ERROR: VIDEO_DIR does not exist: {VIDEO_DIR.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No video files found in {VIDEO_DIR.resolve()}")
        sys.exit(0)

    print(f"Found {len(video_files)} video(s) to process.\n")

    total_prompt_tokens     = 0
    total_completion_tokens = 0

    with OpenRouter(api_key=API_KEY) as client:
        for video_path in tqdm(video_files, desc="Processing videos", unit="video"):
            try:
                tqdm.write(f"Processing {video_path} ...")
                p_tok, c_tok = process_video(video_path, OUTPUT_DIR, client)
                tqdm.write(f"Input tokens: {p_tok}")
                tqdm.write(f"Output tokens: {c_tok}\n")
                total_prompt_tokens     += p_tok
                total_completion_tokens += c_tok
            except Exception as exc:
                tqdm.write(f"[ERROR] Failed to process {video_path.name}: {exc}")

    print("─" * 40)
    print(f"  Total Input tokens  : {total_prompt_tokens}")
    print(f"  Total Output tokens : {total_completion_tokens}")
    print("─" * 40)


if __name__ == "__main__":
    main()
