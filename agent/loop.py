"""TerraVision Agent — conversation loop.

Groq (Llama 3.3 70B) + tool-calling over the TerraVision predict_crop_yield tool.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from agent.tools import CROP_TYPES, TOOL_FUNCTIONS, TOOL_SCHEMAS  # noqa: E402

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = f"""You are the TerraVision AI Assistant, an expert on crop yield forecasting
using satellite (NDVI) and climate (ERA5) data.

Rules:
- For ANY question about crop health, yield, or field conditions at a location, you MUST call
  predict_crop_yield to get real data. Never guess or invent NDVI, yield, or climate numbers.
- The ONLY supported crop types are: {", ".join(CROP_TYPES)}. If the user asks about a crop
  NOT in this list (e.g. cotton, sugarcane), tell them plainly that crop isn't supported yet
  and name the crops that are — do NOT call the tool with an unsupported crop value.
- If the tool result includes a "_note" about demo/mock mode, mention this transparently to the
  user — do not present mock numbers as if they were live satellite data.
- If the tool returns an "error", tell the user plainly what went wrong — do not make up a
  fallback answer, and do not repeatedly retry the same failing call.
- Keep answers concise and reference the actual numbers returned by the tool.
- If the user's question doesn't include enough information to call the tool (missing location
  or crop type), ask them for it instead of guessing coordinates or a crop.
"""

MODEL_NAME = "llama-3.3-70b-versatile"

_YIELD_KEYWORDS = (
    "yield", "crop", "ndvi", "harvest", "field", "wheat", "rice", "maize", "soybean",
)

_DEFINITIONAL_PATTERNS = (
    "what is", "what are", "how is", "how are", "how does", "how do",
    "explain", "define", "definition of", "what does", "meaning of",
)


def _looks_like_yield_question(text: str) -> bool:
    lowered = text.lower()
    if any(p in lowered for p in _DEFINITIONAL_PATTERNS):
        return False
    return any(kw in lowered for kw in _YIELD_KEYWORDS)

def _validate_args(args: dict[str, Any]) -> str | None:
    """Return an error string if args are invalid, else None."""
    lat, lon, crop = args.get("lat"), args.get("lon"), args.get("crop")
    if lat is None or lon is None or crop is None:
        return "Missing required argument(s): lat, lon, and crop are all required."
    if not (-90 <= float(lat) <= 90):
        return f"Invalid latitude {lat!r} — must be between -90 and 90."
    if not (-180 <= float(lon) <= 180):
        return f"Invalid longitude {lon!r} — must be between -180 and 180."
    if crop not in CROP_TYPES:
        return f"Invalid crop {crop!r} — must be one of {CROP_TYPES}."
    return None


def run_agent(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    max_tool_hops: int = 3,
) -> dict[str, Any]:
    """
    Run one turn of the agent. Returns the final answer plus a trace of tool calls
    (for logging/eval — see agent/eval.py).
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    trace: list[dict[str, Any]] = []
    last_failed_call: tuple[str, str] | None = None

    for hop in range(max_tool_hops):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
                tool_choice="auto",
                temperature=0.2,
            )
        except groq.BadRequestError as e:
            # The model generated an argument that failed schema validation server-side
            # (e.g. an unsupported crop). Tell it plainly and let it try again, once.
            err_detail = str(e)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last tool call was rejected: {err_detail}. "
                        f"Only these crop values are valid: {', '.join(CROP_TYPES)}. "
                        "If the user's crop isn't in that list, tell them it isn't "
                        "supported instead of calling the tool again."
                    ),
                }
            )
            trace.append({"tool": "predict_crop_yield", "args": {}, "result": {"error": err_detail}})
            continue
        except groq.APIError as e:
            return {
                "answer": f"There was a problem reaching the language model: {e}",
                "trace": trace,
                "hops": hop,
            }

        msg = response.choices[0].message

        if not msg.tool_calls:
            if hop == 0 and _looks_like_yield_question(user_message):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Please call the predict_crop_yield tool to answer this — "
                            "do not answer from general knowledge."
                        ),
                    }
                )
                continue
            return {"answer": msg.content, "trace": trace, "hops": hop}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            args_key = (fn_name, json.dumps(args, sort_keys=True))

            if args_key == last_failed_call:
                result = {
                    "error": (
                        "This exact call already failed once this turn — "
                        "not retrying with the same arguments."
                    )
                }
            else:
                validation_error = (
                    _validate_args(args) if fn_name == "predict_crop_yield" else None
                )
                if validation_error:
                    result = {"error": validation_error}
                else:
                    fn = TOOL_FUNCTIONS.get(fn_name)
                    result = (
                        {"error": f"Unknown tool: {fn_name}"} if fn is None else fn(**args)
                    )

            if isinstance(result, dict) and "error" in result:
                last_failed_call = args_key

            trace.append({"tool": fn_name, "args": args, "result": result})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result),
                }
            )

    return {
        "answer": (
            "I wasn't able to finish reasoning about this in time — "
            "please try rephrasing your question."
        ),
        "trace": trace,
        "hops": max_tool_hops,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What's the wheat yield outlook near Faisalabad (31.4, 73.1)?"
    )
    result = run_agent(question)

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- TRACE ---")
    for step in result["trace"]:
        print(f"  tool={step['tool']} args={step['args']}")
        print(f"  result={json.dumps(step['result'], indent=2)[:500]}")
