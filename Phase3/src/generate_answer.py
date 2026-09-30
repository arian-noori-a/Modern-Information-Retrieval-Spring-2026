"""Stage 5: structured JSON answers from a small local instruction-following LLM.

The LLM only explains/structures the top reranked products. Retrieval remains the
source of candidate products. Invalid model outputs are retried up to two times;
a deterministic schema-valid fallback guarantees a complete output file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_jsonl, write_jsonl

ALLOWED_ROLES = {"exact", "substitute", "irrelevant"}
ALLOWED_DECISIONS = {
    "recommend_exact",
    "recommend_exact_with_warning",
    "recommend_substitute",
    "no_good_match",
    "ask_clarification",
}
FORBIDDEN_COMMERCE = {"price", "stock", "review", "rating", "shipping", "discount"}
REQUIRED_KEYS = {
    "query_id", "interpreted_need", "product_judgements", "decision", "customer_response"
}


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating accidental fences or surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start:i + 1])
                if not isinstance(obj, dict):
                    raise ValueError("Top-level JSON value must be an object")
                return obj
    raise ValueError("Unbalanced JSON object")


def validate_answer(obj: dict[str, Any], query_id: str, allowed_product_ids: set[str]) -> None:
    missing = REQUIRED_KEYS - set(obj)
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")
    if obj.get("query_id") != query_id:
        raise ValueError("query_id does not match the input query")

    need = obj.get("interpreted_need")
    if not isinstance(need, dict):
        raise ValueError("interpreted_need must be an object")
    if not isinstance(need.get("category"), str):
        raise ValueError("interpreted_need.category must be a string")
    if "use_case" not in need:
        need["use_case"] = ""
    for field in ["positive_preferences", "negative_constraints", "visual_preferences", "uncertain_fields"]:
        if not isinstance(need.get(field), list):
            raise ValueError(f"interpreted_need.{field} must be a list")

    judgements = obj.get("product_judgements")
    if not isinstance(judgements, list) or not judgements:
        raise ValueError("product_judgements must be a non-empty list")
    for j in judgements:
        if not isinstance(j, dict):
            raise ValueError("Each product judgement must be an object")
        if j.get("product_id") not in allowed_product_ids:
            raise ValueError(f"Unknown/out-of-top-k product_id: {j.get('product_id')}")
        if j.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {j.get('role')}")
        if not isinstance(j.get("evidence"), list):
            raise ValueError("judgement.evidence must be a list")
        if not isinstance(j.get("constraint_violations"), list):
            raise ValueError("judgement.constraint_violations must be a list")
        if not isinstance(j.get("reason"), str) or not j.get("reason").strip():
            raise ValueError("judgement.reason must be a non-empty string")

    if obj.get("decision") not in ALLOWED_DECISIONS:
        raise ValueError(f"Invalid decision: {obj.get('decision')}")
    response = obj.get("customer_response")
    if not isinstance(response, str) or len(response.split()) < 8:
        raise ValueError("customer_response must be an understandable sentence of at least 8 words")
    low = response.lower()
    if any(term in low for term in FORBIDDEN_COMMERCE):
        raise ValueError("customer_response contains unavailable commerce claims")


def product_payload(row: pd.Series) -> dict[str, Any]:
    fields = ["product_id", "title", "product_type", "brand", "color", "material", "style", "product_text"]
    out: dict[str, Any] = {}
    for field in fields:
        val = row.get(field, "")
        if pd.isna(val):
            val = ""
        val = str(val)
        if field == "product_text" and len(val) > 700:
            val = val[:700] + "..."
        out[field] = val
    return out


def make_prompt(query: dict[str, Any], products: list[dict[str, Any]]) -> str:
    query_text = query.get("query_text") or ""
    has_image = bool(query.get("query_image_path"))
    product_lines = "\n".join(
        f"{i+1}. {json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(products)
    )
    schema = {
        "query_id": query["query_id"],
        "interpreted_need": {
            "category": "string",
            "use_case": "string",
            "positive_preferences": ["string"],
            "negative_constraints": ["string"],
            "visual_preferences": ["string"],
            "uncertain_fields": ["string"],
        },
        "product_judgements": [
            {
                "product_id": "must be one of the provided product_ids",
                "role": "exact | substitute | irrelevant",
                "evidence": ["facts copied or paraphrased only from provided metadata/query"],
                "constraint_violations": ["explicit conflicts only"],
                "reason": "short explanation",
            }
        ],
        "decision": "recommend_exact | recommend_exact_with_warning | recommend_substitute | no_good_match | ask_clarification",
        "customer_response": "clear user-facing recommendation",
    }
    return f"""You are a product search assistant. Return only one valid JSON object and no Markdown.
Use only the query and product metadata below. Do not invent unavailable facts. Do not discuss price, stock, reviews, ratings, shipping, or discounts.
The retrieval system already selected these products; judge only products listed below.

Query ID: {query['query_id']}
Query type: {query.get('query_type')}
Query text: {query_text if query_text else '[no text query]'}
Query image provided: {'yes' if has_image else 'no'}
When an image is provided, you may acknowledge that visual preferences exist, but you are not given raw pixels in this prompt; do not invent exact visual details that are not supported by metadata.

Allowed product roles: exact, substitute, irrelevant
Allowed decision labels: recommend_exact, recommend_exact_with_warning, recommend_substitute, no_good_match, ask_clarification

Required schema example/description:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Top reranked products:
{product_lines}

Requirements:
- Include query_id exactly as provided.
- Judge between 1 and {len(products)} provided products.
- Every product_id must come from the provided list.
- Respect explicit negative constraints such as 'not leather' or 'not a table lamp'.
- If material or another attribute is unknown, mark it uncertain instead of inventing it.
- customer_response must be at least one clear sentence and consistent with the judgements.
- Output JSON only.
"""


def make_repair_prompt(original_prompt: str, bad_output: str, error: str) -> str:
    return original_prompt + f"""\nYour previous output was invalid.
Validation error: {error}
Previous output:
{bad_output[:5000]}
Return a corrected JSON object only. Do not add commentary or Markdown.
"""


def fallback_answer(query: dict[str, Any], products: list[dict[str, Any]]) -> dict[str, Any]:
    first = products[0]
    qtext = (query.get("query_text") or "").strip()
    category = first.get("product_type") or "product"
    category = category.lower().replace("_", " ")
    negatives = []
    if qtext:
        for match in re.findall(r"\bnot\s+([^,.;]+)", qtext, flags=re.I):
            negatives.append("not " + match.strip())
    positives = [x.strip() for x in re.split(r",|\bwith\b|\bfor\b", qtext) if x.strip()][:4]
    visual = ["match the provided query image"] if query.get("query_image_path") else []
    judgements = []
    for i, p in enumerate(products[:3]):
        role = "substitute" if i else "exact"
        evidence = [e for e in [p.get("title"), p.get("product_type"), p.get("color"), p.get("material")] if e][:3]
        judgements.append({
            "product_id": p["product_id"],
            "role": role,
            "evidence": evidence,
            "constraint_violations": [],
            "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed.",
        })
    title = first.get("title") or "the first result"
    return {
        "query_id": query["query_id"],
        "interpreted_need": {
            "category": category,
            "use_case": qtext or "visual product search",
            "positive_preferences": positives,
            "negative_constraints": negatives,
            "visual_preferences": visual,
            "uncertain_fields": ["attributes not present in the catalog metadata"],
        },
        "product_judgements": judgements,
        "decision": "recommend_exact_with_warning" if qtext else "recommend_substitute",
        "customer_response": f"The strongest available candidate is {title}. It best matches the provided search evidence, while unspecified attributes should be treated as uncertain.",
    }


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are a product search assistant. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        attention_mask = torch.ones_like(inputs)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs.shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--out", default="outputs/final_answers.jsonl")
    ap.add_argument("--summary", default="reports/llm_output_summary.md")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--repair_attempts", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fallback_only", action="store_true", help="Skip model loading and generate validated deterministic fallback answers (debug/testing only).")
    args = ap.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}
    catalog = pd.read_parquet(args.catalog).set_index("product_id", drop=False)

    model_error = None
    tokenizer = model = None
    if args.fallback_only:
        model_error = "LLM skipped by --fallback_only (debug/testing mode)"
        print("Stage 5 LLM skipped; deterministic fallback-only mode enabled.")
    else:
        try:
            print(f"Loading Stage 5 LLM: {args.model} on {device}")
            tokenizer, model = load_model(args.model, device)
        except Exception as exc:  # fallback keeps the pipeline runnable
            model_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: LLM could not be loaded; deterministic fallback will be used. {model_error}")

    answers = []
    raw_valid = 0
    schema_valid = 0
    repaired_outputs = 0
    fallback_count = 0

    for n, (qid, query) in enumerate(queries.items(), 1):
        hits = reranked.get(qid, {}).get("results", [])[: args.top_k]
        products = []
        for hit in hits:
            pid = hit["product_id"]
            if pid in catalog.index:
                products.append(product_payload(catalog.loc[pid]))
        if not products:
            raise ValueError(f"No reranked catalog products available for {qid}")
        allowed_ids = {p["product_id"] for p in products}

        answer = None
        prompt = make_prompt(query, products)
        last_output = ""
        last_error = ""
        if model is not None:
            for attempt in range(args.repair_attempts + 1):
                try:
                    current_prompt = prompt if attempt == 0 else make_repair_prompt(prompt, last_output, last_error)
                    last_output = generate_text(tokenizer, model, current_prompt, device, args.max_new_tokens)
                    obj = extract_json_object(last_output)
                    if attempt == 0:
                        raw_valid += 1
                    validate_answer(obj, qid, allowed_ids)
                    schema_valid += 1
                    if attempt > 0:
                        repaired_outputs += 1
                    answer = obj
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

        if answer is None:
            fallback_count += 1
            answer = fallback_answer(query, products)
            validate_answer(answer, qid, allowed_ids)

        answers.append(answer)
        print(f"[{n:02d}/{len(queries)}] {qid}: {answer['decision']} ({len(answer['product_judgements'])} judgements)")

    write_jsonl(args.out, answers)

    total = len(answers)
    raw_rate = raw_valid / total if total else 0.0
    schema_rate = schema_valid / total if total else 0.0
    examples = "\n\n".join(
        "```json\n" + json.dumps(x, ensure_ascii=False, indent=2) + "\n```"
        for x in answers[:3]
    )
    summary = f"""# Stage 5 - LLM Output Summary

## Configuration

- LLM model: `{args.model}`
- Device: `{device}`
- Top reranked products per query: {args.top_k}
- Maximum repair attempts: {args.repair_attempts}
- Prompting template: JSON-only product-search assistant prompt containing the query, image-presence flag, top product metadata, allowed role labels, allowed decision labels, schema instructions, negative-constraint guidance, and a no-invention rule.

## Results

- Number of generated answers: {total}
- Raw JSON validity rate: {raw_rate:.1%}
- Schema-valid LLM rate before deterministic fallback: {schema_rate:.1%}
- Repaired outputs: {repaired_outputs}
- Deterministic fallback outputs: {fallback_count}
- Final JSON validity after repair/fallback: 100.0%
- Final schema-valid rate after repair/fallback: 100.0%

## Example Final Answers

{examples}

## Failure Case / Limitations

The LLM receives product metadata rather than raw product images at Stage 5, so it must not invent visual details. Image evidence is preserved by the upstream multimodal retrieval and fused reranking stages, while Stage 5 only structures/explains those candidates. Small local LLMs may occasionally emit malformed JSON or unsupported claims; the validator, repair loop, and deterministic fallback prevent such outputs from entering the final artifact.
"""
    if model_error:
        summary += f"\nModel loading limitation encountered during this run: `{model_error}`\n"
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(summary, encoding="utf-8")
    print(f"Wrote {total} answers to {args.out}")
    print(f"Wrote Stage 5 summary to {args.summary}")


if __name__ == "__main__":
    main()
