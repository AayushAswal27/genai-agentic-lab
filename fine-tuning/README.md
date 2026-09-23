# fine-tuning

QLoRA supervised fine-tuning of a small instruct model to enforce a fixed JSON
output schema.

## Why

The rest of this repo leans on prompting and output parsers to get structured
responses (`structured_output`, `output_parsers`). That works most of the time,
but the base model still drifts — it invents its own keys, drops fields, or wraps
the JSON in explanation. This is the case where fine-tuning earns its place: the
gap isn't missing knowledge, it's a *behavior* (always return this exact shape)
that prompting can't fully pin down.

Rule of thumb this project demonstrates: prompt first, RAG for facts, fine-tune
for behavior. Here we fine-tune the behavior.

## Task

Given a plain-text note, return exactly:

```json
{
  "summary": "one line",
  "sentiment": "positive | negative | neutral",
  "topics": ["..."],
  "action_items": ["..."]
}
```

## Stack

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- 4-bit quantization via `bitsandbytes` (NF4, double quant, bf16 compute)
- LoRA adapters via `peft` (r=16, alpha=32, on attention + MLP projections)
- Training via `trl` `SFTTrainer`
- Runs on a free Colab T4 in about 90 seconds

## Result

Same input note, before and after training.

Input:
> The payments service went down for 40 minutes during peak hours. Engineering
> traced it to a bad config push and rolled back. A few hundred transactions
> failed and will need manual retry.

Before (base model) — valid JSON, but invents its own schema:

```json
{
  "status": "success",
  "message": "Payments service went down for 40 minutes during peak hours.",
  "details": {
    "config_push_bad_config": "A few hundred transactions failed and will need manual retry."
  }
}
```

After (fine-tuned) — locks to the target schema:

```json
{
  "summary": "Payments service crashed during peak hours and config pushes were the cause.",
  "sentiment": "negative",
  "topics": ["payment-service", "fraud"],
  "action_items": ["Restore API aftercare", "Add validation in config pushes"]
}
```

Training loss dropped from 2.81 to 0.15 over 5 epochs.

## Files

- `qlora_schema_sft.ipynb` — full pipeline: load data, 4-bit base, LoRA, train,
  save adapter, before/after inference.
- `build_dataset.py` — generates `data/schema_sft.jsonl`. Add source texts here
  to grow the set.
- `data/schema_sft.jsonl` — instruction/response pairs.
- `qlora-schema-adapter/` — the trained LoRA adapter. Load it on top of the base
  model with `PeftModel.from_pretrained`.

## Known limitations

- The dataset is small (60 examples). This is enough to lock the output *format*
  cleanly, but the *content* is rough — e.g. the "after" output above tags "fraud"
  as a topic when it isn't one, and one action item is garbled. Pushing the
  dataset to 150–200 examples would sharpen content quality; the format is already
  solid.
- The generation step needs `model.config.use_cache = True` and `model.eval()`
  after training, since gradient checkpointing disables the KV cache during
  training.
