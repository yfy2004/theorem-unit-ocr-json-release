# Theorem Unit Benchmark Starter Kit v2

This starter kit helps you run a pilot version of the OCR JSON -> theorem unit benchmark described in your NeurIPS 2026 E&D draft.

## What is included

- Pilot annotation CSV templates
- Prompt templates for stage 1 and stage 2
- CSV -> gold JSONL conversion script
- LLM raw-output postprocessing for stage 1 and stage 2
- Evaluation scripts for structure, integrity, and downstream readiness
- Table export scripts for Table 1/2/3/4/8/9/10/11
- A full pipeline runner
- A paper macro filler for synchronizing key numbers into `paper/main.tex`

## Suggested first run

1. Fill `annotations/pilot/pilot_units_annotation_sheet.csv`
2. Convert it into `data/gold/gold_units_test_pilot.jsonl`
3. Generate mock predictions with `scripts/make_mock_predictions.py`
4. Run `scripts/eval_all.py`
5. Export tables with `scripts/export_tables.py`

## Minimal commands

### 1. Convert pilot CSV to gold JSONL

```bash
python scripts/csv_to_gold_jsonl.py   --input_csv annotations/pilot/pilot_units_annotation_sheet.csv   --output_jsonl data/gold/gold_units_test_pilot.jsonl   --doc_id probability_theory
```

### 2. Make mock predictions

```bash
python scripts/make_mock_predictions.py   --gold data/gold/gold_units_test_pilot.jsonl   --output predictions/mock/pred_copy_gold_test_pilot.jsonl   --mode copy_gold
```

### 3. Evaluate a single system

```bash
python scripts/eval_all.py   --gold data/gold/gold_units_test_pilot.jsonl   --pred predictions/mock/pred_copy_gold_test_pilot.jsonl   --output outputs/metrics/mock_copy_gold_test_all.json   --system_name "Mock copy-gold"
```

## Expected OCR JSON naming

The scripts assume page OCR files follow:

- `data/ocr_json/p021.json`
- `data/ocr_json/p022.json`
- etc.

Each OCR JSON should contain one of the following top-level block arrays:

- `blocks`
- `items`
- `layout_blocks`
- `elements`

Each block should ideally have `block_id` or `id`. If not, the scripts fall back to positional indexing.

## Notes

- The kit prioritizes stable execution over sophisticated modeling.
- The downstream readiness script currently uses a fixed proxy, which you can later replace with your real Structured Theorem Parser.
- You should freeze OCR inputs before large-scale annotation.
