# Prediction Postprocessing Rules

## 1. Parse validation
- The model output must be valid JSON.
- If parsing fails, retry or discard.

## 2. Unit type normalization
Map non-canonical outputs into the closed label set:
- thm -> theorem
- def -> definition
- prop -> proposition
- cor -> corollary

If mapping is impossible, use:
- other

## 3. Source block validation
- source_blocks must be integers
- source_blocks must belong to the current page
- remove duplicates
- sort if no explicit reading-order evidence exists

## 4. Reading order repair
If reading_order is missing or invalid:
- set reading_order = sorted(source_blocks)

## 5. Formula span repair
Each formula span must contain:
- formula_id
- text
- source_block

If source_block is missing or invalid, drop that formula span.

## 6. Statement text normalization
Allowed:
- trim whitespace
- normalize repeated spaces
- remove obvious linebreak artifacts

Not allowed:
- paraphrase
- expansion
- hallucinated assumptions
- stylistic rewriting

## 7. Confidence score
If confidence_score is missing:
- fill from model self-score if available
- otherwise use a default value such as 0.5

## 8. Deduplication
If two predictions on the same page have:
- same source_blocks
- same unit_type
- highly similar statement_text

keep only the one with higher confidence_score.
