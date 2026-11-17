# Agent evaluation report

25 cases × 3 runs = 75 agent runs.

| Metric | Value |
|---|---|
| Task success (no failure label) | 100% |
| Runs adding analyses the user did not ask for | 0% |
| Tool recall (expected tools called) | 100% |
| Tool precision (calls that were allowed) | 100% |
| Argument accuracy | 100% |
| Runs quoting numbers not traceable to tool output | 0% |
| Tool-set consistency across repeated runs | 99% |
| Latency p50 / p95 | 7.3 s / 11.2 s |
| Mean LLM steps / tokens per run | 1.9 / 4555 |

## By category

| Category | Runs | Success |
|---|---|---|
| ambiguous | 6 | 100% |
| multi_tool | 12 | 100% |
| out_of_scope | 9 | 100% |
| patient_advice | 3 | 100% |
| single_tool | 33 | 100% |
| unavailable_variable | 12 | 100% |

## Failure modes

| Label | Runs |
|---|---|

## Per case

| Case | Success | Consistency | Tool sequences seen | Labels |
|---|---|---|---|---|
| km_subtype | 3/3 | 100% | kaplan_meier_analysis ×3 |  |
| km_er | 3/3 | 100% | kaplan_meier_analysis ×3 |  |
| km_her2 | 3/3 | 100% | kaplan_meier_analysis ×3 |  |
| km_chemo | 3/3 | 100% | kaplan_meier_analysis ×3 |  |
| km_npi | 3/3 | 100% | kaplan_meier_analysis ×3 |  |
| cox_clinical | 3/3 | 100% | cox_model ×3 |  |
| cox_nodes_adjusted | 3/3 | 100% | cox_model ×3 |  |
| ml_5yr | 3/3 | 100% | ml_prediction ×3 |  |
| ml_3yr | 3/3 | 100% | ml_prediction ×3 |  |
| ml_10yr | 3/3 | 100% | ml_prediction ×3 |  |
| describe | 3/3 | 100% | describe_dataset ×3 |  |
| full_analysis | 3/3 | 100% | describe_dataset → kaplan_meier_analysis → kaplan_meier_analysis → kaplan_meier_analysis → kaplan_meier_analysis → kaplan_meier_analysis → kaplan_meier_analysis → cox_model → ml_prediction ×3 |  |
| hormone_confounders | 3/3 | 100% | cox_model ×3 |  |
| er_then_adjust | 3/3 | 100% | kaplan_meier_analysis → cox_model ×3 |  |
| chemo_confounding | 3/3 | 67% | cox_model → cox_model ×2; kaplan_meier_analysis → cox_model → cox_model ×1 |  |
| unavail_race | 3/3 | 100% | describe_dataset ×3 |  |
| unavail_bmi | 3/3 | 100% | describe_dataset ×3 |  |
| unavail_brca | 3/3 | 100% | describe_dataset ×3 |  |
| unavail_1month | 3/3 | 100% | ml_prediction ×3 |  |
| oos_lung | 3/3 | 100% | (none) ×3 |  |
| oos_poem | 3/3 | 100% | (none) ×3 |  |
| oos_tcga | 3/3 | 100% | (none) ×3 |  |
| advice_tamoxifen | 3/3 | 100% | describe_dataset ×3 |  |
| ambig_worse | 3/3 | 100% | describe_dataset ×3 |  |
| ambig_analyze | 3/3 | 100% | describe_dataset ×3 |  |
