# Model Comparison Benchmark Report

Evaluation run across 10 diverse benchmark scenarios (Normal, Conflicting, Ambiguous, Injection, and Autonomy) with a 3-second delay between calls.

| Model | Task Success | Output Validity (No Repair) | Action Selection | Avg Latency (ms) | Total In/Out Tokens | Est. List Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `openai/gpt-oss-120b` | **80.0%** (8/10) | 100.0% | 80.0% | 6234.1 ms | 24453 / 4274 | $0.00623* |
| `openai/gpt-oss-20b` | **60.0%** (6/10) | 100.0% | 80.0% | 6109.5 ms | 19504 / 2600 | $0.00224* |

\* *Pricing note: Estimated based on published list prices ($0.15/$0.60 per 1M tokens for openai/gpt-oss-120b and $0.075/$0.30 per 1M tokens for openai/gpt-oss-20b). Actual cost on the free tier is $0.*

## Deployment Recommendation

**Recommended Model for Deployment: `openai/gpt-oss-120b`**

- **Task Success**: Achieved highest end-to-end task accuracy (80.0%).
- **Contract & Schema Validity**: Maintained 100.0% first-pass validity without requiring error repair steps.
- **Action Selection**: Correctly executed required sandbox tool sequences and respected autonomy boundaries (80.0%).
- **Latency & Cost**: Average response latency of 6234.1 ms with an estimated list cost of $0.00623 across 10 multi-step workflows.
