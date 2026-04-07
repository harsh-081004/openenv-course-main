---
title: Email Triage Environment
emoji: "📧"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Email Triage and Prioritization Environment

This project is an OpenEnv-compliant RL environment for a real operational task: triaging inbound support and business emails. The agent must classify urgency, route to the correct department, prioritize queue position, and decide whether to escalate based on SLA and risk.

## Why This Is Useful

Email triage is a real workflow used by support, finance, sales, HR, and security teams. This environment is designed to evaluate agents on practical decision quality, not toy gameplay.

## Unique Mechanics

1. Queue-aware triage: the agent predicts `queue_position` (1 to 3), and reward depends on correct prioritization.
2. Compliance-aware routing: emails containing sensitive risk signals (for example SSN or PCI context) enforce safer routing behavior.
3. SLA-aware escalation engine: the agent predicts `escalate` and is rewarded for correct escalation under deadline and impact pressure.

## Action Space

`EmailAction` fields:

- `task_id` (string): `task-urgency`, `task-routing`, `task-full-triage`
- `urgency` (optional string): `urgent`, `normal`, `low`
- `department` (optional string): `billing`, `technical`, `sales`, `hr`, `general`
- `summary` (optional string): one-sentence response summary
- `queue_position` (optional integer): `1`, `2`, `3`
- `escalate` (optional boolean): escalation decision
- `notes` (optional string): reasoning notes

## Observation Space

`EmailObservation` fields:

- `email_id`, `sender`, `subject`, `body`
- `task_id`, `task_name`
- `available_urgency_labels`, `available_departments`
- `compliance_risk` (bool)
- `sender_tier` (`standard`, `enterprise`, `strategic`)
- `business_impact` (0-100)
- `minutes_to_breach` (int)
- `queue_position`, `queue_size`
- `step_count`, `max_steps`

## Tasks and Graders

1. Task 1 (Easy): Urgency Classification
Objective: predict urgency label.
Grader: exact match, score `0.0` or `1.0`.

2. Task 2 (Medium): Department Routing
Objective: choose correct department.
Grader: exact match `1.0`, related department partial `0.5`.

3. Task 3 (Hard): Full Triage
Objective: urgency + department + summary.
Grader: weighted score: urgency `0.3`, department `0.3`, summary `0.4`.

## Reward Function

- Dense partial rewards for valid action fields.
- Grader components included in shaped reward.
- Queue bonus when `queue_position` is correct.
- Escalation bonus for correct `escalate` decision.
- Penalties for missing fields, refusals, repeated actions.
- Compliance penalty for unsafe routing on risk-sensitive emails.

## Full Workflow

### 1. Local Run

```bash
cd my_env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 2. Local API Smoke Test

```bash
curl -s http://localhost:7860/
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task-urgency"}'
curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"task_id":"task-urgency","urgency":"urgent"}'
curl -s http://localhost:7860/state
curl -s http://localhost:7860/tasks
curl -s -X POST http://localhost:7860/grader -H "Content-Type: application/json" -d '{"task_id":"task-urgency","action":{"task_id":"task-urgency","urgency":"urgent"}}'
curl -s -X POST http://localhost:7860/baseline
```

### 3. Validator

```bash
python validate.py
```

### 4. Inference Script (Mandatory Env + OpenAI Client)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
export HF_TOKEN="<your_hf_token>"

# checker-compatible root script (must be named inference.py at repo root)
python inference.py --base-url http://localhost:7860
```

The root inference script uses OpenAI Client for all LLM calls and reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables.

### 5. Docker Build and Run

```bash
docker build -t email-triage-env .
docker run --rm -p 7860:7860 email-triage-env
```

### 6. HF Space Deployment Check

After deploying your Docker Space, verify:

```bash
curl -s https://<your-space-url>/
curl -s -X POST https://<your-space-url>/reset -H "Content-Type: application/json" -d '{"task_id":"task-urgency"}'
OPENENV_BASE_URL="https://<your-space-url>" python validate.py
```

## Visible Output for Uniqueness

Use this pair to demonstrate the escalation engine and compliance logic.

Positive case (correct escalation + routing + queue):

```bash
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task-full-triage","email_id":"email-009"}'
curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"task_id":"task-full-triage","urgency":"urgent","department":"hr","summary":"We are escalating the PCI retention issue and initiating compliance containment now.","queue_position":1,"escalate":true}'
```

Negative case (missed escalation + risky misroute + wrong queue):

```bash
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task-full-triage","email_id":"email-009"}'
curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"task_id":"task-full-triage","urgency":"urgent","department":"technical","summary":"We will check this.","queue_position":3,"escalate":false}'
```

Expected difference:

- Positive output includes components like `escalation_correct` and `queue_match`.
- Negative output includes penalties like `missed_escalation`, `compliance_misroute`, and `queue_mismatch`.

## Dataset Coverage

The environment includes diverse operational emails, including:

- billing failures and quarter-end reconciliation
- API outages and logistics pipeline incidents
- HR/compliance cases with sensitive data risk
- sales pricing inquiries and low-priority feature requests

Added high-stakes examples:

- `email-007`: logistics label generation outage
- `email-008`: quarter-end credit memo mismatch
- `email-009`: PCI data retention compliance incident

## Baseline Scores (Example)

| Task | Score |
| --- | --- |
| task-urgency | 1.000 |
| task-routing | 0.667 |
| task-full-triage | 0.803 |

## Project Structure

```text
my_env/
  app.py
  Dockerfile
  openenv.yaml
  requirements.txt
  README.md
  inference.py
  validate.py
  models.py
  tasks.py
  graders.py
  environment.py
  inference.py
  server/
    app.py
    requirements.txt
```
