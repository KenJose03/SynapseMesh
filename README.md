# SynapseMesh: Autonomous Multi-Agent Cognitive Architecture

**SynapseMesh** is a resilient, production-ready multi-agent system designed for complex reasoning, multi-hop web retrieval, verification, and autonomous prompt optimization.

---

## 🌟 Key Architecture & Capabilities

1. **DAG-Based Task Decomposition** (`Decomposition Agent`):
   - Dynamically analyzes complex or ambiguous queries and deconstructs them into a Directed Acyclic Graph (DAG) with explicit dependency management.
   - Guarantees termination with a synthesis and provenance mapping phase.

2. **Adaptive Orchestration** (`Master Orchestrator`):
   - Inspects running task graph state, agent history, and dependencies to dynamically coordinate execution.

3. **Multi-Hop Web Retrieval** (`Retrieval Agent`):
   - Integrates multi-hop web search (`ddgs`) with automated query refinement.
   - Structured error handling: isolates timeouts, empty search states, and malformed inputs with provenance logging (`[chunk_X]`).

4. **Span-Level Verification & Critique** (`Critique Agent`):
   - Evaluates factual claims produced by preceding agents, identifies inaccurate text spans, assigns confidence scores (0.0 to 1.0), and feeds counter-critique back into synthesis.

5. **Evidence Synthesis & Provenance Mapping** (`Synthesis Agent`):
   - Merges findings from all preceding tasks, resolves flagged contradictions, and attaches a granular provenance map linking statements to source chunks.

6. **Context Budgeting & Dynamic Memory Compression**:
   - Monitored context consumption against a strict token budget.
   - Automatically executes LLM-driven context compression when remaining budget drops below safe thresholds.

7. **Closed-Loop Meta-Optimization** (`Meta-Agent`):
   - Built-in 15-case benchmark harness across Baseline, Ambiguous, and Adversarial categories.
   - Analyzes test failures and automatically proposes or applies optimized system prompts to failing agents.

8. **Asynchronous Streaming API & Observability**:
   - FastAPI server with Server-Sent Events (SSE) streaming live execution milestones (`agent_start`, `plan_created`, `task_complete`, `final_answer`).
   - Detailed job trace inspection endpoint and Dockerized Dozzle log streaming.

---

## 🏗️ System Architecture Overview

```
                          ┌────────────────────────┐
                          │   User Query / API     │
                          └───────────┬────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Decomposition Agent    │
                        │   (DAG Task Graph Gen)    │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │    Retrieval Agent    │           │    Code / Sandbox     │
        │  (Multi-Hop Search)   │           │   (Docker Execution)  │
        └───────────┬───────────┘           └───────────┬───────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │      Synthesis Agent      │
                        │    (Provenance Mapping)   │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │       Critique Agent      │
                        │  (Span-Level Verification)│
                        └─────────────┬─────────────┘
                                      │
                         [Disagreements Detected?]
                                ├─── Yes ───► (Refinement Loop)
                                └─── No  ───► Final Answer & Trace
```

---

## 🚀 Quick Start

### 1. Environment Configuration
Create or review your `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/synapse_mesh
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=synapse_mesh
CONTEXT_BUDGET=4000
```

### 2. Run with Docker Compose (Recommended)
```bash
docker compose up --build
```
Services exposed:
- **SynapseMesh API**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Dozzle Log Viewer**: `http://localhost:8080`
- **PostgreSQL**: `localhost:5432`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/submit` | Submits a query and streams execution events via Server-Sent Events (SSE). |
| `POST` | `/submit/sync` | Synchronous JSON endpoint returning `job_id`, `final_answer`, and full `trace` (ideal for Swagger UI `/docs`). |
| `GET` | `/trace/{job_id}` | Retrieves the complete execution trace, token usage, and provenance history for a job. |
| `GET` | `/eval/latest` | Retrieves latest 2-Tier evaluation summaries (Tier 1 Golden + Tier 2 Red-Team). |
| `POST` | `/eval/re-run` | Triggers targeted Tier 1 re-evaluation on previously failed cases in the background. |
| `POST` | `/eval/red-team` | Triggers a Tier 2 Dynamic Red-Teaming run with LLM-generated queries and cross-model LLM-as-a-Judge grading. |
| `GET` | `/eval/candidates` | Lists failure candidates discovered by Tier 2 awaiting review. |
| `POST` | `/eval/candidates/{candidate_id}/promote` | Promotes a reviewed Tier 2 failure into the permanent Tier 1 Golden Dataset (`golden_dataset.json`). |
| `POST` | `/eval/candidates/{candidate_id}/reject` | Rejects a noisy or invalid synthetic failure candidate. |
| `POST` | `/prompt/approve` | Human-in-the-loop endpoint to approve or reject prompt modifications from the Meta-Agent. |

---

## 🧪 2-Tier Evaluation Harness & Self-Optimization

### Tier 1: Golden Regression Suite
Run the fixed anchor suite (`app/eval/golden_dataset.json`):
```bash
python -m app.eval.harness
# Optional: Force cross-model LLM Judge or re-run only failures
python -m app.eval.harness --judge
python -m app.eval.harness --failed-only
```

### Tier 2: Dynamic LLM Red-Teaming & Promotion Pipeline
Dynamically draft fresh adversarial, multi-hop, ambiguous, and temporal queries, grade them with an evidence-grounded LLM Judge (`nvidia/nemotron-3.5-super-vl-120b-a12b`), and capture failures:
```bash
python -m app.eval.red_team --count 4
python -m app.eval.red_team --list-candidates
python -m app.eval.red_team --promote <candidate_id>
```

### Performance Audit & Meta-Optimization
Generate the combined Tier 1 + Tier 2 performance audit:
```bash
python -m app.eval.performance_report
```

Run the closed-loop Meta-Agent to analyze Tier 1 + Tier 2 failures and optimize failing agents:
```bash
python -m app.eval.run_meta_agent
```
