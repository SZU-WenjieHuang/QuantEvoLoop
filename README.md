# QuantEvoLoop

**Multi-Agent Parallel Evolutionary Framework for Quantitative Strategy Optimization**

[![CI](https://github.com/SZU-WenjieHuang/QuantEvoLoop/actions/workflows/ci.yml/badge.svg)](https://github.com/SZU-WenjieHuang/QuantEvoLoop/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<p align="center">
  <img src="docs/architecture.png" alt="QuantEvoLoop Architecture" width="800">
</p>

<p align="center">
  <img src="docs/evolution_loop.gif" alt="Evolution Loop" width="800">
</p>

## What is QuantEvoLoop?

QuantEvoLoop is an **open-source multi-agent evolutionary framework** that automatically optimizes quantitative trading strategies through hypothesis-driven mutation, rigorous statistical validation, and RL-guided selection.

Unlike traditional hyperparameter optimizers (Optuna, Hyperopt), QuantEvoLoop treats strategy code as a **living organism** — it diagnoses weaknesses, proposes structural mutations, and evolves the strategy through natural selection with academic-grade statistical gates.

## Key Innovations

### 1. Agent-as-Backend Architecture
Instead of building a custom LLM agent loop, QuantEvoLoop leverages **Claude Code CLI / Codex CLI / Qoder CLI** as "code mutation engines". These production-grade agents provide ReAct reasoning, AST-aware code editing, error recovery, and context management — capabilities that would take months to replicate.

```mermaid
graph LR
    A[Diagnose] --> B[Hypothesize]
    B --> C[Mutate via CLI Agent]
    C --> D[Evaluate 5-Gate]
    D --> E{Promote?}
    E -->|Yes| F[New Champion]
    E -->|No| G[Dead End / Reject]
    F --> A
```

### 2. 5-Layer Statistical Gate Pipeline
Every candidate must survive 5 sequential statistical filters before promotion:

```mermaid
graph TB
    C[Candidate Strategy]
    C --> G1
    G1[G1: Hard Constraints
    Risk regression, trade count >= 50,
    overfit gap check]
    G1 -->|Pass| G2
    G1 -->|Fail| R[Reject]
    G2[G2: Composite Score
    min-segments Sharpe + CAGR - MaxDD > 0]
    G2 -->|Pass| G3
    G2 -->|Fail| R
    G3[G3: Probabilistic Sharpe Ratio
    PSR test >= 0.85, PSR train >= 0.80]
    G3 -->|Pass| G4
    G3 -->|Fail| R
    G4[G4: Bootstrap CI + Drop-top-K
    CI lower bound > 0, top-2 < 50% PnL]
    G4 -->|Pass| G5
    G4 -->|Fail| R
    G5[G5: Walk-Forward + Holdout OOS
    3-fold consistency, cross-regime check]
    G5 -->|Pass| P[Promote]
    G5 -->|Fail| R
```

<p align="center">
  <img src="docs/pipeline.png" alt="5-Layer Statistical Gate Pipeline" width="700">
</p>

### 3. RL-Inspired UCB1 Selection
Mutation types are treated as **multi-armed bandit arms**. UCB1 balances exploration of untried mutations vs exploitation of known-good ones, adapting the search strategy based on historical success rates.

### 4. Multi-Lane Tournament Selection
N parallel SubAgent lanes evolve the strategy independently. Each generation, a tournament selects the best candidate across all lanes for potential promotion.

<p align="center">
  <img src="docs/tournament.gif" alt="Multi-Lane Tournament Selection" width="700">
</p>

### 5. Cross-Campaign Knowledge Accumulation
Structured knowledge base tracks which mutation types work (High-EV) and which are dead ends, informing future campaign planning and avoiding repeated failures.

### 6. IM-First Monitoring
Telegram / Discord / Webhook notifications for real-time evolution monitoring — know when a new champion is crowned without watching the terminal.

### Evolution Loop

```mermaid
sequenceDiagram
    participant C as Coordinator
    participant L as LeadAgent
    participant S as SubAgent x N
    participant J as JudgeAgent
    participant T as Tournament
    participant P as Promoter

    C->>C: Bootstrap Champion (train/test/holdout)
    loop Each Generation
        C->>L: Diagnose champion trades
        L-->>C: Weaknesses + Hypotheses
        C->>S: Launch N parallel mutations
        S-->>C: Mutated strategies
        C->>J: Evaluate each candidate (5 gates)
        J-->>C: Verdicts + Scores
        C->>T: Tournament selection
        T-->>C: Winner (or none)
        alt Winner found
            C->>P: Promote new champion
            C->>C: Update baseline metrics
        else No winner
            C->>C: Record rejection
        end
        C->>C: Save state + notify IM
    end
```

## Architecture

```mermaid
graph TB
    subgraph CLI["CLI / Gateway"]
        direction LR
        INIT[init] --- RUN[run] --- DIAG[diagnose] --- STATUS[status] --- GW[gateway]
    end
    subgraph Core["Core Modules"]
        direction LR
        subgraph Agents
            LeadAgent
            SubAgent
            JudgeAgent
        end
        subgraph Selection
            UCB1
            Tournament
            Population
        end
        subgraph Evolution
            Campaign
            Knowledge
            DeadEnds
            Promoter
        end
        subgraph Evaluation
            PSR
            Bootstrap
            WalkForward
            Holdout
        end
        subgraph Channels
            Telegram
            Discord
            Webhook
        end
    end
    subgraph Backends["Backends (CLI Adapters)"]
        direction LR
        CC[Claude Code] --- CX[Codex CLI] --- QC[Qoder CLI]
    end
    subgraph Engines["Backtest Engines"]
        direction LR
        FT[Freqtrade] --- MK[Mock] --- BT[Backtrader] --- ZL[Zipline]
    end
    CLI --> Core
    Core --> Backends
    Core --> Engines
```

## Prerequisites

- **Python** 3.11+
- **One of**: [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) | [Codex CLI](https://github.com/openai/codex) | [Qoder CLI](https://qoder.com)
- **Freqtrade** (for real backtesting, or use Mock engine for testing)

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Initialize workspace with your Freqtrade strategy
quantevoloop init \
    --strategy path/to/your_strategy.py \
    --backend claude-code \
    --workspace ./evo_workspace

# Check backend availability
quantevoloop backend-check

# View workspace status
quantevoloop status

# Start the evolution loop (3 parallel lanes, max 20 generations)
quantevoloop run --max-gens 20 --lanes 3

# Monitor with Streamlit dashboard
streamlit run src/quantevoloop/dashboard/__init__.py -- --workspace ./evo_workspace

# Start with Gateway + Telegram bot (interactive commands: /pause /diagnose /history)
quantevoloop gateway --engine freqtrade
```

### Docker Quick Start

```bash
# Copy environment file and fill in your API keys
cp .env.example .env

# Run evolution (mock engine, for testing)
docker compose up evolution

# Run with Freqtrade engine
docker compose --profile freqtrade up freqtrade-evolution

# Launch dashboard
docker compose --profile dashboard up dashboard

# Launch gateway with Telegram bot
docker compose --profile gateway up gateway

# Run tests
docker compose --profile tests up tests
```

## Configuration

All settings live in `evo_workspace/config.yaml`. Key sections:

```yaml
workspace_dir: ./evo_workspace
strategy_path: ./my_strategy.py
backend:
  type: claude-code          # claude-code | codex | qoder-cli
  max_turns: 15
  timeout_seconds: 300
n_lanes: 3                   # parallel evolution lanes
max_campaign_iter: 20        # max generations per campaign
data_splits:
  train_start: "20220101"
  train_end: "20240701"
  test_start: "20240701"
  test_end: "20260101"
  holdout_start: "20210101"
  holdout_end: "20220101"
im:
  telegram_enabled: true
  telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"
cost:
  budget_usd: 50.0           # stop when budget exhausted
```

## Project Structure

```
quantevoloop/
├── src/quantevoloop/
│   ├── agents/          # LeadAgent, SubAgent, JudgeAgent
│   ├── backends/        # Claude Code / Codex / Qoder CLI adapters
│   ├── channels/        # Telegram / Discord / Webhook adapters
│   ├── dashboard/       # Streamlit real-time monitoring UI
│   ├── engine/          # BacktestEngine ABC + Freqtrade + Mock + Backtrader + Zipline
│   ├── evaluation/      # PSR, Bootstrap CI, DSR, Scorer, Diagnostics, WFA
│   ├── evolution/       # Campaign, Knowledge, DeadEnds, Promoter, State
│   ├── gateway/         # EventBus, LaneQueue, Coordinator (main loop)
│   ├── selection/       # UCB1 Bandit, Tournament, Population, Reward
│   ├── workspace/       # Hypothesis taxonomy, workspace init
│   ├── templates/       # Jinja2 prompt templates (mutation, diagnose, judge)
│   ├── config.py        # Pydantic config model
│   └── cli.py           # Click CLI entry point
├── tests/               # pytest unit + E2E integration tests
├── examples/            # Sample configs + workspace data
├── paper/               # Academic paper (LaTeX)
└── .github/workflows/   # CI pipeline
```

## Statistical Rigor

QuantEvoLoop implements the full statistical validation pipeline from the quantitative finance literature:

| Method | Reference | Purpose |
|--------|-----------|---------|
| Probabilistic Sharpe Ratio | Bailey & López de Prado (2014) | Is the true Sharpe > 0? |
| Bootstrap CI | Efron & Tibshirani (1993) | Confidence interval for Sharpe |
| Deflated Sharpe Ratio | Bailey & López de Prado (2014) | Multi-testing correction |
| Drop-top-K | Concentration risk | Top-2 winners don't carry PnL |
| Walk-Forward Analysis | Pardo (2008) | 3-fold temporal robustness |
| Holdout OOS | Standard practice | Cross-regime validation |

## Testing

```bash
# Run all tests (59 tests)
pytest tests/ -v

# Run specific module
pytest tests/test_evaluation.py -v

# Run E2E integration tests
pytest tests/test_e2e_coordinator.py -v

# Run bot command tests
pytest tests/test_bot_commands.py -v
```

## Roadmap

- [x] Core framework (evaluation, evolution, selection, agents, channels)
- [x] CLI + Mock engine + 45 unit/E2E tests
- [x] Freqtrade engine integration (subprocess BT, multi-format result parsing)
- [x] Real multi-lane parallel execution with asyncio + retry
- [x] Walk-forward fold execution (dynamic 3-fold from config)
- [x] Champion bootstrap baseline (train/test/holdout backtests)
- [x] Diagnosis → Hypothesis → Mutation full pipeline
- [x] Backend interface alignment (BackendMutationContext)
- [x] Web UI dashboard (Streamlit real-time monitoring)
- [x] Academic paper skeleton (LaTeX, 3 tables: config, results, mutation analysis)
- [x] Additional backtest engine adapters (Backtrader, Zipline — skeleton)
- [x] IM interactive commands (/pause, /diagnose, /history, /champion, /bandit, /status)
- [x] Docker Compose for one-click deployment (evolution + gateway + dashboard)
- [x] Sample Freqtrade strategy for e2e testing (EMA+RSI trend-following)
- [x] CLI --engine flag (mock / freqtrade / backtrader / zipline)
- [ ] Real Freqtrade end-to-end run with BTC/USDT data (requires market data)
- [ ] Experimental results with real strategies (paper Table 1 — requires running experiments)

## Contributing

Contributions are welcome! Please read the architecture overview and submit PRs with tests.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{quantevoloop2025,
  title={QuantEvoLoop: Multi-Agent Parallel Evolutionary Framework for Quantitative Strategy Optimization},
  author={QuantEvoLoop Contributors},
  year={2025},
  url={https://github.com/SZU-WenjieHuang/QuantEvoLoop}
}
```
