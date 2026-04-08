---
title: ChatMux
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# ChatMux

**Meta PyTorch OpenEnv Hackathon 2026**

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-ChatMux-blue?style=flat-square)](https://huggingface.co/spaces/Ygxanix/chatmux)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square)
![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.2+-green?style=flat-square)

</div>

## What is ChatMux?

ChatMux is an **OpenEnv-compatible Reinforcement Learning environment** that trains AI agents to automatically prioritize chat messages by urgency. The environment simulates a busy inbox where 10-15 users send 3-12 messages each - the agent must identify and prioritize the most urgent messages.

### Why This Matters?

- **Real-world RL task**: Message prioritization is a common problem in virtual assistants, email systems, and chat apps
- **Keyword-based urgency detection**: Uses 11,578+ keywords across CRITICAL, HIGH, MEDIUM, LOW, and SPAM categories
- **Fuzzy matching**: Uses rapidfuzz for better keyword matching even with typos
- **LLM-powered baseline**: Integrates with OpenAI API for intelligent message analysis

---

## Quick Start

### Run Locally

```bash
# Clone the repo
git clone https://github.com/ygxanix/chatmux.git
cd chatmux

# Install dependencies
pip install openenv-core openai rapidfuzz pydantic fastapi uvicorn

# Run server
python app.py

# Or run with custom port
python app.py --port 8000

# Test baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_openai_api_key
python baseline.py --difficulty medium
```

### Try Online

**HuggingFace Space**: https://ygxanix-chatmux.hf.space

```bash
# Test API
curl -X POST https://ygxanix-chatmux.hf.space/reset
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode with 10-15 users, 3-12 messages each |
| `/step` | POST | Execute an action (prioritize, mark_spam) |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List available grading tasks |
| `/grader` | POST | Grade agent performance |
| `/baseline` | POST | Run LLM baseline inference |
| `/action-schema` | GET | Get action schema |
| `/docs` | GET | Swagger API documentation |

---

## How It Works

### The RL Environment

1. **Reset**: Environment generates 10-15 users with varying relationship types:
   - Family, Boss, Close Friend, Acquaintance, Unknown, Group, Bot

2. **Each message has**:
   - Message text
   - Sender type (determines base priority)
   - Relationship strength (0.0-1.0)
   - Time since arrival
   - Urgency keywords (detected from 11,578+ keywords)
   - Promotional/forwarded flags

3. **Agent Actions**:
   - `prioritize`: Move message to prioritized queue
   - `mark_spam`: Mark message as spam

4. **Rewards**:
   - +25: Correctly prioritize urgent message
   - +15: Prioritize important sender (boss/family)
   - +10: Prioritize good contact
   - -20: Prioritize spam/bot
   - -25: Mark important message as spam
   - -1: Per step (time penalty)

### Keyword Detection

The environment uses **11,578 keywords** across categories:
- **CRITICAL**: emergency, urgent, help, asap
- **HIGH**: deadline, important, critical
- **MEDIUM**: question, help, quick
- **SPAM**: winner, prize, offer, congratulations

Fuzzy matching with rapidfuzz handles typos and variations.

---

## Task Graders

Three difficulty levels with programmatic graders:

| Task | Difficulty | Description |
|------|------------|-------------|
| `easy_urgent_vs_spam` | Easy | Classify obvious urgent vs spam |
| `medium_mixed_urgency` | Medium | Handle conflicting signals |
| `hard_conflicting_signals` | Hard | High volume, conflicting signals |

### Example Grading

```bash
curl -X POST https://ygxanix-chatmux.hf.space/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy_urgent_vs_spam",
    "prioritized_messages": [...],
    "all_messages": [...]
  }'
```

---

## Docker Deployment

### Build

```bash
docker build -t chatmux .
```

### Run

```bash
# Without LLM
docker run -p 8000:8000 chatmux

# With LLM
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key chatmux
```

### HuggingFace Spaces

The project is deployed at: **https://ygxanix-chatmux.hf.space**

---

## Project Structure

```
chatmux/
├── app.py                       # Main FastAPI server (entry point)
├── baseline.py                  # CLI baseline inference script
├── client.py                    # OpenEnv client
├── grader.py                    # Task graders (3 difficulties)
├── models.py                    # Data models (Action, Observation)
├── WORK.md                      # LLM system prompt
├── README.md                    # This file
├── index.html                   # Landing page
├── pyproject.toml               # Python project config
├── openenv.yaml                 # OpenEnv configuration
├── Dockerfile                   # Main Dockerfile
├── chat_keywords/               # 11,578 urgency keywords
│   ├── priority_config.json
│   ├── emergency_critical_keywords.txt
│   ├── technical_support_keywords.txt
│   └── ...
└── server/
    └── urgent_chat_prioritizer_environment.py  # Core RL environment
```

---

## Requirements

- Python 3.10+
- openenv-core >= 0.2.2
- openai >= 1.0.0
- rapidfuzz >= 3.0.0
- pydantic
- fastapi
- uvicorn

For LLM baseline: `OPENAI_API_KEY` environment variable

---

## Testing

```bash
# Test environment
python -c "
from server.urgent_chat_prioritizer_environment import UrgentChatPrioritizerEnvironment
from models import UrgentChatPrioritizerAction, PriorityLevel

env = UrgentChatPrioritizerEnvironment(difficulty='easy')
obs = env.reset()
print(f'Loaded {len(obs.pending_chats)} messages')

action = UrgentChatPrioritizerAction(
    action_type='prioritize',
    chat_id=obs.pending_chats[0].message_id,
    new_priority=PriorityLevel.HIGHEST
)
obs = env.step(action)
print(f'Reward: {obs.reward}')
"
```

---

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

See LICENSE file in root directory.

---

## Links

- **HuggingFace Space**: https://ygxanix-chatmux.hf.space
- **HuggingFace Model Repo**: https://huggingface.co/Ygxanix/chatmux-model
- **GitHub**: https://github.com/ygxanix/chatmux
- **OpenEnv Course**: https://github.com/huggingface/openenv-course
- **Meta-PyTorch OpenEnv**: https://github.com/meta-pytorch/OpenEnv