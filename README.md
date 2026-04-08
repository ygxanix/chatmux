---
title: ChatMux - RL Environment
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - chat-prioritization
  - meta-hackathon
  - llm-based
---

# ChatMux

This is an OpenEnv-compatible RL environment where AI agents learn to prioritize chat messages. It simulates a busy inbox with 10-15 users sending 3-12 messages each - the agent must identify and prioritize the most urgent messages.

## What's Here

- 10-15 chat users with varying relationship types
- Urgency detection using keywords, sender importance, and relationship strength
- OpenAI API integration for message analysis (baseline.py)
- Point-based reward system with penalties for bad decisions
- Three difficulty levels with graders

## Running It

### With LLM (full baseline)

```bash
export OPENAI_API_KEY=your_openai_api_key
python3 baseline.py --difficulty medium
```

### Server Only

```bash
python3 -m server.app --port 8000
```

### Quick Test Without LLM

```bash
python3 -c "
from server.urgent_chat_prioritizer_environment import UrgentChatPrioritizerEnvironment
from models import UrgentChatPrioritizerAction, PriorityLevel

env = UrgentChatPrioritizerEnvironment()
obs = env.reset()
action = UrgentChatPrioritizerAction(action_type='prioritize', chat_id=obs.pending_chats[0].message_id, new_priority=PriorityLevel.HIGHEST)
obs = env.step(action)
print(f'Reward: {obs.reward}, Result: {obs.last_action_result}')
"
```

## API

| Endpoint | Method |
|----------|--------|
| /reset | POST |
| /step | POST |
| /state | GET |
| /tasks | GET |
| /grader | POST |
| /baseline | POST |
| /action-schema | GET |

## Docker

```bash
docker build -t chatmux .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key chatmux
```

## What's Inside

```
├── WORK.md                    # LLM system prompt
├── baseline.py               # CLI baseline
├── grader.py                # Task graders
├── models.py                # Data models
├── server/
│   ├── app.py               # Main server
│   ├── urgent_chat_prioritizer_environment.py
│   └── Dockerfile
└── openenv.yaml
```

## What You Need

- Python 3.10+
- openenv-core
- openai
- OPENAI_API_KEY environment variable

---

Built for Meta PyTorch OpenEnv Hackathon 2026