# WORK.md - How the AI System Works

## Overview

This environment trains an AI to automatically prioritize chat messages by urgency. When messages come in, the AI analyzes each message and decides which ones are most important to respond to first.

---

## How the System Works

### 1. Messages Arrive (Simulated)
- 10-15 users send messages (3-12 each)
- Each message has: text, sender type, relationship, time, keywords

### 2. AI Analyzes Each Message
For each pending message, the AI must determine:
- **Urgency % (0-100%)** - How important is this to respond now?

### 3. AI Assigns Urgency Percentage

The AI analyzes based on these factors:

| Factor | How It Affects Urgency |
|--------|---------------------|
| Urgency keywords | "urgent", "ASAP", "emergency", "help", "deadline", "now" = +40% |
| Sender type | Boss/Family = +30%, Close friend = +15%, Unknown = -20% |
| Relationship | 0.0-1.0 scale, higher = more important |
| Time | Newer messages = slightly higher |
| Is spam/promotional | -50% if promotional or bot |

### 4. AI Sorts by Urgency
- Messages with highest urgency % go to top
- AI prioritizes the top one first

### 5. Reward is Given
The system watches the AI's work and gives rewards:

| Action | Reward |
|--------|--------|
| Priority urgent message to top | +25 |
| Priority important contact | +15 |
| Priority regular important | +10 |
| Priority spam to top | -20 |
| Mark important as spam | -15 |
| Each step (time penalty) | -1 |

---

## What the AI Must Do

### Input (from environment):
```
pending_chats: [
  {
    "message_id": "msg_0_1",
    "message_text": "URGENT: Need help ASAP!",
    "sender_name": "Boss",
    "sender_type": "boss",
    "relationship_strength": 0.9,
    "time_since_arrival": 5.0,
    "urgency_keywords": ["urgent", "asap"]
  },
  ...
]
```

### Expected Output:
```
{
  "action_type": "prioritize",
  "chat_id": "msg_0_1",  // The most urgent message
  "new_priority": "highest"
}
```

---

## Urgency Calculation Guide

Use this to assign urgency %:

```
90-100%: EMERGENCY
  - Contains emergency/urgent keywords AND from boss/family
  - "Emergency! Need help now!" from Mom/Boss

70-89%: HIGH PRIORITY
  - Contains urgency keywords (urgent, ASAP, deadline)
  - OR important sender (boss, family)

50-69%: NORMAL
  - Regular important messages
  - Close friends, colleagues

30-49%: LOW
  - Casual messages
  - Non-urgent

10-29%: VERY LOW
  - Forwarded messages
  - Group messages

0-9%: SPAM
  - Promotional content
  - Bot messages
  - "Congratulations you won!"
```

---

## Example AI Decision Making

### Example 1: Clear Emergency
```
Message: "URGENT: Need help ASAP! Call me now!"
Sender: Boss
Keywords: ["urgent", "asap", "now"]
Relationship: 0.9

AI Analysis:
- Has urgent keywords: +40%
- Boss sender: +30%
- High relationship: +20%
- Total: 90%

-> Priority: HIGHEST (90%)
-> Action: prioritize this message
-> Reward: +25 (correct!)
```

### Example 2: Clear Spam
```
Message: "Congratulations! You've WON a prize!"
Sender: Bot
Keywords: []
Relationship: 0.1

AI Analysis:
- No keywords: 0%
- Bot sender: -20%
- Low relationship: -10%
- Total: -30% -> clamp to 0%

-> Priority: SPAM
-> Action: mark_spam
-> Reward: +10 (correct!)
```

### Example 3: Conflicting Signals
```
Message: "Hey, can you check this when free?"
Sender: Close Friend
Keywords: []
Relationship: 0.6

AI Analysis:
- No urgent keywords: 0%
- Close friend: +15%
- Medium relationship: +10%
- Total: 25%

-> Priority: LOW (25%)
-> Action: snooze/mark low
-> Reward: +5 (good decision)
```

---

## How to Test

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_key

# Run baseline
python baseline.py

# Output shows:
# Step 1: Selected msg_0_1 (85% urgency)
#   Reason: urgent keyword + boss sender
#   Reward: +25
```

---

## Keys to Success

1. **Look for urgency keywords** - Words like "urgent", "ASAP", "emergency", "deadline"
2. **Check sender type** - Boss/Family = highest, Unknown/Bot = lowest
3. **Consider relationship** - Higher relationship = more important to respond
4. **Watch for spam** - Promotional, bot messages = low priority
5. **-act fast** - Each step costs -1, so don't over-think

---

## Files in This Project

- `baseline.py` - AI baseline that uses OpenAI to analyze urgency
- `grader.py` - Grades AI performance (score 0.0-1.0)
- `server/urgent_chat_prioritizer_environment.py` - Core environment
- `models.py` - Data models (Action, Observation)
- `openenv.yaml` - OpenEnv configuration