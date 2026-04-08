#!/usr/bin/env python3
"""
ChatMux Inference Script
Used for presubmission validation - follows exact sample format
"""

import os
import sys
import json
import logging

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("START")
    
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=API_BASE_URL
    )

    difficulty = os.environ.get("DIFFICULTY", "medium")
    max_steps = int(os.environ.get("MAX_STEPS", "20"))
    
    from server.urgent_chat_prioritizer_environment import UrgentChatPrioritizerEnvironment
    from models import UrgentChatPrioritizerAction, PriorityLevel

    env = UrgentChatPrioritizerEnvironment(difficulty=difficulty)
    obs = env.reset()

    total_reward = 0.0
    step = 0
    message_analysis = {}

    while not obs.done and step < max_steps:
        step += 1
        logger.info(f"STEP {step}")
        
        if not obs.pending_chats:
            break

        messages_to_analyze = obs.pending_chats[:10]
        analysis_results = []

        for msg in messages_to_analyze:
            if msg.message_id in message_analysis:
                analysis_results.append((msg.message_id, message_analysis[msg.message_id]))
            else:
                prompt = f"""Analyze the urgency of this message.

MESSAGE: "{msg.message_text}"
SENDER: {msg.sender_name} (type: {msg.sender_type.value})
RELATIONSHIP STRENGTH: {msg.relationship_strength:.2f}
TIME SINCE ARRIVAL: {msg.time_since_arrival:.1f} minutes
KEYWORDS: {', '.join(msg.urgency_keywords) if msg.urgency_keywords else 'None'}

Respond with JSON: {{"urgency": 0-100, "reason": "brief explanation"}}"""

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes message urgency."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.0,
                    )
                    data = json.loads(response.choices[0].message.content)
                    analysis = {
                        "urgency": data.get("urgency", 50),
                        "reason": data.get("reason", "")
                    }
                except Exception as e:
                    analysis = {"urgency": 50, "reason": f"error: {e}"}

                message_analysis[msg.message_id] = analysis
                analysis_results.append((msg.message_id, analysis))

        analysis_results.sort(key=lambda x: x[1].get("urgency", 0), reverse=True)

        if analysis_results:
            best_id = analysis_results[0][0]
            pct = analysis_results[0][1].get("urgency", 0)
            priority = PriorityLevel.HIGHEST if pct >= 70 else PriorityLevel.HIGH if pct >= 50 else PriorityLevel.NORMAL if pct >= 30 else PriorityLevel.LOW

            action = UrgentChatPrioritizerAction(
                action_type="prioritize",
                chat_id=best_id,
                new_priority=priority
            )
        else:
            break

        obs = env.step(action)
        total_reward += obs.reward

    logger.info(f"END total_reward={total_reward:.2f} steps={step}")
    return {"total_reward": total_reward, "steps": step}


if __name__ == "__main__":
    main()