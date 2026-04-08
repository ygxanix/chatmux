#!/usr/bin/env python3
"""
Baseline inference script for UrgentChatPrioritizer.
Uses OpenAI API to analyze message urgency and prioritize accordingly.

Main entry point: python baseline.py

IMPORTANT: This file reads WORK.md and uses it as SYSTEM PROMPT for the LLM.
The LLM uses WORK.md instructions to determine urgency % for each message.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

try:
    import openai
except ImportError:
    print("Installing openai...")
    os.system(f"{sys.executable} -m pip install openai -q")
    import openai

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.urgent_chat_prioritizer_environment import UrgentChatPrioritizerEnvironment
from models import UrgentChatPrioritizerAction, PriorityLevel


def load_work_prompt() -> str:
    """Load WORK.md content to use as system prompt."""
    work_md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WORK.md")
    if os.path.exists(work_md_path):
        with open(work_md_path, 'r') as f:
            return f.read()
    return ""


def get_openai_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)


def analyze_urgency_percentage(client: openai.OpenAI, message_text: str, 
                               sender_type: str, sender_name: str,
                               relationship: float, keywords: List[str],
                               time_since_arrival: float, system_prompt: str) -> Dict[str, Any]:
    """Use LLM with WORK.md as system prompt to analyze message urgency."""
    
    prompt = f"""MESSAGE DETAILS:
- Message: "{message_text}"
- Sender: {sender_name} (type: {sender_type})
- Relationship Strength: {relationship:.2f} (0=unknown, 1=very close)
- Time Since Arrival: {time_since_arrival:.1f} minutes ago
- Urgency Keywords Found: {', '.join(keywords) if keywords else 'None'}

Analyze this message and calculate urgency percentage (0-100%).

Respond in JSON format:
{{"urgency_percentage": 75, "priority_rank": "HIGH", "reason": "short explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=250,
        )
        result_text = response.choices[0].message.content
        
        data = json.loads(result_text)
        return {
            "urgency_percentage": data.get("urgency_percentage", 50),
            "priority_rank": data.get("priority_rank", "NORMAL"),
            "reason": data.get("reason", "No reason provided")
        }
    except Exception as e:
        print(f"Error analyzing: {e}")
        return {"urgency_percentage": 50, "priority_rank": "NORMAL", "reason": "Error - using default"}


def run_baseline(client: openai.OpenAI, env: UrgentChatPrioritizerEnvironment, 
                task_id: str = "all", system_prompt: str = "") -> Dict[str, float]:
    """Run baseline inference with percentage-based urgency."""
    
    print("=" * 60)
    print("Running Baseline Inference with % Urgency")
    print("Using WORK.md as LLM system prompt")
    print("=" * 60)
    
    obs = env.reset()
    print(f"\nEpisode started with {len(obs.pending_chats)} messages")
    print(f"Total users: {obs.metadata.get('num_users', 'N/A')}")
    
    step = 0
    total_reward = 0.0
    message_urgency = {}
    
    while not obs.done and step < 30:
        step += 1
        
        if not obs.pending_chats:
            break
        
        messages = obs.pending_chats
        
        analysis_results = []
        for msg in messages[:10]:
            if msg.message_id in message_urgency:
                analysis_results.append((msg.message_id, message_urgency[msg.message_id]))
            else:
                analysis = analyze_urgency_percentage(
                    client,
                    msg.message_text,
                    msg.sender_type.value,
                    msg.sender_name,
                    msg.relationship_strength,
                    msg.urgency_keywords,
                    msg.time_since_arrival,
                    system_prompt
                )
                message_urgency[msg.message_id] = analysis
                analysis_results.append((msg.message_id, analysis))
        
        analysis_results.sort(key=lambda x: x[1].get("urgency_percentage", 0), reverse=True)
        
        best_msg_id = analysis_results[0][0]
        best_percentage = analysis_results[0][1].get("urgency_percentage", 0)
        
        if best_percentage >= 70:
            priority = PriorityLevel.HIGHEST
        elif best_percentage >= 50:
            priority = PriorityLevel.HIGH
        elif best_percentage >= 30:
            priority = PriorityLevel.NORMAL
        else:
            priority = PriorityLevel.LOW
        
        print(f"\nStep {step}:")
        print(f"  Selected: {best_msg_id[:20]}... ({best_percentage}% urgency)")
        print(f"  Reason: {analysis_results[0][1].get('reason', 'N/A')[:50]}")
        
        print(f"  Top 3:")
        for i, (msg_id, analysis) in enumerate(analysis_results[:3]):
            pct = analysis.get("urgency_percentage", 0)
            print(f"    {i+1}. {pct}% - {analysis.get('reason', '')[:40]}")
        
        action = UrgentChatPrioritizerAction(
            action_type="prioritize",
            chat_id=best_msg_id,
            new_priority=priority
        )
        
        obs = env.step(action)
        total_reward += obs.reward
        
        print(f"  Reward: {obs.reward:.1f}")
    
    print(f"\n{'=' * 60}")
    print(f"Baseline Complete!")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward per Step: {total_reward/max(step,1):.2f}")
    print(f"{'=' * 60}")
    
    return {
        "task_id": task_id,
        "total_steps": step,
        "total_reward": total_reward,
        "avg_reward": total_reward / max(step, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for UrgentChatPrioritizer")
    parser.add_argument("--task", default="all", help="Task ID to run (default: all)")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    args = parser.parse_args()
    
    try:
        client = get_openai_client()
        print("OpenAI client initialized successfully")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Load WORK.md as system prompt
    system_prompt = load_work_prompt()
    print(f"Loaded WORK.md: {len(system_prompt)} characters")
    
    env = UrgentChatPrioritizerEnvironment(difficulty=args.difficulty)
    
    results = run_baseline(client, env, args.task, system_prompt)
    
    print("\nResults Summary:")
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    main()