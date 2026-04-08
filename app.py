# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ChatMux - RL Environment for Chat Message Prioritization

MAIN FILE: app.py
This is the main entry point for the OpenEnv server.
Run with: python app.py
Or: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from models import UrgentChatPrioritizerAction, UrgentChatPrioritizerObservation
from server.urgent_chat_prioritizer_environment import UrgentChatPrioritizerEnvironment


app = create_app(
    UrgentChatPrioritizerEnvironment,
    UrgentChatPrioritizerAction,
    UrgentChatPrioritizerObservation,
    env_name="chatmux",
    max_concurrent_envs=1,
)


def load_work_prompt() -> str:
    """Load WORK.md to use as system prompt for LLM."""
    work_md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WORK.md")
    if os.path.exists(work_md_path):
        with open(work_md_path, 'r') as f:
            return f.read()
    return ""


WORK_PROMPT = load_work_prompt()


@app.get("/tasks")
async def get_tasks():
    """Get list of available tasks with graders."""
    from grader import get_tasks
    return {"tasks": get_tasks()}


@app.post("/grader")
async def grade_task(request: dict):
    """Grade agent's performance on a task."""
    from grader import grade_task
    
    task_id = request.get("task_id")
    prioritized_messages = request.get("prioritized_messages", [])
    all_messages = request.get("all_messages", [])
    
    if not task_id:
        return {"error": "task_id is required"}
    
    result = grade_task(task_id, prioritized_messages, all_messages)
    return result


@app.post("/baseline")
async def run_baseline(request: dict = None):
    """Run baseline inference using LLM (OpenAI API). Requires OPENAI_API_KEY."""
    if request is None:
        request = {}
    
    import json
    
    difficulty = request.get("difficulty", "medium")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable is required"}
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to create OpenAI client: {str(e)}"}
    
    env = UrgentChatPrioritizerEnvironment(difficulty=difficulty)
    obs = env.reset()
    
    total_reward = 0.0
    step = 0
    message_urgency = {}
    
    while not obs.done and step < 20:
        step += 1
        if not obs.pending_chats:
            break
        
        analysis_results = []
        for msg in obs.pending_chats[:10]:
            if msg.message_id in message_urgency:
                analysis_results.append((msg.message_id, message_urgency[msg.message_id]))
            else:
                prompt = f"""MESSAGE DETAILS:
- Message: "{msg.message_text}"
- Sender: {msg.sender_name} (type: {msg.sender_type.value})
- Relationship Strength: {msg.relationship_strength:.2f}
- Time Since Arrival: {msg.time_since_arrival:.1f} minutes
- Keywords: {', '.join(msg.urgency_keywords) if msg.urgency_keywords else 'None'}

Calculate urgency percentage (0-100%).
Respond in JSON: {{"urgency_percentage": 75, "reason": "short"}}"""
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": WORK_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.0,
                    )
                    data = json.loads(response.choices[0].message.content)
                    analysis = {
                        "urgency_percentage": data.get("urgency_percentage", 50),
                        "reason": data.get("reason", "")
                    }
                except:
                    analysis = {"urgency_percentage": 50, "reason": "error"}
                
                message_urgency[msg.message_id] = analysis
                analysis_results.append((msg.message_id, analysis))
        
        analysis_results.sort(key=lambda x: x[1].get("urgency_percentage", 0), reverse=True)
        
        best_id = analysis_results[0][0]
        pct = analysis_results[0][1].get("urgency_percentage", 0)
        
        priority = "highest" if pct >= 70 else "high" if pct >= 50 else "normal" if pct >= 30 else "low"
        
        action = UrgentChatPrioritizerAction(
            action_type="prioritize",
            chat_id=best_id,
            new_priority=priority
        )
        obs = env.step(action)
        total_reward += obs.reward
    
    return {
        "task_id": "all",
        "total_steps": step,
        "total_reward": total_reward,
        "avg_reward": total_reward / max(step, 1),
    }


@app.get("/action-schema")
async def get_action_schema():
    """Get action schema."""
    from models import UrgentChatPrioritizerAction
    return {
        "action_schema": UrgentChatPrioritizerAction.model_json_schema()
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


@app.get("/")
async def root():
    """Serve landing page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatMux - RL Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 800px; text-align: center; }
        h1 {
            font-size: 3.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle { font-size: 1.3rem; color: #888; margin-bottom: 40px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.5rem; }
        .card p { color: #aaa; line-height: 1.6; }
        .endpoints {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .endpoint {
            background: rgba(0,212,255,0.1);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(0,212,255,0.2);
        }
        .endpoint code { color: #00d4ff; font-weight: bold; }
        .endpoint span { display: block; color: #888; font-size: 0.85rem; margin-top: 5px; }
        .badge {
            display: inline-block;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin: 10px;
        }
        .footer { margin-top: 40px; color: #555; font-size: 0.9rem; }
        .footer a { color: #00d4ff; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ChatMux</h1>
        <p class="subtitle">RL Environment for Chat Message Prioritization</p>
        <div class="badge">Meta PyTorch OpenEnv Hackathon 2026</div>
        <div class="badge">Reinforcement Learning</div>
        <div class="card">
            <h2>What is ChatMux?</h2>
            <p>ChatMux is an OpenEnv-compatible RL environment that trains AI agents to automatically prioritize chat messages by urgency.</p>
        </div>
        <div class="card">
            <h2>API Endpoints</h2>
            <div class="endpoints">
                <div class="endpoint"><code>POST /reset</code><span>Start new episode</span></div>
                <div class="endpoint"><code>POST /step</code><span>Execute action</span></div>
                <div class="endpoint"><code>GET /tasks</code><span>List tasks</span></div>
                <div class="endpoint"><code>POST /grader</code><span>Grade performance</span></div>
                <div class="endpoint"><code>POST /baseline</code><span>Run LLM baseline</span></div>
                <div class="endpoint"><code>GET /action-schema</code><span>Get action schema</span></div>
            </div>
        </div>
        <div class="card">
            <h2>Features</h2>
            <p>• 10-15 chat users<br>• 11,578+ urgency keywords<br>• Fuzzy matching<br>• OpenAI API<br>• 3 difficulty levels</p>
        </div>
        <p class="footer">Built for Meta PyTorch OpenEnv Hackathon 2026</p>
    </div>
</body>
</html>
    """


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
