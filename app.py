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
Or: uvicorn app:app --host 0.0.0.0 --port 7860
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from fastapi.responses import HTMLResponse

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


@app.get("/")
async def root():
    """Serve landing page."""
    html = """<!DOCTYPE html>
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
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        h2 { color: #00d4ff; margin: 25px 0 15px; font-size: 1.4rem; }
        h3 { color: #aaa; margin: 15px 0 10px; font-size: 1.1rem; }
        .subtitle { font-size: 1.2rem; color: #888; margin-bottom: 25px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card p, .card li { color: #ccc; line-height: 1.6; }
        code {
            background: rgba(0,212,255,0.15);
            padding: 2px 8px;
            border-radius: 4px;
            color: #00d4ff;
            font-size: 0.9em;
        }
        pre {
            background: #0d1117;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }
        pre code { background: none; padding: 0; }
        .badge {
            display: inline-block;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            padding: 6px 15px;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 5px;
        }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }
        .endpoint { background: rgba(0,212,255,0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(0,212,255,0.2); }
        .endpoint code { display: block; font-weight: bold; margin-bottom: 4px; }
        .endpoint span { color: #888; font-size: 0.8rem; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        th { color: #00d4ff; }
        .footer { margin-top: 30px; color: #555; font-size: 0.85rem; text-align: center; }
        a { color: #00d4ff; text-decoration: none; }
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
            <p>OpenEnv-compatible RL environment that trains AI agents to automatically prioritize chat messages by urgency. Simulates a busy inbox with 10-15 users sending 3-12 messages each.</p>
        </div>

        <div class="card">
            <h2>Quick Start</h2>
            <h3>Run Locally</h3>
            <pre><code>pip install openenv-core openai rapidfuzz pydantic fastapi uvicorn
python app.py --port 7860</code></pre>
            <h3>With Docker</h3>
            <pre><code>docker build -t chatmux .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key chatmux</code></pre>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <div class="grid">
                <div class="endpoint"><code>POST /reset</code><span>Start new episode</span></div>
                <div class="endpoint"><code>POST /step</code><span>Execute action</span></div>
                <div class="endpoint"><code>GET /tasks</code><span>List tasks</span></div>
                <div class="endpoint"><code>POST /grader</code><span>Grade performance</span></div>
                <div class="endpoint"><code>POST /baseline</code><span>Run LLM baseline</span></div>
                <div class="endpoint"><code>GET /action-schema</code><span>Get schema</span></div>
            </div>
        </div>

        <div class="card">
            <h2>LLM API Setup</h2>
            <p>Set <code>OPENAI_API_KEY</code> environment variable to use LLM features:</p>
            <pre><code>export OPENAI_API_KEY=sk-...</code></pre>
            <h3>Environment Variables</h3>
            <table>
                <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
                <tr><td>OPENAI_API_KEY</td><td>-</td><td>Required for LLM calls</td></tr>
                <tr><td>API_BASE_URL</td><td>https://api.openai.com/v1</td><td>OpenAI-compatible API</td></tr>
                <tr><td>MODEL_NAME</td><td>gpt-3.5-turbo</td><td>Model to use</td></tr>
                <tr><td>DIFFICULTY</td><td>medium</td><td>easy/medium/hard</td></tr>
                <tr><td>MAX_STEPS</td><td>20</td><td>Max steps per episode</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Baseline Inference</h2>
            <p>Use <code>baseline.py</code> or <code>inference.py</code> for LLM-powered inference:</p>
            <pre><code># Using baseline.py
export OPENAI_API_KEY=sk-...
python baseline.py --difficulty medium

# Using inference.py (presubmission format)
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-3.5-turbo
python inference.py</code></pre>
        </div>

        <div class="card">
            <h2>Docker Deployment</h2>
            <pre><code># Build
docker build -t chatmux .

# Run without LLM
docker run -p 7860:7860 chatmux

# Run with OpenAI API
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key chatmux</code></pre>
        </div>

        <div class="card">
            <h2>Features</h2>
            <p>10-15 chat users | 11,578+ urgency keywords | Fuzzy matching | OpenAI API | 3 difficulty levels</p>
        </div>

        <p class="footer">Built for Meta PyTorch OpenEnv Hackathon 2026 | <a href="https://huggingface.co/spaces/Ygxanix/chatmux">HuggingFace Space</a></p>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)


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


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
