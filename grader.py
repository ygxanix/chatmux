# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task graders for UrgentChatPrioritizer environment.
Provides programmatic grading for 3 difficulty levels.
"""

from typing import Dict, Any, List
from models import SenderType, PriorityLevel


class TaskGrader:
    """Grader for evaluating agent performance on prioritization tasks."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.task_configs = {
            "easy_urgent_vs_spam": {
                "name": "Easy: Urgent vs Spam",
                "description": "Sort obvious emergencies from obvious junk",
                "difficulty": "easy",
            },
            "medium_mixed_urgency": {
                "name": "Medium: Mixed Urgency",
                "description": "Handle messages with conflicting signals",
                "difficulty": "medium",
            },
            "hard_conflicting_signals": {
                "name": "Hard: Conflicting Signals",
                "description": "Prioritize correctly when many messages arrive with conflicting signals",
                "difficulty": "hard",
            },
        }
    
    def grade(self, prioritized_messages: List, all_messages: List) -> Dict[str, Any]:
        """Grade agent's prioritization performance."""
        
        if self.task_id not in self.task_configs:
            return {"score": 0.0, "error": "Invalid task_id"}
        
        config = self.task_configs[self.task_id]
        difficulty = config["difficulty"]
        
        urgent_messages = [
            m for m in all_messages
            if len(m.urgency_keywords) > 0 or m.sender_type in [SenderType.BOSS, SenderType.FAMILY]
        ]
        
        spam_messages = [
            m for m in all_messages
            if m.sender_type == SenderType.BOT or m.is_promotional
        ]
        
        prioritized_ids = set(m.message_id for m in prioritized_messages[:10])
        
        correct_urgent = sum(
            1 for m in urgent_messages if m.message_id in prioritized_ids
        )
        
        correct_spam_rejection = sum(
            1 for m in spam_messages if m.message_id not in prioritized_ids
        )
        
        urgent_precision = correct_urgent / max(len(urgent_messages), 1)
        spam_precision = correct_spam_rejection / max(len(spam_messages), 1)
        
        if difficulty == "easy":
            score = (urgent_precision * 0.7 + spam_precision * 0.3) * 1.0
        elif difficulty == "medium":
            score = (urgent_precision * 0.6 + spam_precision * 0.4) * 0.85
        else:
            score = (urgent_precision * 0.5 + spam_precision * 0.5) * 0.7
        
        return {
            "score": min(score, 1.0),
            "task_id": self.task_id,
            "task_name": config["name"],
            "difficulty": difficulty,
            "details": {
                "total_urgent": len(urgent_messages),
                "correct_urgent_prioritized": correct_urgent,
                "urgent_precision": urgent_precision,
                "total_spam": len(spam_messages),
                "correct_spam_rejected": correct_spam_rejection,
                "spam_precision": spam_precision,
            },
        }


def get_tasks() -> List[Dict[str, Any]]:
    """Return list of available tasks."""
    grader = TaskGrader("")
    return [
        {"id": task_id, **config}
        for task_id, config in grader.task_configs.items()
    ]


def grade_task(task_id: str, prioritized_messages: List, all_messages: List) -> Dict[str, Any]:
    """Grade a specific task."""
    grader = TaskGrader(task_id)
    return grader.grade(prioritized_messages, all_messages)