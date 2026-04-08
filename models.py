# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Urgent Chat Prioritizer Environment.

A real-world RL environment that trains AI agents to prioritize chats by urgency.
"""

from enum import Enum
from typing import Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class SenderType(str, Enum):
    FAMILY = "family"
    BOSS = "boss"
    CLOSE_FRIEND = "close_friend"
    ACQUAINTANCE = "acquaintance"
    UNKNOWN = "unknown"
    GROUP = "group"
    BOT = "bot"


class PriorityLevel(str, Enum):
    HIGHEST = "highest"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class ChatMessage(BaseModel):
    message_id: str
    message_text: str
    sender_name: str
    sender_type: SenderType
    relationship_strength: float = Field(ge=0.0, le=1.0)
    time_since_arrival: float = Field(ge=0.0, description="Minutes since message arrived")
    urgency_keywords: list[str] = Field(default_factory=list)
    has_question: bool = False
    is_forwarded: bool = False
    is_promotional: bool = False
    conversation_active: bool = False
    user_replies_quickly: bool = False
    style_similarity: float = Field(ge=0.0, le=1.0, default=0.5)


class UrgentChatPrioritizerAction(Action):
    """Action for the Urgent Chat Prioritizer environment.
    
    The agent chooses which chat to prioritize or how to handle it.
    """
    action_type: str = Field(..., description="Type of action: prioritize, snooze, mark_spam")
    chat_id: str = Field(..., description="ID of the chat message to act on")
    new_priority: Optional[PriorityLevel] = Field(default=None, description="New priority level")


class UrgentChatPrioritizerObservation(Observation):
    """Observation from the Urgent Chat Prioritizer environment.
    
    Contains the current list of pending chats and their details.
    Also includes reward feedback after each action.
    """
    pending_chats: list[ChatMessage] = Field(default_factory=list, description="List of pending chats")
    prioritized_chats: list[ChatMessage] = Field(default_factory=list, description="Chats already prioritized")
    total_messages: int = Field(default=0, description="Total messages in episode")
    processed_messages: int = Field(default=0, description="Messages already processed")
    episode_time: float = Field(default=0.0, description="Time elapsed in episode (minutes)")
    done: bool = Field(default=False, description="Whether episode is complete")
    reward: float = Field(default=0.0, description="Reward from last action")
    total_reward: float = Field(default=0.0, description="Cumulative reward for episode")
    reward_breakdown: dict = Field(default_factory=dict, description="Detailed breakdown of reward components")
    last_action_result: str = Field(default="", description="Description of what happened with last action")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
