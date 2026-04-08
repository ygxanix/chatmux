# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Urgent Chat Prioritizer Environment Implementation.

A real-world RL environment that trains AI agents to prioritize chats by urgency.
Uses fuzzy string matching for keyword detection.
"""

import os
import re
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Using basic string matching.")

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        UrgentChatPrioritizerAction,
        UrgentChatPrioritizerObservation,
        ChatMessage,
        SenderType,
        PriorityLevel,
    )
except ImportError:
    from models import (
        UrgentChatPrioritizerAction,
        UrgentChatPrioritizerObservation,
        ChatMessage,
        SenderType,
        PriorityLevel,
    )


# Load keywords from dataset
KEYWORDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_keywords")

PRIORITY_KEYWORDS: Dict[str, Set[str]] = {
    "CRITICAL": set(),
    "HIGH": set(),
    "MEDIUM": set(),
    "LOW": set(),
    "SPAM": set(),
}


def load_keywords():
    """Load all keywords from the dataset."""
    priority_config_path = os.path.join(KEYWORDS_DIR, "priority_config.json")
    
    if os.path.exists(priority_config_path):
        import json
        with open(priority_config_path, 'r') as f:
            config = json.load(f)
        
        priorities = config.get("priorities", {})
        
        for level, info in priorities.items():
            files = info.get("files", [])
            for fname in files:
                fpath = os.path.join(KEYWORDS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        for line in f:
                            line = line.strip().lower()
                            if line and not line.startswith('#') and len(line) > 2:
                                PRIORITY_KEYWORDS[level].add(line)
    
    print(f"Loaded keywords: CRITICAL={len(PRIORITY_KEYWORDS['CRITICAL'])}, HIGH={len(PRIORITY_KEYWORDS['HIGH'])}, MEDIUM={len(PRIORITY_KEYWORDS['MEDIUM'])}, SPAM={len(PRIORITY_KEYWORDS['SPAM'])}")


# Load keywords on module import
load_keywords()


# Fuzzy matching threshold
FUZZY_THRESHOLD = 85


def fuzzy_match(text: str, keyword: str) -> float:
    """Use rapidfuzz for fuzzy string matching. Returns score 0-100."""
    if not FUZZY_AVAILABLE:
        # Fallback to basic matching
        return 100.0 if keyword in text.lower() else 0.0
    
    # Use partial ratio for substring matching
    return fuzz.partial_ratio(text.lower(), keyword)


def detect_keywords(text: str) -> tuple[List[str], str]:
    """Detect urgency keywords using fuzzy matching. Returns (keywords_found, highest_priority)."""
    text_lower = text.lower()
    found = []
    priority_level = None
    best_score = 0
    
    # Priority order: CRITICAL -> HIGH -> MEDIUM -> SPAM
    priority_order = ["CRITICAL", "HIGH", "MEDIUM", "SPAM"]
    
    for priority in priority_order:
        if priority_level is not None:
            break
            
        for keyword in PRIORITY_KEYWORDS[priority]:
            # First try exact match (fast)
            if keyword in text_lower:
                found.append(keyword)
                priority_level = priority
                break
            
            # Then try fuzzy match (if available and no exact match found)
            if FUZZY_AVAILABLE and priority_level is None:
                score = fuzzy_match(text_lower, keyword)
                if score >= FUZZY_THRESHOLD:
                    found.append(keyword)
                    best_score = score
                    priority_level = priority
                    break
        
        # If fuzzy match found, add to found list
        if priority_level and not any(kw in text_lower for kw in found):
            pass  # Already added
    
    return found, priority_level or "NORMAL"


USER_DATA = {
    "family": ["Mom", "Dad", "Sister", "Brother", "Wife", "Husband"],
    "boss": ["Manager", "Team Lead", "Director", "CEO", "Boss"],
    "close_friend": ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Riley"],
    "acquaintance": ["Colleague", "Neighbor", "Cousin", "Work Friend"],
    "unknown": ["Unknown Number", "Random Caller", "New Contact"],
    "group": ["Family Group", "Work Team", "Project Group", "Friends Chat"],
    "bot": ["Reminder Bot", "Alert Bot", "Notification Bot"]
}

MESSAGE_TEMPLATES = {
    "urgent": [
        "URGENT: Need help ASAP!",
        "Emergency! Please respond now!",
        "CRITICAL: Deadline is today!",
        "Help needed immediately!",
        "Important! This needs your attention NOW!",
        "ASAP - Please handle this right away!",
    ],
    "normal": [
        "Hey, can you check this when you get a chance?",
        "Quick question for you",
        "Can you help me with something?",
        "What's your thought on this?",
    ],
    "casual": [
        "Hey! What are you up to?",
        "Did you see the game last night?",
        "Let's catch up soon!",
        "Just checking in",
    ],
    "spam": [
        "Congratulations! You've WON a prize!",
        "Limited time OFFER! 50% off!",
        "ACT NOW! Special promotion ends soon!",
    ]
}


class ChatUser:
    def __init__(self, user_id: str, user_name: str, sender_type: SenderType, relationship_strength: float):
        self.user_id = user_id
        self.user_name = user_name
        self.sender_type = sender_type
        self.relationship_strength = relationship_strength
        self.messages: List[ChatMessage] = []
    
    def add_message(self, message: ChatMessage):
        self.messages.append(message)


class UrgentChatPrioritizerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._difficulty = difficulty
        self._rng = random.Random(seed)
        
        self._users: Dict[str, ChatUser] = {}
        self._all_messages: List[ChatMessage] = []
        self._pending_messages: List[ChatMessage] = []
        self._prioritized_messages: List[ChatMessage] = []
        self._processed_ids: set = set()
        
        self._episode_time = 0.0
        self._total_reward = 0.0

    def reset(self) -> UrgentChatPrioritizerObservation:
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        
        num_users = self._rng.randint(10, 15)
        self._users = self._generate_users(num_users)
        
        self._all_messages = []
        for user in self._users.values():
            self._all_messages.extend(user.messages)
        
        self._rng.shuffle(self._all_messages)
        
        self._pending_messages = self._all_messages.copy()
        self._prioritized_messages = []
        self._processed_ids = set()
        self._episode_time = 0.0
        self._total_reward = 0.0
        
        return UrgentChatPrioritizerObservation(
            pending_chats=self._pending_messages,
            prioritized_chats=self._prioritized_messages,
            total_messages=len(self._all_messages),
            processed_messages=0,
            episode_time=0.0,
            done=False,
            reward=0.0,
            total_reward=0.0,
            reward_breakdown={},
            last_action_result="Episode started - focus on urgent messages first!",
            metadata={"episode_id": self._state.episode_id, "num_users": num_users}
        )

    def _generate_users(self, count: int) -> Dict[str, ChatUser]:
        users = {}
        sender_types = [
            (SenderType.FAMILY, 0.15),
            (SenderType.BOSS, 0.10),
            (SenderType.CLOSE_FRIEND, 0.25),
            (SenderType.ACQUAINTANCE, 0.20),
            (SenderType.UNKNOWN, 0.15),
            (SenderType.GROUP, 0.10),
            (SenderType.BOT, 0.05),
        ]
        
        for i in range(count):
            sender_type = self._rng.choices(
                [st for st, _ in sender_types],
                weights=[w for _, w in sender_types]
            )[0]
            
            user_names = USER_DATA.get(sender_type.value, ["Unknown"])
            user_name = self._rng.choice(user_names)
            
            rel_range = {
                SenderType.FAMILY: (0.8, 1.0),
                SenderType.BOSS: (0.7, 1.0),
                SenderType.CLOSE_FRIEND: (0.6, 0.9),
                SenderType.ACQUAINTANCE: (0.3, 0.6),
                SenderType.UNKNOWN: (0.0, 0.2),
                SenderType.GROUP: (0.2, 0.5),
                SenderType.BOT: (0.0, 0.1),
            }.get(sender_type, (0.5, 0.8))
            
            rel_strength = self._rng.uniform(*rel_range)
            
            user = ChatUser(
                user_id=f"user_{i}",
                user_name=user_name,
                sender_type=sender_type,
                relationship_strength=rel_strength
            )
            
            num_messages = self._rng.randint(3, 12)
            user.messages = self._generate_user_messages(user, num_messages, i)
            
            users[user.user_id] = user
        
        return users

    def _generate_user_messages(self, user: ChatUser, count: int, user_idx: int) -> List[ChatMessage]:
        messages = []
        config = self._get_difficulty_config()
        
        urgent_ratio = config["urgent_ratio"]
        spam_ratio = config["spam_ratio"]
        normal_ratio = 1.0 - urgent_ratio - spam_ratio
        
        urgent_count = int(count * urgent_ratio)
        spam_count = int(count * spam_ratio)
        normal_count = count - urgent_count - spam_count
        
        msg_types = ["urgent"] * urgent_count + ["spam"] * spam_count + ["normal"] * normal_count
        self._rng.shuffle(msg_types)
        
        base_time = datetime.now() - timedelta(hours=8)
        
        for j, msg_type in enumerate(msg_types):
            urgency_keywords = []
            message_text = ""
            
            if msg_type == "urgent":
                message_text = self._rng.choice(MESSAGE_TEMPLATES["urgent"])
            elif msg_type == "spam":
                message_text = self._rng.choice(MESSAGE_TEMPLATES["spam"])
            else:
                message_text = self._rng.choice(MESSAGE_TEMPLATES["normal"] + MESSAGE_TEMPLATES["casual"])
            
            keywords_found, priority_level = detect_keywords(message_text)
            urgency_keywords = keywords_found
            
            time_offset = self._rng.uniform(0, 480)
            
            # Only mark as promotional if it's actually a spam type message
            # Not based on content (which might have words like "offer" in urgent messages)
            is_promotional = (msg_type == "spam")
            
            message = ChatMessage(
                message_id=f"msg_{user_idx}_{j}",
                message_text=message_text,
                sender_name=user.user_name,
                sender_type=user.sender_type,
                relationship_strength=user.relationship_strength,
                time_since_arrival=time_offset,
                urgency_keywords=urgency_keywords,
                has_question="?" in message_text,
                is_forwarded=self._rng.random() < 0.1,
                is_promotional=is_promotional,
                conversation_active=self._rng.random() < 0.3,
                user_replies_quickly=self._rng.random() < user.relationship_strength,
                style_similarity=self._rng.uniform(0.3, 0.9)
            )
            messages.append(message)
        
        messages.sort(key=lambda m: m.time_since_arrival, reverse=True)
        return messages

    def _get_difficulty_config(self) -> dict:
        configs = {
            "easy": {"urgent_ratio": 0.25, "spam_ratio": 0.25},
            "medium": {"urgent_ratio": 0.18, "spam_ratio": 0.22},
            "hard": {"urgent_ratio": 0.12, "spam_ratio": 0.18},
        }
        return configs.get(self._difficulty, configs["medium"])

    def step(self, action: UrgentChatPrioritizerAction) -> UrgentChatPrioritizerObservation:
        self._state.step_count += 1
        self._episode_time += random.uniform(2, 8)
        
        reward, breakdown, action_result = self._calculate_reward(action)
        self._total_reward += reward
        
        chat_id = action.chat_id
        
        if chat_id in [m.message_id for m in self._pending_messages]:
            msg = next(m for m in self._pending_messages if m.message_id == chat_id)
            self._processed_ids.add(chat_id)
            self._pending_messages = [m for m in self._pending_messages if m.message_id != chat_id]
            
            if action.action_type == "prioritize":
                self._prioritized_messages.append(msg)
        
        done = len(self._pending_messages) == 0 or self._state.step_count >= len(self._all_messages)
        
        bonus = 0.0
        if done and len(self._prioritized_messages) > 0:
            bonus = self._calculate_completion_bonus()
            self._total_reward += bonus
            if bonus > 0:
                breakdown["completion_bonus"] = bonus
                action_result += f" | Bonus: +{bonus:.1f}"
        
        return UrgentChatPrioritizerObservation(
            pending_chats=self._pending_messages,
            prioritized_chats=self._prioritized_messages,
            total_messages=len(self._all_messages),
            processed_messages=len(self._processed_ids),
            episode_time=self._episode_time,
            done=done,
            reward=reward,
            total_reward=self._total_reward,
            reward_breakdown=breakdown,
            last_action_result=action_result,
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id
            }
        )

    def _calculate_reward(self, action: UrgentChatPrioritizerAction) -> tuple[float, dict, str]:
        reward = -1.0
        breakdown = {"time_penalty": -1.0}
        action_result = "Step processed"
        
        chat_id = action.chat_id
        available_ids = [m.message_id for m in self._pending_messages]
        
        if chat_id not in available_ids:
            return reward, breakdown, "Invalid chat_id"
        
        msg = next((m for m in self._pending_messages if m.message_id == chat_id), None)
        if not msg:
            return reward, breakdown, "Message not found"
        
        # Check if sender is spam/bot FIRST (highest priority check)
        is_spam = msg.sender_type == SenderType.BOT or msg.is_promotional
        
        # Check if message has urgent keywords
        has_urgent_keywords = len(msg.urgency_keywords) > 0
        
        # Check if sender is important
        is_important_sender = msg.sender_type in [SenderType.BOSS, SenderType.FAMILY]
        
        if action.action_type == "prioritize":
            # CASE 1: Spam/Bot sender - always penalize (regardless of keywords!)
            if is_spam:
                reward -= 20.0
                breakdown["spam_error"] = -20.0
                action_result = f"Wrong: prioritized spam from [{msg.sender_type.value}]: {msg.message_text[:25]}... (-20)"
            
            # CASE 2: Critical keywords + highest priority (BEST case)
            elif has_urgent_keywords and action.new_priority == PriorityLevel.HIGHEST:
                # Even if has keywords, if sender is suspicious (low relationship), give less
                if msg.relationship_strength < 0.3 and not is_important_sender:
                    reward += 10.0  # Reduced reward - suspicious urgent message
                    breakdown["suspicious_urgent"] = 10.0
                    action_result = f"OK: urgent but relationship weak: {msg.message_text[:25]}... (+10)"
                else:
                    reward += 25.0
                    breakdown["correct_urgent"] = 25.0
                    action_result = f"Good: urgent message: {msg.message_text[:25]}... (+25)"
            
            # CASE 3: Urgent keywords but not highest priority
            elif has_urgent_keywords:
                reward += 15.0
                breakdown["urgent_normal"] = 15.0
                action_result = f"OK: urgent message (not top priority): {msg.message_text[:25]}... (+15)"
            
            # CASE 4: Important sender (boss/family) - no keywords needed
            elif is_important_sender:
                reward += 15.0
                breakdown["important_sender"] = 15.0
                action_result = f"OK: important sender: {msg.message_text[:25]}... (+15)"
            
            # CASE 5: Good relationship strength
            elif msg.relationship_strength > 0.6:
                reward += 10.0
                breakdown["relationship_priority"] = 10.0
                action_result = f"OK: good contact: {msg.message_text[:25]}... (+10)"
            
            # CASE 6: Normal message from normal sender
            else:
                reward += 5.0
                breakdown["neutral_action"] = 5.0
                action_result = f"OK: normal message: {msg.message_text[:25]}... (+5)"
        
        elif action.action_type == "mark_spam":
            if is_spam:
                reward += 10.0
                breakdown["spam_correct"] = 10.0
                action_result = f"OK: flagged spam correctly: {msg.message_text[:25]}... (+10)"
            else:
                # Marked important message as spam - BIG penalty
                if is_important_sender or has_urgent_keywords:
                    reward -= 25.0  # More severe for important messages
                    breakdown["false_spam_important"] = -25.0
                    action_result = f"Wrong: marked important message as spam: {msg.message_text[:25]}... (-25)"
                else:
                    reward -= 15.0
                    breakdown["false_spam"] = -15.0
                    action_result = f"Wrong: marked non-spam as spam: {msg.message_text[:25]}... (-15)"
        
        return reward, breakdown, action_result

    def _calculate_completion_bonus(self) -> float:
        if not self._prioritized_messages:
            return 0.0
        
        urgent_count = sum(
            1 for m in self._all_messages 
            if len(m.urgency_keywords) > 0 or m.sender_type in [SenderType.BOSS, SenderType.FAMILY]
        )
        
        if urgent_count == 0:
            return 0.0
        
        prioritized_ids = set(m.message_id for m in self._prioritized_messages[:5])
        correct_urgent = sum(
            1 for m in self._all_messages 
            if m.message_id in prioritized_ids and (
                len(m.urgency_keywords) > 0 or m.sender_type in [SenderType.BOSS, SenderType.FAMILY]
            )
        )
        
        return (correct_urgent / urgent_count) * 30.0

    @property
    def state(self) -> State:
        return self._state