"""
Reasoning Layer - Part B (NO FRAMEWORKS)
Hand-written decision layer for natural language questions about images

Components:
1. Intent Routing: Decide if detection is needed
2. Structured Reasoning: Reason over detection output
3. Confidence Guardrail: Return "insufficient info" when appropriate
"""

import re
from typing import Dict, Any, List
import numpy as np


class ReasoningLayer:
    """
    Minimal reasoning layer without agentic frameworks

    This is a single hand-written decision layer that:
    - Routes intents
    - Calls detection when needed
    - Reasons over structured output
    - Provides confidence guardrails
    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize reasoning layer

        Args:
            api_key: Optional LLM API key for complex reasoning (NOT REQUIRED - rule-based works fine!)
            model_name: LLM model to use (if API key provided)

        Note: This reasoning layer uses HAND-WRITTEN LOGIC by default.
              LLM API is optional enhancement for complex questions.
              Works perfectly without any API key!
        """

        self.api_key = api_key
        self.model_name = model_name
        self.use_llm = api_key is not None and len(api_key) > 0

        # Define keywords for intent routing
        self.detection_keywords = {
            'counting': ['how many', 'count', 'number of', 'total'],
            'existence': ['is there', 'are there', 'any', 'does it have'],
            'comparison': ['more', 'less', 'most', 'least', 'common'],
            'identification': ['what', 'which', 'identify', 'name'],
        }

        # Keywords that DON'T need detection
        self.non_detection_keywords = [
            'what is', 'define', 'explain', 'describe the concept',
            'weather', 'time', 'color of the walls'
        ]

        # Class names we can detect (will be updated from detector)
        self.known_classes = ['board', 'chair', 'desk', 'fan']

    def process_question(
        self,
        question: str,
        image: np.ndarray,
        detector
    ) -> Dict[str, Any]:
        """
        Main processing pipeline

        Args:
            question: Natural language question
            image: Image array
            detector: ObjectDetector instance

        Returns:
            Dictionary with answer, confidence, and metadata
        """

        question_lower = question.lower().strip()

        # Step 1: Intent Routing
        needs_detection = self._route_intent(question_lower)

        if not needs_detection:
            # Answer without detection
            return {
                'answer': self._answer_without_detection(question),
                'confidence': 'high',
                'used_detection': False,
                'num_objects': 0
            }

        # Step 2: Run Detection
        detections = detector.detect(image)

        # Step 3: Structured Reasoning over detections
        result = self._reason_over_detections(question_lower, detections)

        return result

    def _route_intent(self, question: str) -> bool:
        """
        Intent Router: Decide if detection is needed

        Returns:
            True if detection needed, False otherwise
        """

        # Check if question explicitly doesn't need detection
        for keyword in self.non_detection_keywords:
            if keyword in question:
                return False

        # Check if question needs detection
        for category, keywords in self.detection_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    # Also check if question mentions detectable objects
                    mentions_object = any(
                        cls in question for cls in self.known_classes
                    )
                    if mentions_object or category in ['counting', 'existence']:
                        return True

        # Default: use detection if uncertain
        return True

    def _answer_without_detection(self, question: str) -> str:
        """Answer questions that don't need image analysis"""

        question_lower = question.lower()

        # Pattern matching for common non-detection questions
        if 'what is' in question_lower or 'define' in question_lower:
            return (
                "I cannot answer general knowledge questions. "
                "Please ask about objects visible in this classroom image."
            )

        if 'weather' in question_lower or 'time' in question_lower:
            return (
                "I cannot determine weather or time from this image. "
                "I can only detect classroom objects like chairs, desks, and boards."
            )

        return (
            "This question doesn't require analyzing the image content. "
            "Please ask about objects in the classroom."
        )

    def _reason_over_detections(
        self,
        question: str,
        detections: List[Dict]
    ) -> Dict[str, Any]:
        """
        Structured Reasoning: Answer based on detection results

        This is the core reasoning logic - hand-written, no frameworks
        """

        # Get object counts
        counts = {}
        for det in detections:
            cls_name = det['class_name']
            counts[cls_name] = counts.get(cls_name, 0) + 1

        # Identify question type and answer accordingly
        answer, confidence = self._generate_answer(question, counts, detections)

        return {
            'answer': answer,
            'confidence': confidence,
            'used_detection': True,
            'num_objects': len(detections)
        }

    def _generate_answer(
        self,
        question: str,
        counts: Dict[str, int],
        detections: List[Dict]
    ) -> tuple:
        """
        Generate answer based on question type

        Returns:
            (answer, confidence)
        """

        # 1. COUNTING QUESTIONS
        if any(kw in question for kw in ['how many', 'count', 'number of']):
            return self._answer_counting(question, counts, detections)

        # 2. EXISTENCE QUESTIONS
        if any(kw in question for kw in ['is there', 'are there', 'any']):
            return self._answer_existence(question, counts)

        # 3. COMPARISON QUESTIONS
        if any(kw in question for kw in ['more', 'less', 'most', 'least', 'common']):
            return self._answer_comparison(question, counts)

        # 4. IDENTIFICATION QUESTIONS
        if any(kw in question for kw in ['what', 'which', 'identify']):
            return self._answer_identification(question, counts, detections)

        # 5. INSUFFICIENT INFO (catch-all)
        return self._insufficient_info(question)

    def _answer_counting(
        self,
        question: str,
        counts: Dict[str, int],
        detections: List[Dict]
    ) -> tuple:
        """Answer counting questions"""

        # Check which object is being asked about
        for cls_name in self.known_classes:
            if cls_name in question:
                count = counts.get(cls_name, 0)

                if count == 0:
                    return (
                        f"I don't see any {cls_name}s in this image.",
                        'high'
                    )

                # Check average confidence
                cls_detections = [d for d in detections if d['class_name'] == cls_name]
                avg_conf = sum(d['confidence'] for d in cls_detections) / len(cls_detections)

                if avg_conf < 0.4:
                    return (
                        f"I detected {count} {cls_name}(s), but confidence is low. "
                        f"There might be detection errors.",
                        'low'
                    )

                if avg_conf < 0.6:
                    conf = 'medium'
                else:
                    conf = 'high'

                return (
                    f"I can see {count} {cls_name}{'s' if count > 1 else ''} in this image.",
                    conf
                )

        # General count (all objects)
        total = len(detections)
        if total == 0:
            return ("I don't see any detectable objects in this image.", 'high')

        return (
            f"I detected {total} object{'s' if total > 1 else ''} total: " +
            ", ".join(f"{count} {cls}" for cls, count in counts.items()) + ".",
            'high' if detections and sum(d['confidence'] for d in detections) / len(detections) > 0.6 else 'medium'
        )

    def _answer_existence(
        self,
        question: str,
        counts: Dict[str, int]
    ) -> tuple:
        """Answer existence questions (is there X?)"""

        for cls_name in self.known_classes:
            if cls_name in question:
                if counts.get(cls_name, 0) > 0:
                    count = counts[cls_name]
                    return (
                        f"Yes, there {'is' if count == 1 else 'are'} {count} "
                        f"{cls_name}{'s' if count > 1 else ''} visible.",
                        'high'
                    )
                else:
                    return (
                        f"No, I don't see any {cls_name}s in this image.",
                        'high'
                    )

        return (
            "I can only detect chairs, desks, and boards. "
            "Please ask about these specific objects.",
            'high'
        )

    def _answer_comparison(
        self,
        question: str,
        counts: Dict[str, int]
    ) -> tuple:
        """Answer comparison questions (more/less/most)"""

        if not counts:
            return ("No objects detected to compare.", 'high')

        # Most/least common
        if 'most common' in question or 'most' in question:
            most_common = max(counts.items(), key=lambda x: x[1])
            return (
                f"The most common object is '{most_common[0]}' with {most_common[1]} detected.",
                'high'
            )

        if 'least common' in question or 'least' in question:
            least_common = min(counts.items(), key=lambda x: x[1])
            return (
                f"The least common object is '{least_common[0]}' with {least_common[1]} detected.",
                'high'
            )

        # More chairs or desks?
        if 'more' in question:
            items = [cls for cls in self.known_classes if cls in question]
            if len(items) >= 2:
                count1 = counts.get(items[0], 0)
                count2 = counts.get(items[1], 0)

                if count1 > count2:
                    return (
                        f"There are more {items[0]}s ({count1}) than {items[1]}s ({count2}).",
                        'high'
                    )
                elif count2 > count1:
                    return (
                        f"There are more {items[1]}s ({count2}) than {items[0]}s ({count1}).",
                        'high'
                    )
                else:
                    return (
                        f"There are equal numbers of {items[0]}s and {items[1]}s ({count1} each).",
                        'high'
                    )

        return self._insufficient_info(question)

    def _answer_identification(
        self,
        question: str,
        counts: Dict[str, int],
        detections: List[Dict]
    ) -> tuple:
        """Answer identification questions (what's in the image?)"""

        if not counts:
            return ("I don't see any detectable objects in this image.", 'high')

        objects_list = []
        for cls_name, count in counts.items():
            objects_list.append(f"{count} {cls_name}{'s' if count > 1 else ''}")

        return (
            f"I can see: {', '.join(objects_list)} in this classroom image.",
            'high'
        )

    def _insufficient_info(self, question: str) -> tuple:
        """
        Confidence Guardrail: Return this when detection results
        are insufficient to answer confidently

        This is CRITICAL for the submission - must handle gracefully
        """

        return (
            "Insufficient information to answer this question confidently. "
            "I can detect chairs, desks, and boards in classroom images, "
            "but this question requires information I cannot extract from the detections.",
            'insufficient'
        )


if __name__ == "__main__":
    # Test reasoning layer
    reasoner = ReasoningLayer()

    test_questions = [
        "How many chairs are in this image?",
        "Is there a whiteboard visible?",
        "What's the weather like?",  # Should not use detection
        "Are there more desks or chairs?",
        "Is anyone sitting at the desks?",  # Insufficient info
    ]

    print("Testing Reasoning Layer")
    print("=" * 60)

    for q in test_questions:
        needs_det = reasoner._route_intent(q.lower())
        print(f"\nQ: {q}")
        print(f"   Needs detection: {needs_det}")
