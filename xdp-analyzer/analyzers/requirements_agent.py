"""
Requirements Agent - Conversational Document Analysis
Engages in human-like conversation to gather requirements and provide tailored analysis
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict
import random

class ConversationState(Enum):
    INITIAL = "initial"
    GATHERING_CONTEXT = "gathering_context"
    ASKING_QUESTIONS = "asking_questions"
    CLARIFYING = "clarifying"
    ANALYZING = "analyzing"
    COMPLETED = "completed"

class QuestionType(Enum):
    PURPOSE = "purpose"
    AUDIENCE = "audience"
    SCOPE = "scope"
    CONSTRAINTS = "constraints"
    PRIORITIES = "priorities"
    TIMELINE = "timeline"
    SUCCESS_CRITERIA = "success_criteria"

@dataclass
class ConversationContext:
    """Context gathered during conversation"""
    user_description: str = ""
    document_type: str = ""
    primary_purpose: str = ""
    target_audience: str = ""
    business_domain: str = ""
    analysis_scope: str = ""
    constraints: List[str] = None
    priorities: List[str] = None
    timeline: str = ""
    success_criteria: List[str] = None
    special_requirements: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.priorities is None:
            self.priorities = []
        if self.success_criteria is None:
            self.success_criteria = []
        if self.special_requirements is None:
            self.special_requirements = []

@dataclass
class ConversationMessage:
    """A message in the conversation"""
    sender: str  # 'agent' or 'user'
    message: str
    timestamp: float
    message_type: str = "text"  # 'text', 'question', 'summary', 'analysis'
    metadata: Optional[Dict] = None

@dataclass
class RequirementsAnalysis:
    """Final requirements-based analysis"""
    requirements_summary: str
    tailored_insights: List[str]
    specific_recommendations: List[str]
    risk_assessment: List[str]
    implementation_roadmap: List[str]
    success_metrics: List[str]
    conversation_context: ConversationContext
    confidence_score: float

class RequirementsAgent:
    """
    Conversational agent that gathers requirements and provides tailored document analysis
    """
    
    def __init__(self, intelligent_analyzer=None):
        self.logger = logging.getLogger(__name__)
        self.intelligent_analyzer = intelligent_analyzer
        
        # Conversation state
        self.state = ConversationState.INITIAL
        self.context = ConversationContext()
        self.conversation_history: List[ConversationMessage] = []
        self.questions_asked = set()
        self.current_question_type = None
        
        # Question templates for different domains
        self.question_templates = {
            QuestionType.PURPOSE: [
                "What is the main business problem this document is meant to solve?",
                "What decisions will be made based on this analysis?",
                "Who will be the primary users of this analysis?",
            ],
            QuestionType.AUDIENCE: [
                "Who is your target audience for this analysis? (e.g., executives, analysts, operations team)",
                "What level of technical detail should the analysis include?",
                "Are there specific stakeholders who need particular insights?",
            ],
            QuestionType.SCOPE: [
                "What specific areas should I focus on in my analysis?",
                "Are there particular metrics or KPIs you're most interested in?",
                "Should I look at specific time periods or data ranges?",
            ],
            QuestionType.CONSTRAINTS: [
                "Are there any limitations or constraints I should be aware of?",
                "Are there specific compliance or regulatory requirements?",
                "What data sensitivity considerations should I keep in mind?",
            ],
            QuestionType.PRIORITIES: [
                "What are your top 3 priorities for this analysis?",
                "Is speed more important than comprehensiveness, or vice versa?",
                "Which aspects are most critical for your business decision?",
            ],
            QuestionType.TIMELINE: [
                "When do you need this analysis completed?",
                "Is this for an immediate decision or longer-term planning?",
                "Are there upcoming deadlines I should consider?",
            ],
            QuestionType.SUCCESS_CRITERIA: [
                "How will you measure if this analysis is successful?",
                "What specific outcomes are you hoping to achieve?",
                "What would make this analysis most valuable to you?",
            ]
        }
        
        # Domain-specific follow-up questions
        self.domain_questions = {
            'financial': [
                "Are you looking at profitability, cash flow, or investment analysis?",
                "What financial metrics are most important to your organization?",
                "Are there specific financial periods or scenarios to analyze?",
            ],
            'operational': [
                "Which operational processes are you most concerned about?",
                "Are you looking to optimize efficiency or identify bottlenecks?",
                "What operational metrics drive your business success?",
            ],
            'analytical': [
                "What trends or patterns are you hoping to discover?",
                "Do you need predictive insights or historical analysis?",
                "What data sources should I consider in the analysis?",
            ],
            'strategic': [
                "What strategic decisions is this analysis supporting?",
                "Are you evaluating new opportunities or optimizing existing ones?",
                "What competitive factors should I consider?",
            ]
        }
        
        # Response patterns for natural conversation
        self.response_patterns = {
            'acknowledgment': [
                "I understand.",
                "That makes sense.",
                "Got it.",
                "Thank you for clarifying.",
                "That's helpful context.",
            ],
            'follow_up': [
                "Let me ask you a bit more about that.",
                "Building on what you said,",
                "To better understand your needs,",
                "That's interesting - could you tell me more about",
                "Following up on that point,",
            ],
            'transition': [
                "Now, let's move to another important aspect.",
                "That gives me good insight. Next,",
                "Perfect. I'd also like to understand",
                "Thanks for that context. Another question:",
                "Great. Now I'm curious about",
            ],
            'completion': [
                "Excellent! I have enough information to provide you with a comprehensive analysis.",
                "Perfect. Based on our conversation, I can now deliver a tailored analysis.",
                "Great! I'm ready to analyze your document with these requirements in mind.",
                "Thank you for the detailed information. Let me prepare your customized analysis.",
            ]
        }
    
    def start_conversation(self, user_description: str, document_type: str) -> str:
        """Initialize conversation with user description"""
        
        self.state = ConversationState.GATHERING_CONTEXT
        self.context.user_description = user_description
        self.context.document_type = document_type
        
        # Log initial message
        self._add_message("user", user_description)
        
        # Analyze user description to determine domain and initial questions
        domain = self._detect_business_domain(user_description)
        self.context.business_domain = domain
        
        # Generate personalized opening response
        opening_response = self._generate_opening_response(user_description, document_type, domain)
        self._add_message("agent", opening_response)
        
        self.state = ConversationState.ASKING_QUESTIONS
        
        return opening_response
    
    def process_user_response(self, user_message: str) -> str:
        """Process user response and generate next question or analysis"""
        
        self._add_message("user", user_message)
        
        if self.state == ConversationState.ASKING_QUESTIONS:
            return self._handle_question_response(user_message)
        elif self.state == ConversationState.CLARIFYING:
            return self._handle_clarification(user_message)
        elif self.state == ConversationState.COMPLETED:
            return "Our conversation is complete. Please proceed to get your analysis."
        else:
            return self._generate_next_question()
    
    def _detect_business_domain(self, description: str) -> str:
        """Detect business domain from user description"""
        
        description_lower = description.lower()
        domain_scores = defaultdict(int)
        
        # Financial keywords
        financial_keywords = ['budget', 'revenue', 'profit', 'cost', 'financial', 'money', 'roi', 'investment', 'expense']
        for keyword in financial_keywords:
            if keyword in description_lower:
                domain_scores['financial'] += 2
        
        # Operational keywords
        operational_keywords = ['process', 'efficiency', 'operations', 'workflow', 'productivity', 'performance', 'quality']
        for keyword in operational_keywords:
            if keyword in description_lower:
                domain_scores['operational'] += 2
        
        # Analytical keywords
        analytical_keywords = ['analysis', 'data', 'metrics', 'trends', 'insights', 'report', 'dashboard', 'kpi']
        for keyword in analytical_keywords:
            if keyword in description_lower:
                domain_scores['analytical'] += 2
        
        # Strategic keywords
        strategic_keywords = ['strategy', 'planning', 'goals', 'growth', 'market', 'competitive', 'roadmap']
        for keyword in strategic_keywords:
            if keyword in description_lower:
                domain_scores['strategic'] += 2
        
        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else 'general'
    
    def _generate_opening_response(self, description: str, doc_type: str, domain: str) -> str:
        """Generate personalized opening response"""
        
        responses = [
            f"Thanks for providing that context about your {doc_type} document. I can see this is related to {domain} analysis.",
            f"I appreciate the details about your {doc_type} file. It looks like you're working on {domain} matters.",
            f"Great! I understand you have a {doc_type} document for {domain} analysis.",
        ]
        
        base_response = random.choice(responses)
        
        follow_up = [
            "To provide you with the most relevant analysis, I'd like to ask you a few questions.",
            "Let me ask a few targeted questions so I can tailor my analysis to your specific needs.",
            "I have a few questions that will help me give you the most valuable insights.",
        ]
        
        return f"{base_response} {random.choice(follow_up)}"
    
    def _handle_question_response(self, response: str) -> str:
        """Handle response to a question and generate next question"""
        
        # Extract information from response and update context
        self._update_context_from_response(response)
        
        # Decide if we need more information or can proceed
        if len(self.questions_asked) >= 4 or self._has_sufficient_context():
            # Transition to completion
            self.state = ConversationState.COMPLETED
            completion_message = random.choice(self.response_patterns['completion'])
            self._add_message("agent", completion_message)
            return completion_message
        else:
            # Ask next question
            return self._generate_next_question()
    
    def _generate_next_question(self) -> str:
        """Generate the next logical question based on context"""
        
        # Determine which question type to ask next
        next_question_type = self._select_next_question_type()
        
        if next_question_type:
            question = self._get_question_for_type(next_question_type)
            self.questions_asked.add(next_question_type)
            self.current_question_type = next_question_type
            
            # Add natural transition
            transition = random.choice(self.response_patterns['transition'])
            full_response = f"{transition} {question}"
            
            self._add_message("agent", full_response)
            return full_response
        else:
            # No more questions needed
            return self._complete_conversation()
    
    def _select_next_question_type(self) -> Optional[QuestionType]:
        """Select the most appropriate next question type"""
        
        priority_order = [
            QuestionType.PURPOSE,
            QuestionType.AUDIENCE, 
            QuestionType.SCOPE,
            QuestionType.PRIORITIES,
            QuestionType.CONSTRAINTS,
            QuestionType.SUCCESS_CRITERIA,
            QuestionType.TIMELINE,
        ]
        
        for question_type in priority_order:
            if question_type not in self.questions_asked:
                return question_type
        
        return None
    
    def _get_question_for_type(self, question_type: QuestionType) -> str:
        """Get an appropriate question for the given type"""
        
        if question_type in self.question_templates:
            questions = self.question_templates[question_type]
            
            # Add domain-specific questions if available
            if self.context.business_domain in self.domain_questions:
                domain_specific = self.domain_questions[self.context.business_domain]
                
                # Mix general and domain-specific questions
                if question_type == QuestionType.SCOPE:
                    questions.extend(domain_specific)
            
            return random.choice(questions)
        
        return "Could you tell me more about your requirements?"
    
    def _update_context_from_response(self, response: str):
        """Extract and update context information from user response"""
        
        response_lower = response.lower()
        
        # Update context based on current question type
        if self.current_question_type == QuestionType.PURPOSE:
            self.context.primary_purpose = response[:200]  # Store first 200 chars
        elif self.current_question_type == QuestionType.AUDIENCE:
            self.context.target_audience = response[:100]
        elif self.current_question_type == QuestionType.SCOPE:
            self.context.analysis_scope = response[:200]
        elif self.current_question_type == QuestionType.CONSTRAINTS:
            self.context.constraints.append(response[:100])
        elif self.current_question_type == QuestionType.PRIORITIES:
            # Extract priority items
            priorities = re.findall(r'(\d+\.?\s*[^.]+)', response)
            self.context.priorities.extend([p.strip() for p in priorities])
        elif self.current_question_type == QuestionType.TIMELINE:
            self.context.timeline = response[:100]
        elif self.current_question_type == QuestionType.SUCCESS_CRITERIA:
            self.context.success_criteria.append(response[:100])
    
    def _has_sufficient_context(self) -> bool:
        """Determine if we have enough context for analysis"""
        
        essential_info = [
            bool(self.context.primary_purpose),
            bool(self.context.target_audience or self.context.analysis_scope),
            len(self.questions_asked) >= 3
        ]
        
        return all(essential_info)
    
    def _complete_conversation(self) -> str:
        """Complete the conversation and prepare for analysis"""
        
        self.state = ConversationState.COMPLETED
        completion_message = random.choice(self.response_patterns['completion'])
        
        # Add summary of gathered information
        summary = self._generate_context_summary()
        full_response = f"{completion_message}\n\n{summary}"
        
        self._add_message("agent", full_response)
        return full_response
    
    def _generate_context_summary(self) -> str:
        """Generate a summary of gathered context"""
        
        summary_parts = []
        
        if self.context.primary_purpose:
            summary_parts.append(f"**Purpose**: {self.context.primary_purpose}")
        
        if self.context.target_audience:
            summary_parts.append(f"**Audience**: {self.context.target_audience}")
        
        if self.context.analysis_scope:
            summary_parts.append(f"**Scope**: {self.context.analysis_scope}")
        
        if self.context.priorities:
            priorities_text = "; ".join(self.context.priorities[:3])
            summary_parts.append(f"**Priorities**: {priorities_text}")
        
        if self.context.business_domain:
            summary_parts.append(f"**Domain**: {self.context.business_domain}")
        
        if summary_parts:
            return "**Here's what I've gathered:**\n\n" + "\n".join(summary_parts)
        else:
            return "I have enough information to proceed with the analysis."
    
    def _handle_clarification(self, response: str) -> str:
        """Handle clarification responses"""
        
        self._update_context_from_response(response)
        
        acknowledgment = random.choice(self.response_patterns['acknowledgment'])
        return f"{acknowledgment} {self._generate_next_question()}"
    
    def _add_message(self, sender: str, message: str, message_type: str = "text"):
        """Add message to conversation history"""
        
        msg = ConversationMessage(
            sender=sender,
            message=message,
            timestamp=time.time(),
            message_type=message_type
        )
        self.conversation_history.append(msg)
    
    def generate_requirements_analysis(self, document_analysis) -> RequirementsAnalysis:
        """Generate tailored analysis based on gathered requirements"""
        
        # Create tailored insights based on requirements
        tailored_insights = self._generate_tailored_insights(document_analysis)
        
        # Generate specific recommendations based on context
        specific_recommendations = self._generate_specific_recommendations(document_analysis)
        
        # Assess risks in context of requirements
        risk_assessment = self._generate_contextual_risk_assessment(document_analysis)
        
        # Create implementation roadmap
        implementation_roadmap = self._generate_implementation_roadmap()
        
        # Define success metrics based on conversation
        success_metrics = self._generate_success_metrics()
        
        # Generate requirements summary
        requirements_summary = self._generate_requirements_summary()
        
        # Calculate confidence score
        confidence_score = self._calculate_requirements_confidence()
        
        return RequirementsAnalysis(
            requirements_summary=requirements_summary,
            tailored_insights=tailored_insights,
            specific_recommendations=specific_recommendations,
            risk_assessment=risk_assessment,
            implementation_roadmap=implementation_roadmap,
            success_metrics=success_metrics,
            conversation_context=self.context,
            confidence_score=confidence_score
        )
    
    def _generate_tailored_insights(self, document_analysis) -> List[str]:
        """Generate insights tailored to user requirements"""
        
        insights = []
        
        # Base insights from document analysis
        if hasattr(document_analysis, 'key_insights'):
            base_insights = document_analysis.key_insights
        else:
            base_insights = []
        
        # Filter and enhance based on requirements
        if self.context.primary_purpose:
            purpose_keywords = self.context.primary_purpose.lower().split()
            
            # Add purpose-specific insights
            if any(word in purpose_keywords for word in ['decision', 'choose', 'select']):
                insights.append(f"Decision Support: This document provides key data points for {self.context.primary_purpose}")
            
            if any(word in purpose_keywords for word in ['optimize', 'improve', 'enhance']):
                insights.append(f"Optimization Opportunities: Based on your goal to {self.context.primary_purpose}, focus areas have been identified")
        
        # Add domain-specific insights
        if self.context.business_domain == 'financial':
            insights.append("Financial Impact: Key metrics and financial implications have been analyzed")
        elif self.context.business_domain == 'operational':
            insights.append("Operational Efficiency: Process improvement opportunities identified")
        
        # Add scope-specific insights
        if self.context.analysis_scope:
            insights.append(f"Focused Analysis: Analysis concentrated on {self.context.analysis_scope} as requested")
        
        return insights[:5]
    
    def _generate_specific_recommendations(self, document_analysis) -> List[str]:
        """Generate recommendations specific to user requirements"""
        
        recommendations = []
        
        # Base recommendations
        if hasattr(document_analysis, 'recommendations'):
            base_recs = document_analysis.recommendations[:2]  # Take top 2
            recommendations.extend(base_recs)
        
        # Add requirement-specific recommendations
        if self.context.target_audience:
            if 'executive' in self.context.target_audience.lower():
                recommendations.append("Create executive summary dashboard with key metrics and ROI calculations")
            elif 'analyst' in self.context.target_audience.lower():
                recommendations.append("Develop detailed analytical models with drill-down capabilities")
            elif 'team' in self.context.target_audience.lower():
                recommendations.append("Build collaborative workspace with shared metrics and action items")
        
        # Timeline-based recommendations
        if self.context.timeline:
            if any(word in self.context.timeline.lower() for word in ['urgent', 'asap', 'immediate']):
                recommendations.append("Prioritize quick wins and immediate actionable items")
            elif any(word in self.context.timeline.lower() for word in ['long-term', 'strategic', 'future']):
                recommendations.append("Focus on strategic improvements with sustained impact")
        
        # Priority-based recommendations
        if self.context.priorities:
            top_priority = self.context.priorities[0] if self.context.priorities else ""
            if top_priority:
                recommendations.append(f"Immediate Focus: Address '{top_priority}' as your stated top priority")
        
        return recommendations[:5]
    
    def _generate_contextual_risk_assessment(self, document_analysis) -> List[str]:
        """Generate risk assessment tailored to requirements"""
        
        risks = []
        
        # Base risks from document
        if hasattr(document_analysis, 'risk_indicators'):
            base_risks = document_analysis.risk_indicators[:2]
            risks.extend(base_risks)
        
        # Context-specific risks
        if self.context.constraints:
            for constraint in self.context.constraints:
                if 'time' in constraint.lower():
                    risks.append("Timeline Risk: Time constraints may impact analysis depth")
                elif 'budget' in constraint.lower():
                    risks.append("Resource Risk: Budget limitations may restrict implementation options")
                elif 'compliance' in constraint.lower():
                    risks.append("Compliance Risk: Regulatory requirements need careful consideration")
        
        # Domain-specific risks
        if self.context.business_domain == 'financial':
            risks.append("Financial Risk: Market volatility may affect projections")
        elif self.context.business_domain == 'operational':
            risks.append("Operational Risk: Process changes may temporarily reduce efficiency")
        
        return risks[:5]
    
    def _generate_implementation_roadmap(self) -> List[str]:
        """Generate implementation roadmap based on requirements"""
        
        roadmap = []
        
        # Phase 1: Immediate actions
        roadmap.append("Phase 1 (Immediate): Review and validate key findings with stakeholders")
        
        if self.context.priorities:
            top_priority = self.context.priorities[0] if self.context.priorities else ""
            if top_priority:
                roadmap.append(f"Phase 2 (Short-term): Begin implementation of '{top_priority}'")
        
        # Phase 3: Based on domain
        if self.context.business_domain == 'financial':
            roadmap.append("Phase 3 (Medium-term): Implement financial tracking and monitoring systems")
        elif self.context.business_domain == 'operational':
            roadmap.append("Phase 3 (Medium-term): Roll out process improvements and training")
        else:
            roadmap.append("Phase 3 (Medium-term): Execute main improvement initiatives")
        
        # Final phase
        if self.context.success_criteria:
            roadmap.append("Phase 4 (Long-term): Measure success against defined criteria and iterate")
        else:
            roadmap.append("Phase 4 (Long-term): Monitor results and optimize based on outcomes")
        
        return roadmap
    
    def _generate_success_metrics(self) -> List[str]:
        """Generate success metrics based on conversation"""
        
        metrics = []
        
        # Use explicitly stated success criteria
        if self.context.success_criteria:
            for criterion in self.context.success_criteria:
                metrics.append(f"Success Metric: {criterion}")
        
        # Add domain-appropriate metrics
        if self.context.business_domain == 'financial':
            metrics.append("ROI improvement of at least 15%")
            metrics.append("Cost reduction in targeted areas")
        elif self.context.business_domain == 'operational':
            metrics.append("Process efficiency gains")
            metrics.append("Reduced manual effort and errors")
        elif self.context.business_domain == 'analytical':
            metrics.append("Improved data accuracy and insights")
            metrics.append("Faster decision-making cycles")
        
        # Add audience-specific metrics
        if self.context.target_audience:
            if 'executive' in self.context.target_audience.lower():
                metrics.append("Executive satisfaction with strategic insights")
            elif 'analyst' in self.context.target_audience.lower():
                metrics.append("Analyst productivity improvements")
        
        return metrics[:5]
    
    def _generate_requirements_summary(self) -> str:
        """Generate comprehensive requirements summary"""
        
        summary_parts = [
            f"**Document Type**: {self.context.document_type}",
            f"**Business Domain**: {self.context.business_domain}",
        ]
        
        if self.context.primary_purpose:
            summary_parts.append(f"**Primary Purpose**: {self.context.primary_purpose}")
        
        if self.context.target_audience:
            summary_parts.append(f"**Target Audience**: {self.context.target_audience}")
        
        if self.context.priorities:
            priorities_text = "; ".join(self.context.priorities[:3])
            summary_parts.append(f"**Key Priorities**: {priorities_text}")
        
        if self.context.timeline:
            summary_parts.append(f"**Timeline**: {self.context.timeline}")
        
        return "\n".join(summary_parts)
    
    def _calculate_requirements_confidence(self) -> float:
        """Calculate confidence based on information gathered"""
        
        confidence = 0.6  # Base confidence
        
        if self.context.primary_purpose:
            confidence += 0.15
        
        if self.context.target_audience:
            confidence += 0.1
        
        if len(self.context.priorities) > 0:
            confidence += 0.1
        
        if len(self.questions_asked) >= 4:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def get_conversation_history(self) -> List[ConversationMessage]:
        """Get full conversation history"""
        return self.conversation_history.copy()
    
    def reset_conversation(self):
        """Reset conversation state for new session"""
        self.state = ConversationState.INITIAL
        self.context = ConversationContext()
        self.conversation_history = []
        self.questions_asked = set()
        self.current_question_type = None

# Example usage
if __name__ == "__main__":
    agent = RequirementsAgent()
    
    # Simulate conversation
    user_desc = "I have an Excel financial model and need help understanding if it's suitable for budget planning"
    response = agent.start_conversation(user_desc, "Excel")
    print("Agent:", response)
    
    # Simulate user responses
    test_responses = [
        "I need to present this to senior management for quarterly budget approval",
        "The audience is C-level executives who need high-level insights",
        "My main priority is accuracy and the ability to show different scenarios",
        "I need this completed by end of week for the board meeting"
    ]
    
    for user_response in test_responses:
        print(f"\nUser: {user_response}")
        agent_response = agent.process_user_response(user_response)
        print(f"Agent: {agent_response}")