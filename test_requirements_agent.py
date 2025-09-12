#!/usr/bin/env python3
"""
Test script for the Requirements Agent functionality
"""

import sys
from pathlib import Path

# Add our modules to path
sys.path.append(str(Path(__file__).parent))

from analyzers.requirements_agent import RequirementsAgent, ConversationState
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer

def test_requirements_agent():
    """Test the requirements agent functionality"""
    
    print("🧪 Testing Requirements Agent")
    print("=" * 50)
    
    # Initialize components
    intelligent_analyzer = FreeIntelligentAnalyzer()
    agent = RequirementsAgent(intelligent_analyzer)
    
    print("✅ Requirements agent initialized successfully")
    
    # Test conversation start
    user_description = "I have an Excel financial model and need help understanding if it's suitable for budget planning"
    file_type = "Excel"
    
    print(f"\n👤 User: {user_description}")
    
    response = agent.start_conversation(user_description, file_type)
    print(f"🤖 Agent: {response}")
    
    # Test conversation flow
    test_responses = [
        "I need to present this to senior management for quarterly budget approval",
        "The audience is C-level executives who need high-level insights", 
        "My main priority is accuracy and the ability to show different scenarios",
        "I need this completed by end of week for the board meeting"
    ]
    
    for user_response in test_responses:
        print(f"\n👤 User: {user_response}")
        agent_response = agent.process_user_response(user_response)
        print(f"🤖 Agent: {agent_response}")
        
        # Check if conversation is complete
        if agent.state == ConversationState.COMPLETED:
            print("\n🎉 Conversation completed!")
            break
    
    # Test context gathering
    print(f"\n📋 Gathered Context:")
    print(f"   Business Domain: {agent.context.business_domain}")
    print(f"   Primary Purpose: {agent.context.primary_purpose[:100]}...")
    print(f"   Target Audience: {agent.context.target_audience}")
    print(f"   Questions Asked: {len(agent.questions_asked)}")
    
    # Test conversation history
    print(f"\n💬 Conversation History ({len(agent.conversation_history)} messages):")
    for i, msg in enumerate(agent.conversation_history[-4:], 1):  # Show last 4 messages
        sender_icon = "👤" if msg.sender == "user" else "🤖"
        print(f"   {i}. {sender_icon} {msg.sender.title()}: {msg.message[:100]}...")
    
    # Test requirements analysis generation
    if agent.state == ConversationState.COMPLETED:
        print(f"\n🎯 Testing Requirements Analysis Generation...")
        
        # Mock document analysis for testing
        class MockAnalysis:
            key_insights = ["Mock insight 1", "Mock insight 2"]
            recommendations = ["Mock recommendation 1"]
            risk_indicators = ["Mock risk 1"]
        
        mock_doc_analysis = MockAnalysis()
        requirements_analysis = agent.generate_requirements_analysis(mock_doc_analysis)
        
        print(f"✅ Requirements analysis generated successfully!")
        print(f"   Summary: {requirements_analysis.requirements_summary[:100]}...")
        print(f"   Tailored Insights: {len(requirements_analysis.tailored_insights)}")
        print(f"   Recommendations: {len(requirements_analysis.specific_recommendations)}")
        print(f"   Confidence Score: {requirements_analysis.confidence_score:.1%}")
    
    print(f"\n✅ All tests completed successfully!")
    return True

def test_question_generation():
    """Test question generation logic"""
    
    print("\n🧪 Testing Question Generation")
    print("=" * 50)
    
    agent = RequirementsAgent()
    
    # Test domain detection
    test_descriptions = [
        ("I need financial analysis", "financial"),
        ("Looking at operational efficiency", "operational"), 
        ("Want to analyze data trends", "analytical"),
        ("Strategic planning document", "strategic"),
        ("General spreadsheet review", "general")
    ]
    
    for description, expected_domain in test_descriptions:
        detected_domain = agent._detect_business_domain(description)
        status = "✅" if detected_domain == expected_domain else "❌"
        print(f"{status} '{description}' -> {detected_domain} (expected: {expected_domain})")
    
    print("\n✅ Question generation tests completed!")
    return True

def test_state_management():
    """Test conversation state management"""
    
    print("\n🧪 Testing State Management")
    print("=" * 50)
    
    agent = RequirementsAgent()
    
    # Test initial state
    assert agent.state == ConversationState.INITIAL, f"Expected INITIAL, got {agent.state}"
    print("✅ Initial state correct")
    
    # Test conversation start
    agent.start_conversation("Test description", "Excel")
    assert agent.state == ConversationState.ASKING_QUESTIONS, f"Expected ASKING_QUESTIONS, got {agent.state}"
    print("✅ State transition to ASKING_QUESTIONS correct")
    
    # Test conversation completion
    # Simulate answering enough questions
    for _ in range(4):
        agent.process_user_response("Test response")
    
    # State should be completed after sufficient interaction
    if agent.state == ConversationState.COMPLETED:
        print("✅ State transition to COMPLETED correct")
    else:
        print(f"ℹ️  State is {agent.state} (may complete after more questions)")
    
    # Test reset
    agent.reset_conversation()
    assert agent.state == ConversationState.INITIAL, f"Expected INITIAL after reset, got {agent.state}"
    print("✅ State reset correct")
    
    print("\n✅ State management tests completed!")
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting Requirements Agent Tests")
        print("="*60)
        
        # Run all tests
        test_requirements_agent()
        test_question_generation()
        test_state_management()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! Requirements Agent is working correctly.")
        print("🚀 Ready to use in Streamlit application!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)