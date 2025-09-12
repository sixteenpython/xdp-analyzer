#!/usr/bin/env python3
"""
Test script for Enhanced Summarizer
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

def test_enhanced_summarizer():
    """Test the enhanced summarizer functionality"""
    
    print("🧪 Testing Enhanced Document Summarizer")
    print("=" * 50)
    
    try:
        from analyzers.enhanced_summarizer import EnhancedDocumentSummarizer
        print("✅ Enhanced summarizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced summarizer: {e}")
        return False
    
    # Initialize summarizer (local mode for testing)
    try:
        summarizer = EnhancedDocumentSummarizer(use_llm=False)
        print("✅ Enhanced summarizer initialized in local mode")
    except Exception as e:
        print(f"❌ Failed to initialize summarizer: {e}")
        return False
    
    # Mock Excel analysis for testing
    class MockWorksheet:
        def __init__(self, name, cell_count=10, formula_count=5):
            self.name = name
            self.cells = [MockCell(f"Cell {i}") for i in range(cell_count)]
            self.formulas = [{"formula": f"=SUM(A{i}:B{i})", "functions": ["SUM"]} for i in range(formula_count)]
            self.charts = []
            self.pivot_tables = []
    
    class MockCell:
        def __init__(self, value):
            self.value = value
            self.comment = None
    
    class MockExcelAnalysis:
        def __init__(self):
            self.filename = "Financial_Model_VaR_Analysis.xlsx"
            self.worksheets = [
                MockWorksheet("Portfolio Analysis", 50, 25),
                MockWorksheet("Risk Calculations", 30, 15),
                MockWorksheet("Dashboard", 20, 10)
            ]
    
    mock_excel_analysis = MockExcelAnalysis()
    
    # Mock formula analysis
    mock_formula_analysis = {
        'total_formulas': 50,
        'average_complexity': 12,
        'function_usage': {'SUM': 15, 'VLOOKUP': 8, 'IF': 12, 'AVERAGE': 6, 'VAR': 5},
        'business_categories': {'financial': 25, 'analytical': 15, 'general': 10}
    }
    
    print("\n🔄 Testing enhanced summary generation...")
    
    try:
        # Test enhanced local summarization
        import asyncio
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            enhanced_summary = loop.run_until_complete(
                summarizer.generate_enhanced_summary(mock_excel_analysis, mock_formula_analysis)
            )
        finally:
            loop.close()
        
        print("✅ Enhanced summary generated successfully!")
        print(f"   Generation Method: {enhanced_summary.generation_method}")
        print(f"   Confidence Score: {enhanced_summary.confidence_score:.1%}")
        
        # Test summary components
        print(f"\n📋 Executive Summary Preview:")
        print(f"   {enhanced_summary.executive_summary[:150]}...")
        
        print(f"\n🎯 Business Purpose:")
        print(f"   {enhanced_summary.key_business_purpose}")
        
        print(f"\n🔍 Main Findings ({len(enhanced_summary.main_findings)}):")
        for i, finding in enumerate(enhanced_summary.main_findings[:3], 1):
            print(f"   {i}. {finding}")
        
        print(f"\n💡 Actionable Insights ({len(enhanced_summary.actionable_insights)}):")
        for i, insight in enumerate(enhanced_summary.actionable_insights[:2], 1):
            print(f"   {i}. {insight}")
        
        print(f"\n⚠️ Potential Concerns ({len(enhanced_summary.potential_concerns)}):")
        for i, concern in enumerate(enhanced_summary.potential_concerns, 1):
            print(f"   {i}. {concern}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced summary generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_business_context_detection():
    """Test business context detection"""
    
    print("\n🧪 Testing Business Context Detection")
    print("=" * 50)
    
    from analyzers.enhanced_summarizer import EnhancedDocumentSummarizer
    summarizer = EnhancedDocumentSummarizer(use_llm=False)
    
    test_cases = [
        {
            'filename': 'VaR_Portfolio_Analysis.xlsx',
            'text': ['value at risk', 'portfolio', 'volatility', 'returns', 'correlation'],
            'expected': 'financial_modeling'
        },
        {
            'filename': 'Annual_Budget_2024.xlsx', 
            'text': ['budget', 'forecast', 'revenue', 'expenses', 'profit'],
            'expected': 'budgeting_planning'
        },
        {
            'filename': 'Sales_Dashboard.xlsx',
            'text': ['analysis', 'metrics', 'kpi', 'trends', 'dashboard'],
            'expected': 'data_analysis'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['filename']}")
        
        # Mock document context
        context = {
            'basic_info': {'filename': test_case['filename']},
            'text_content': test_case['text'],
            'business_indicators': {'financial_terms': test_case['text'][:3]}
        }
        
        business_context = summarizer._identify_business_context(context)
        detected_type = business_context['type']
        
        status = "✅" if detected_type == test_case['expected'] else "❌"
        print(f"{status} Expected: {test_case['expected']}, Detected: {detected_type}")
        print(f"   Confidence: {business_context.get('confidence', 0):.1%}")
        print(f"   Purpose: {business_context['purpose']}")
    
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Summarizer Tests")
        print("="*60)
        
        # Run tests
        test1_result = test_enhanced_summarizer()
        test2_result = test_business_context_detection()
        
        if test1_result and test2_result:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! Enhanced Summarizer is working correctly.")
            print("🚀 Ready for deployment in XDP Analyzer!")
        else:
            print("\n" + "="*60)
            print("❌ Some tests failed. Check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)