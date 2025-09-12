#!/usr/bin/env python3
"""
Test script to validate all XDP Analyzer imports
Run this script to verify the installation is working correctly
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    
    print("Testing XDP Analyzer Module Imports...")
    print("=" * 50)
    
    # Test parsers
    try:
        from parsers.simple_excel_parser import SimpleExcelParser
        print("✓ SimpleExcelParser imported successfully")
    except ImportError as e:
        print(f"✗ SimpleExcelParser import failed: {e}")
        return False
    
    try:
        from parsers.excel_parser import AdvancedExcelParser
        print("✓ AdvancedExcelParser imported successfully")
    except ImportError as e:
        print(f"✗ AdvancedExcelParser import failed: {e}")
        print("  Note: Requires openpyxl for full functionality")
    
    # Test analyzers
    try:
        from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer
        print("✓ FreeIntelligentAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ FreeIntelligentAnalyzer import failed: {e}")
        return False
    
    try:
        from analyzers.enhanced_summarizer import EnhancedDocumentSummarizer
        print("✓ EnhancedDocumentSummarizer imported successfully")
    except ImportError as e:
        print(f"✗ EnhancedDocumentSummarizer import failed: {e}")
        return False
    
    try:
        from analyzers.statistical_analyzer import AdvancedStatisticalAnalyzer
        print("✓ AdvancedStatisticalAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ AdvancedStatisticalAnalyzer import failed: {e}")
        return False
    
    try:
        from analyzers.time_series_analyzer import AdvancedTimeSeriesAnalyzer
        print("✓ AdvancedTimeSeriesAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ AdvancedTimeSeriesAnalyzer import failed: {e}")
        return False
    
    # Test main analyzer
    try:
        from main import EnhancedXDPAnalyzer
        print("✓ EnhancedXDPAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ EnhancedXDPAnalyzer import failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All core modules imported successfully!")
    print("XDP Analyzer is ready for use.")
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    
    print("\nTesting Basic Functionality...")
    print("=" * 30)
    
    try:
        # Initialize simple parser
        from parsers.simple_excel_parser import SimpleExcelParser
        parser = SimpleExcelParser()
        print("✓ Simple parser initialized")
        
        # Initialize analyzers
        from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer
        analyzer = FreeIntelligentAnalyzer()
        print("✓ Intelligent analyzer initialized")
        
        from analyzers.enhanced_summarizer import EnhancedDocumentSummarizer
        summarizer = EnhancedDocumentSummarizer()
        print("✓ Enhanced summarizer initialized")
        
        from analyzers.statistical_analyzer import AdvancedStatisticalAnalyzer
        stat_analyzer = AdvancedStatisticalAnalyzer()
        print("✓ Statistical analyzer initialized")
        
        from analyzers.time_series_analyzer import AdvancedTimeSeriesAnalyzer
        ts_analyzer = AdvancedTimeSeriesAnalyzer()
        print("✓ Time series analyzer initialized")
        
        # Initialize main analyzer
        from main import EnhancedXDPAnalyzer
        main_analyzer = EnhancedXDPAnalyzer()
        print("✓ Main XDP analyzer initialized")
        
        print("\nAll analyzers initialized successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("XDP Analyzer - Enhanced Excel Analysis Tool")
    print("Test Suite")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! XDP Analyzer is ready for use.")
            print("\nUsage:")
            print("  python main.py <excel_file> --mode comprehensive")
            print("  python main.py <excel_file> --output-dir results/")
            return 0
        else:
            print("\n❌ Functionality tests failed.")
            return 1
    else:
        print("\n❌ Import tests failed. Please install required dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())