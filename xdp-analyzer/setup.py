#!/usr/bin/env python3
"""
XDP Analyzer Setup Script - 100% FREE VERSION
Quick setup for the completely free document analysis platform
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔄 Installing free Python packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📥 Downloading free NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        datasets = ['punkt', 'stopwords', 'vader_lexicon', 'averaged_perceptron_tagger']
        
        for dataset in datasets:
            print(f"  Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except ImportError:
        print("⚠️ NLTK not installed, skipping data download")
        return False
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        from parsers.excel_parser import AdvancedExcelParser
        from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer
        
        print("✅ Core modules imported successfully!")
        
        # Test basic functionality
        parser = AdvancedExcelParser()
        analyzer = FreeIntelligentAnalyzer()
        
        print("✅ Core functionality works!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🆓 XDP Analyzer - FREE Setup")
    print("=" * 40)
    print("Setting up your completely free document analysis platform...")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not install_requirements():
        return False
    
    print()
    
    # Download NLTK data
    download_nltk_data()
    
    print()
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed. Please check the error messages above.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    
    print("\n🚀 Quick Start Options:")
    print()
    print("1️⃣ Launch Web Interface (Recommended):")
    print("   streamlit run streamlit_app.py")
    print()
    print("2️⃣ Command Line Analysis:")
    print("   python main.py your_file.xlsx")
    print()
    print("3️⃣ Python API:")
    print("   from parsers.excel_parser import AdvancedExcelParser")
    print("   parser = AdvancedExcelParser()")
    print("   analysis = parser.parse('your_file.xlsx')")
    print()
    
    print("📚 Documentation: Check README.md")
    print("🆓 100% Free - No API keys needed!")
    print("🔒 Complete privacy - all processing local")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)