# 🆓 XDP Analyzer - 100% FREE Edition

**Completely free** document analysis platform for Excel, Word, and PowerPoint files with intelligent insights and business process intelligence. **No APIs, No Costs, Complete Privacy!**

## ✨ Key Highlights

🆓 **100% FREE** - No subscriptions, no API costs, no hidden fees  
🔒 **Privacy First** - All processing happens locally on your machine  
🚀 **No Setup** - Just install Python packages and run  
💡 **Smart Analysis** - Advanced rule-based intelligence without external AI  
📊 **Beautiful UI** - Modern Streamlit web interface  

## 🚀 Features

### Document Parsing
- **Excel Support**: xlsx, xlsm, xlsb, xls formats
- **Word Support**: Coming soon
- **PowerPoint Support**: Coming soon

### Business Intelligence (FREE)
- **Formula Analysis**: Deep dive into Excel calculations and business logic
- **VBA Code Extraction**: Analyze automation scripts and macros
- **Risk Assessment**: Flag potential issues and vulnerabilities
- **Automation Opportunities**: Identify areas for improvement
- **Content Analysis**: Extract themes, sentiment, and keywords
- **🤖 Requirements Agent (NEW!)**: Conversational AI that gathers requirements and provides tailored analysis

### Local Intelligence (No External APIs)
- **Rule-Based Analysis**: Pattern recognition using local algorithms
- **NLTK Integration**: Free natural language processing
- **Sentiment Analysis**: Local sentiment scoring
- **Keyword Extraction**: Term frequency and importance analysis
- **Readability Scoring**: Text complexity assessment

## 📁 Project Structure

```
xdp-analyzer/
├── parsers/                    # Document parsers
│   ├── excel_parser.py        # Excel file parser
│   ├── word_parser.py         # Word document parser
│   └── powerpoint_parser.py   # PowerPoint parser
├── llm_integration/           # AI/LLM integrations
│   └── multi_llm_analyzer.py  # Multi-provider LLM system
├── analyzers/                 # Analysis engines
│   ├── intelligent_analyzer.py        # Free intelligent analysis
│   ├── requirements_agent.py         # Conversational requirements agent
│   └── business_process_mapper.py     # Process mapping
├── tests/                     # Test suite
│   └── test_suite.py         # Comprehensive tests
└── requirements.txt          # Dependencies
```

## 🛠️ Installation (Super Easy!)

1. **Navigate to the project folder**:
```bash
cd xdp-analyzer
```

2. **Install free dependencies** (one-time setup):
```bash
pip install -r requirements.txt
```

3. **That's it!** No API keys needed, no configuration required!

## 🚀 Quick Start

### Option 1: Beautiful Web Interface (Recommended)

```bash
# Launch the web app
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` and start analyzing! 🎉

### Option 2: Command Line

```python
from parsers.excel_parser import AdvancedExcelParser
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer

# Parse Excel file
parser = AdvancedExcelParser()
analysis = parser.parse("your_file.xlsx")

# Get intelligent insights (FREE)
analyzer = FreeIntelligentAnalyzer()
insights = analyzer.analyze_excel_content(analysis)

print(f"📊 Worksheets: {len(analysis.worksheets)}")
print(f"🧮 Formulas: {len(analysis.formulas_summary)}")
print(f"💡 Summary: {insights.summary}")
print(f"⚠️ Risks: {len(insights.risk_indicators)}")
print("\n🛠️ Recommendations:")
for rec in insights.recommendations:
    print(f"- {rec}")
```

### Option 3: Command Line Tool

```bash
# Analyze a file directly
python main.py path/to/your/file.xlsx

# With detailed output
python main.py path/to/your/file.xlsx --output analysis_report.json
```

## 🤖 Requirements Agent - NEW FEATURE!

The **Requirements Agent** is a conversational AI that engages with you to understand your specific needs before analyzing your documents. This ensures you get tailored, relevant insights.

### How it works:
1. **Upload** your document
2. **Describe** your requirements in natural language
3. **Conversation** - The agent asks 3-5 targeted questions about:
   - Your business goals and objectives  
   - Target audience (executives, analysts, etc.)
   - Analysis scope and priorities
   - Timeline and constraints
   - Success criteria
4. **Tailored Analysis** - Get customized insights, recommendations, and roadmaps

### Example Conversation:
```
You: "I have an Excel financial model and need help understanding if it's suitable for budget planning"

Agent: "Thanks for that context. I can see this is related to financial analysis. What is the main business problem this document is meant to solve?"

You: "I need to present this to senior management for quarterly budget approval"

Agent: "That makes sense. Who is your target audience for this analysis? Are we talking about C-level executives, finance team, or a broader audience?"

You: "C-level executives who need high-level insights"

Agent: "Perfect. What are your top 3 priorities for this analysis?"

[Conversation continues...]
```

### Benefits:
- ✅ **Personalized Analysis** - Tailored to your specific needs
- ✅ **Context-Aware Insights** - Understands your business domain  
- ✅ **Audience-Specific Output** - Matches your target audience
- ✅ **Implementation Roadmap** - Actionable next steps
- ✅ **Risk Assessment** - Customized to your constraints

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/test_suite.py -v

# Run specific test categories
python -m pytest tests/test_suite.py -k "excel" -v
python -m pytest tests/test_suite.py -k "integration" -v
python -m pytest tests/test_suite.py -k "performance" -v
```

## 📈 Use Cases

### Financial Analysis
- **Budget Planning**: Analyze financial models and forecasts
- **Risk Assessment**: Identify calculation errors and dependencies
- **Audit Support**: Document business logic and data flows

### Business Process Optimization
- **Workflow Mapping**: Visualize process flows from Excel workflows
- **Automation Opportunities**: Identify manual processes for automation
- **Compliance**: Document business rules and decision logic

### Document Intelligence
- **Content Extraction**: Extract structured data from documents
- **Pattern Recognition**: Identify common business patterns
- **Multi-format Analysis**: Analyze Excel, Word, and PowerPoint together

## ⚙️ Configuration

### Environment Variables

```bash
# Required for LLM features
OPENAI_API_KEY=your_openai_api_key
CLAUDE_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional configurations
XDP_LOG_LEVEL=INFO
XDP_CACHE_SIZE=1000
XDP_MAX_FILE_SIZE=100MB
```

### Configuration File (config.json)

```json
{
    "llm_providers": {
        "openai": {
            "model": "gpt-4-turbo-preview",
            "max_tokens": 1000,
            "temperature": 0.3
        },
        "claude": {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "temperature": 0.3
        }
    },
    "parsing": {
        "max_complexity_score": 100,
        "enable_vba_analysis": true,
        "extract_images": true
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenPyXL team for Excel parsing capabilities
- Python-docx and python-pptx for Office document support
- OpenAI, Anthropic, and Google for LLM APIs
- NetworkX for graph analysis capabilities

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Open a GitHub issue
- **Documentation**: Check the `/docs` folder (coming soon)
- **Examples**: See `/examples` folder for more use cases

---

**Built with ❤️ for document intelligence and business process analysis**