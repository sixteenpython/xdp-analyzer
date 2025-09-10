"""
XDP Analyzer - Free Streamlit Web Interface
100% Free document analysis with beautiful web UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import tempfile
from pathlib import Path
import sys
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add our modules to path
sys.path.append(str(Path(__file__).parent))

from parsers.excel_parser import AdvancedExcelParser
from parsers.word_parser import AdvancedWordParser
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer

# Page configuration
st.set_page_config(
    page_title="XDP Analyzer - Free Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
}
.insight-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #1f77b4;
    margin: 1rem 0;
}
.recommendation-box {
    background-color: #f0f9ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #10b981;
    margin: 1rem 0;
}
.risk-box {
    background-color: #fef2f2;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ef4444;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class StreamlitXDPAnalyzer:
    """Free Streamlit interface for XDP Analyzer"""
    
    def __init__(self):
        self.excel_parser = AdvancedExcelParser()
        self.word_parser = AdvancedWordParser()
        self.intelligent_analyzer = FreeIntelligentAnalyzer()
    
    def run(self):
        """Main Streamlit app"""
        
        # Header
        st.markdown('<h1 class="main-header">📊 XDP Analyzer</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">🆓 <strong>100% Free</strong> Document Intelligence • No APIs • No Costs • Complete Privacy</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("🚀 Getting Started")
            st.markdown("""
            **Supported Files:**
            - 📊 Excel: .xlsx, .xlsm, .xls
            - 📝 Word: .docx ✅
            - 🎯 PowerPoint: .pptx *(coming soon)*
            
            **Features:**
            - ✅ Formula Analysis (Excel)
            - ✅ Business Logic Detection
            - ✅ Word Document Analysis **NEW!**
            - ✅ Text Content Intelligence  
            - ✅ Risk Assessment
            - ✅ Automation Opportunities
            - ✅ Intelligent Insights
            """)
            
            st.header("🛡️ Privacy First")
            st.success("All analysis runs locally on your machine. No data leaves your computer!")
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Dashboard", "ℹ️ About"])
        
        with tab1:
            self.upload_and_analyze_tab()
        
        with tab2:
            self.dashboard_tab()
        
        with tab3:
            self.about_tab()
    
    def upload_and_analyze_tab(self):
        """File upload and analysis tab"""
        
        st.header("📁 Upload Your Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['xlsx', 'xlsm', 'xls', 'docx'],
            help="Upload Excel or Word files for comprehensive business intelligence analysis"
        )
        
        if uploaded_file is not None:
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Filename", uploaded_file.name)
            with col2:
                st.metric("📊 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("🗂️ File Type", uploaded_file.type.split('/')[-1].upper())
            
            # Analyze button
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                self.analyze_document(uploaded_file)
    
    def analyze_document(self, uploaded_file):
        """Analyze uploaded document"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Step 1: Parse document
            status_text.text("🔄 Parsing document structure...")
            progress_bar.progress(25)
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['xlsx', 'xlsm', 'xls']:
                # Excel document
                document_analysis = self.excel_parser.parse(tmp_path)
                st.session_state.document_analysis = document_analysis
                st.session_state.document_type = 'excel'
                
                # Step 2: Intelligent analysis
                status_text.text("🧠 Performing intelligent analysis...")
                progress_bar.progress(50)
                
                intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(document_analysis)
                
            elif file_extension == 'docx':
                # Word document
                document_analysis = self.word_parser.parse(tmp_path)
                st.session_state.document_analysis = document_analysis
                st.session_state.document_type = 'word'
                
                # Step 2: Intelligent analysis
                status_text.text("🧠 Performing intelligent analysis...")
                progress_bar.progress(50)
                
                intelligent_analysis = self.intelligent_analyzer.analyze_word_content(document_analysis)
                
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            st.session_state.intelligent_analysis = intelligent_analysis
            
            # Step 3: Generate insights
            status_text.text("💡 Generating insights...")
            progress_bar.progress(75)
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            self.display_analysis_results(document_analysis, intelligent_analysis, st.session_state.document_type)
            
            # Clean up
            Path(tmp_path).unlink()
            
        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
            status_text.text("❌ Analysis failed")
    
    def display_analysis_results(self, document_analysis, intelligent_analysis, document_type):
        """Display comprehensive analysis results"""
        
        st.success("🎉 Analysis completed successfully!")
        
        # Enhanced Document Summary Section
        st.markdown("## 📋 Document Summary")
        
        # Create an enhanced, detailed summary based on document type
        if document_type == 'excel':
            summary_text = self._generate_excel_summary(document_analysis, intelligent_analysis)
        elif document_type == 'word':
            summary_text = self._generate_word_summary(document_analysis, intelligent_analysis)
        else:
            summary_text = "Document processed successfully."
        
        # Display the summary in a nice box
        st.markdown(f"""
        <div class="insight-box" style="background-color: #f0f9ff; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #1f77b4; margin: 1rem 0;">
            <h4 style="color: #1f77b4; margin-bottom: 1rem;">📄 What is this document?</h4>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #2d3748; margin-bottom: 0;">{summary_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overview metrics
        st.subheader("📈 Document Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if document_type == 'excel':
            with col1:
                st.metric(
                    "📊 Worksheets", 
                    len(document_analysis.worksheets),
                    help="Number of worksheets in the Excel file"
                )
            
            with col2:
                total_formulas = sum(len(ws.formulas) for ws in document_analysis.worksheets)
                st.metric(
                    "🧮 Formulas", 
                    total_formulas,
                    help="Total number of formulas found"
                )
        
        elif document_type == 'word':
            with col1:
                st.metric(
                    "📄 Pages", 
                    document_analysis.page_count,
                    help="Number of pages in the Word document"
                )
            
            with col2:
                st.metric(
                    "📝 Words", 
                    f"{document_analysis.total_words:,}",
                    help="Total number of words"
                )
        
        with col3:
            st.metric(
                "🎯 Confidence", 
                f"{intelligent_analysis.confidence_score:.1%}",
                help="Analysis confidence score"
            )
        
        with col4:
            risk_count = len(intelligent_analysis.risk_indicators)
            st.metric(
                "⚠️ Risk Items", 
                risk_count,
                delta=f"-{risk_count}" if risk_count > 0 else "No risks",
                delta_color="inverse",
                help="Number of risk indicators found"
            )
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Summary", "💡 Insights", "🛠️ Recommendations", "⚠️ Risks", "📊 Details"])
        
        with tab1:
            self.display_summary_tab(intelligent_analysis)
        
        with tab2:
            self.display_insights_tab(intelligent_analysis)
        
        with tab3:
            self.display_recommendations_tab(intelligent_analysis)
        
        with tab4:
            self.display_risks_tab(intelligent_analysis)
        
        with tab5:
            self.display_details_tab(excel_analysis, intelligent_analysis)
    
    def display_summary_tab(self, analysis):
        """Display summary information"""
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary")
        st.write(analysis.summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Business Logic")
        st.write(analysis.business_logic)
        
        # Content themes
        if analysis.content_themes:
            st.markdown("### 🏷️ Content Themes")
            
            # Create theme badges
            theme_html = ""
            for theme in analysis.content_themes:
                theme_html += f'<span style="background-color: #1f77b4; color: white; padding: 0.2rem 0.5rem; border-radius: 1rem; margin: 0.2rem; display: inline-block;">{theme}</span>'
            
            st.markdown(theme_html, unsafe_allow_html=True)
    
    def display_insights_tab(self, analysis):
        """Display key insights"""
        
        st.markdown("### 💡 Key Insights")
        
        for i, insight in enumerate(analysis.key_insights, 1):
            st.markdown(f"""
            <div class="insight-box">
                <strong>💡 Insight {i}:</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Keyword analysis visualization
        if analysis.keyword_analysis and 'keyword_density' in analysis.keyword_analysis:
            st.markdown("### 🔤 Keyword Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Word cloud
                if analysis.keyword_analysis['keyword_density']:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis'
                    ).generate_from_frequencies(analysis.keyword_analysis['keyword_density'])
                    
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            with col2:
                # Top keywords
                st.markdown("**🏆 Top Keywords:**")
                for keyword, count in list(analysis.keyword_analysis['keyword_density'].items())[:10]:
                    st.write(f"• {keyword}: {count}")
        
        # Sentiment analysis
        if analysis.sentiment_analysis and analysis.sentiment_analysis.get('compound', 0) != 0:
            st.markdown("### 😊 Sentiment Analysis")
            
            sentiment = analysis.sentiment_analysis
            
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment['compound'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Sentiment"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "lightcoral"},
                        {'range': [-0.3, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def display_recommendations_tab(self, analysis):
        """Display recommendations"""
        
        st.markdown("### 🛠️ Actionable Recommendations")
        
        for i, recommendation in enumerate(analysis.recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>✅ Recommendation {i}:</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        # Automation opportunities
        if analysis.automation_opportunities:
            st.markdown("### 🤖 Automation Opportunities")
            
            for i, opportunity in enumerate(analysis.automation_opportunities, 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>🚀 Opportunity {i}:</strong> {opportunity}
                </div>
                """, unsafe_allow_html=True)
    
    def display_risks_tab(self, analysis):
        """Display risk assessment"""
        
        st.markdown("### ⚠️ Risk Assessment")
        
        if analysis.risk_indicators:
            for i, risk in enumerate(analysis.risk_indicators, 1):
                st.markdown(f"""
                <div class="risk-box">
                    <strong>⚠️ Risk {i}:</strong> {risk}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No major risks identified!")
        
        # Risk mitigation suggestions
        st.markdown("### 🛡️ Risk Mitigation Suggestions")
        
        mitigation_suggestions = [
            "Implement regular backup procedures",
            "Add data validation to prevent input errors", 
            "Document complex formulas with comments",
            "Test formulas with edge cases",
            "Consider formula auditing tools"
        ]
        
        for suggestion in mitigation_suggestions:
            st.write(f"• {suggestion}")
    
    def display_details_tab(self, excel_analysis, intelligent_analysis):
        """Display detailed technical information"""
        
        st.markdown("### 📊 Technical Details")
        
        # Worksheet details
        st.markdown("#### 📋 Worksheet Analysis")
        
        worksheet_data = []
        for ws in excel_analysis.worksheets:
            worksheet_data.append({
                'Worksheet': ws.name,
                'Cells with Data': len(ws.cells),
                'Formulas': len(ws.formulas),
                'Charts': len(ws.charts),
                'Tables': len(ws.pivot_tables),
                'Hidden': ws.hidden
            })
        
        if worksheet_data:
            df_worksheets = pd.DataFrame(worksheet_data)
            st.dataframe(df_worksheets, use_container_width=True)
            
            # Visualization
            if len(worksheet_data) > 1:
                fig = px.bar(
                    df_worksheets, 
                    x='Worksheet', 
                    y='Formulas',
                    title="Formula Distribution by Worksheet",
                    color='Formulas',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Readability scores
        if intelligent_analysis.readability_scores:
            st.markdown("#### 📖 Readability Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                for score_name, score_value in intelligent_analysis.readability_scores.items():
                    if isinstance(score_value, (int, float)):
                        st.metric(
                            score_name.replace('_', ' ').title(),
                            f"{score_value:.1f}"
                        )
        
        # Export functionality
        st.markdown("### 📤 Export Results")
        
        if st.button("💾 Download Analysis Report", type="secondary"):
            # Create downloadable JSON report
            report = {
                'summary': intelligent_analysis.summary,
                'insights': intelligent_analysis.key_insights,
                'recommendations': intelligent_analysis.recommendations,
                'risks': intelligent_analysis.risk_indicators,
                'automation_opportunities': intelligent_analysis.automation_opportunities,
                'technical_details': {
                    'worksheets': len(excel_analysis.worksheets),
                    'total_formulas': sum(len(ws.formulas) for ws in excel_analysis.worksheets),
                    'confidence_score': intelligent_analysis.confidence_score
                }
            }
            
            st.download_button(
                label="📄 Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name="xdp_analysis_report.json",
                mime="application/json"
            )
    
    def dashboard_tab(self):
        """Dashboard overview tab"""
        
        st.header("📊 Analysis Dashboard")
        
        if 'excel_analysis' not in st.session_state:
            st.info("👆 Upload and analyze a document in the first tab to see the dashboard!")
            return
        
        # Display cached analysis
        excel_analysis = st.session_state.excel_analysis
        intelligent_analysis = st.session_state.intelligent_analysis
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Worksheets", len(excel_analysis.worksheets))
        
        with col2:
            total_formulas = sum(len(ws.formulas) for ws in excel_analysis.worksheets)
            st.metric("🧮 Total Formulas", total_formulas)
        
        with col3:
            st.metric("🎯 Analysis Confidence", f"{intelligent_analysis.confidence_score:.1%}")
        
        # Quick insights
        st.subheader("⚡ Quick Insights")
        
        for insight in intelligent_analysis.key_insights[:3]:
            st.info(f"💡 {insight}")
    
    def about_tab(self):
        """About and help tab"""
        
        st.header("ℹ️ About XDP Analyzer")
        
        st.markdown("""
        ### 🎯 What is XDP Analyzer?
        
        XDP Analyzer is a **100% free** document intelligence platform that analyzes Excel, Word, and PowerPoint files 
        to extract business insights, identify risks, and suggest improvements.
        
        ### ✨ Key Features
        
        - 🆓 **Completely Free** - No subscriptions, no API costs, no hidden fees
        - 🔒 **Privacy First** - All processing happens locally on your machine
        - 🧠 **Intelligent Analysis** - Advanced rule-based analysis without external AI
        - 📊 **Business Insights** - Understand your documents' business logic
        - ⚠️ **Risk Assessment** - Identify potential issues and vulnerabilities
        - 🤖 **Automation Opportunities** - Find areas for process improvement
        
        ### 🛠️ Supported File Types
        
        - **Excel**: .xlsx, .xlsm, .xls
        - **Word**: .docx ✅ **NEW!**
        - **PowerPoint**: .pptx *(coming soon)*
        
        ### 🚀 How It Works
        
        1. **Upload** your document using the file uploader
        2. **Analyze** - Our algorithms examine structure, formulas, and content
        3. **Insights** - Get intelligent recommendations and risk assessments
        4. **Export** - Download detailed analysis reports
        
        ### 🔧 Technology Stack
        
        - **Frontend**: Streamlit (Free)
        - **Document Parsing**: openpyxl, python-docx, python-pptx (Free)
        - **Text Analysis**: NLTK, textstat (Free) 
        - **Visualization**: Plotly, Matplotlib (Free)
        - **Intelligence**: Rule-based algorithms (No external APIs)
        
        ### 📞 Support
        
        For questions or issues, please check our documentation or create an issue on GitHub.
        
        ### 📜 License
        
        This project is open source and available under the MIT License.
        
        ---
        
        **Built with ❤️ for the community**
        """)
    
    def _generate_excel_summary(self, excel_analysis, intelligent_analysis):
        """Generate detailed Excel document summary"""
        
        # Basic document info
        file_name = excel_analysis.filename
        worksheet_count = len(excel_analysis.worksheets)
        formula_count = getattr(intelligent_analysis, 'formula_analysis', {}).get('total_formulas', 0)
        
        # Document type classification
        if formula_count > 50 and worksheet_count > 3:
            doc_type = "comprehensive financial model or business analysis workbook"
        elif formula_count > 20:
            doc_type = "calculation-intensive spreadsheet with business logic"
        elif worksheet_count > 5:
            doc_type = "multi-sheet data organization workbook"
        elif formula_count > 0:
            doc_type = "analytical spreadsheet with calculations"
        else:
            doc_type = "data storage or reference spreadsheet"
        
        # Business context
        business_context = ""
        if any("budget" in sheet.name.lower() for sheet in excel_analysis.worksheets):
            business_context += " with budget planning elements"
        if any("revenue" in str(getattr(intelligent_analysis, 'business_logic', '')).lower()):
            business_context += " containing revenue analysis"
        if getattr(intelligent_analysis, 'risk_indicators', []):
            business_context += f" with {len(intelligent_analysis.risk_indicators)} identified risk factors"
        
        # Complexity assessment
        complexity = getattr(intelligent_analysis, 'formula_analysis', {}).get('complexity_level', 'Unknown')
        
        summary = f"""
        This Excel file "{file_name}" is a {doc_type}{business_context}. 
        
        The workbook contains {worksheet_count} worksheet{"s" if worksheet_count != 1 else ""} with {formula_count} formula{"s" if formula_count != 1 else ""}, 
        indicating a {complexity.lower()}-complexity document. {getattr(intelligent_analysis, 'summary', 'Document analysis completed.')}
        
        This appears to be used for {"business analysis and decision-making" if formula_count > 10 else "data organization and basic calculations"}.
        """
        
        return summary.strip()
    
    def _generate_word_summary(self, word_analysis, intelligent_analysis):
        """Generate detailed Word document summary"""
        
        # Basic document info  
        file_name = word_analysis.filename
        word_count = word_analysis.total_words
        page_count = word_analysis.page_count
        headings_count = len(word_analysis.headings)
        tables_count = word_analysis.tables_count
        
        # Document type classification
        if headings_count > 10 and word_count > 2000:
            doc_type = "comprehensive report or detailed documentation"
        elif headings_count > 5:
            doc_type = "structured business document"
        elif tables_count > 3:
            doc_type = "data-rich report with tabular information"
        elif word_count > 1000:
            doc_type = "detailed written document"
        else:
            doc_type = "brief business document"
        
        # Content analysis
        content_focus = ""
        themes = getattr(intelligent_analysis, 'themes', [])
        if 'strategy' in themes:
            content_focus += " focused on strategic planning"
        elif 'finance' in themes:
            content_focus += " with financial analysis content"
        elif 'project' in themes:
            content_focus += " related to project management"
        elif themes:
            content_focus += f" covering {', '.join(themes).lower()} topics"
        
        # Reading time and complexity
        reading_time = word_analysis.reading_time
        time_desc = f"approximately {reading_time:.1f} minutes to read"
        
        summary = f"""
        This Word document "{file_name}" is a {doc_type}{content_focus}. 
        
        The document spans {page_count} page{"s" if page_count != 1 else ""} with {word_count:,} words and takes {time_desc}. 
        It contains {headings_count} heading{"s" if headings_count != 1 else ""} and {tables_count} table{"s" if tables_count != 1 else ""}, 
        suggesting {"well-structured content" if headings_count > 0 else "simple formatting"}.
        
        {getattr(intelligent_analysis, 'summary', 'Document analysis completed.')}
        
        This document appears to be {"a formal business report" if headings_count > 3 else "an informal document"} 
        suitable for {"executive review" if word_count > 1500 else "quick reference"}.
        """
        
        return summary.strip()

def main():
    """Main application entry point"""
    app = StreamlitXDPAnalyzer()
    app.run()

if __name__ == "__main__":
    main()