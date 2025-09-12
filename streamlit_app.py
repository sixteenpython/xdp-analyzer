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
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer
from analyzers.requirements_agent import RequirementsAgent

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
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: #fafafa;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e3f2fd;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    text-align: right;
}
.agent-message {
    background-color: #f1f8e9;
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    text-align: left;
}
.conversation-state {
    background-color: #fff3e0;
    padding: 0.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff9800;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

class StreamlitXDPAnalyzer:
    """Free Streamlit interface for XDP Analyzer"""
    
    def __init__(self):
        self.excel_parser = AdvancedExcelParser()
        self.intelligent_analyzer = FreeIntelligentAnalyzer()
        self.requirements_agent = RequirementsAgent(self.intelligent_analyzer)
    
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
            - 📝 Word: .docx *(coming soon)*
            - 🎯 PowerPoint: .pptx *(coming soon)*
            
            **Features:**
            - ✅ Formula Analysis
            - ✅ Business Logic Detection  
            - ✅ Risk Assessment
            - ✅ Automation Opportunities
            - ✅ Intelligent Insights
            - 🤖 Requirements Agent (NEW!)
            """)
            
            st.header("🛡️ Privacy First")
            st.success("All analysis runs locally on your machine. No data leaves your computer!")
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Analyze", "🤖 Requirements Agent", "📊 Dashboard", "ℹ️ About"])
        
        with tab1:
            self.upload_and_analyze_tab()
        
        with tab2:
            self.requirements_agent_tab()
        
        with tab3:
            self.dashboard_tab()
        
        with tab4:
            self.about_tab()
    
    def upload_and_analyze_tab(self):
        """File upload and analysis tab"""
        
        st.header("📁 Upload Your Document")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['xlsx', 'xlsm', 'xls'],
            help="Upload Excel files for comprehensive business intelligence analysis"
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
            
            excel_analysis = self.excel_parser.parse(tmp_path)
            st.session_state.excel_analysis = excel_analysis
            
            # Step 2: Intelligent analysis
            status_text.text("🧠 Performing intelligent analysis...")
            progress_bar.progress(50)
            
            intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(excel_analysis)
            st.session_state.intelligent_analysis = intelligent_analysis
            
            # Step 3: Generate insights
            status_text.text("💡 Generating insights...")
            progress_bar.progress(75)
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            self.display_analysis_results(excel_analysis, intelligent_analysis)
            
            # Clean up
            Path(tmp_path).unlink()
            
        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
            status_text.text("❌ Analysis failed")
    
    def display_analysis_results(self, excel_analysis, intelligent_analysis):
        """Display comprehensive analysis results"""
        
        st.success("🎉 Analysis completed successfully!")
        
        # Overview metrics
        st.subheader("📈 Document Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Worksheets", 
                len(excel_analysis.worksheets),
                help="Number of worksheets in the Excel file"
            )
        
        with col2:
            total_formulas = sum(len(ws.formulas) for ws in excel_analysis.worksheets)
            st.metric(
                "🧮 Formulas", 
                total_formulas,
                help="Total number of formulas found"
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
        
        # Display enhanced summary if available
        if hasattr(analysis, 'enhanced_summary') and analysis.enhanced_summary:
            self._display_enhanced_summary(analysis.enhanced_summary)
        else:
            # Fallback to basic summary
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
    
    def _display_enhanced_summary(self, enhanced_summary):
        """Display the enhanced business-focused summary"""
        
        # Generation method indicator
        method_indicator = {
            'llm': '🤖 AI-Generated',
            'hybrid': '🔄 AI + Local Analysis',
            'enhanced_local': '🧠 Advanced Local Analysis'
        }
        
        st.success(f"✨ **Enhanced Business Summary** - {method_indicator.get(enhanced_summary.generation_method, '📊 Generated')}")
        
        # Executive Summary (prominent display)
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary")
        st.markdown(enhanced_summary.executive_summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Two-column layout for key information
        col1, col2 = st.columns(2)
        
        with col1:
            # What this document does
            st.markdown("### 📄 What This Document Does")
            st.write(enhanced_summary.what_this_document_does)
            
            # Business Purpose
            st.markdown("### 🎯 Key Business Purpose")
            st.write(enhanced_summary.key_business_purpose)
            
            # Target Audience
            st.markdown("### 👥 Target Audience")
            st.write(enhanced_summary.target_audience_insights)
        
        with col2:
            # Document Complexity in Plain English
            st.markdown("### 📊 Complexity Assessment")
            st.info(enhanced_summary.document_complexity_plain_english)
            
            # Confidence Score
            confidence_color = "🟢" if enhanced_summary.confidence_score > 0.8 else "🟡" if enhanced_summary.confidence_score > 0.6 else "🔴"
            st.metric(
                "🎯 Analysis Confidence",
                f"{enhanced_summary.confidence_score:.1%}",
                help=f"Confidence in the analysis quality {confidence_color}"
            )
        
        # Main Findings
        if enhanced_summary.main_findings:
            st.markdown("### 🔍 Main Findings")
            for i, finding in enumerate(enhanced_summary.main_findings, 1):
                st.markdown(f"**{i}.** {finding}")
        
        # Business Implications
        if enhanced_summary.business_implications:
            st.markdown("### 💼 Business Implications")
            for implication in enhanced_summary.business_implications:
                st.markdown(f"• {implication}")
        
        # Actionable Insights (highlight box)
        if enhanced_summary.actionable_insights:
            st.markdown("### 💡 Actionable Insights")
            for i, insight in enumerate(enhanced_summary.actionable_insights, 1):
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong>💡 Insight {i}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Potential Concerns
        if enhanced_summary.potential_concerns:
            st.markdown("### ⚠️ Potential Concerns")
            for concern in enhanced_summary.potential_concerns:
                st.markdown(f"""
                <div class="risk-box">
                    <strong>⚠️</strong> {concern}
                </div>
                """, unsafe_allow_html=True)
        
        # Next Steps
        if enhanced_summary.next_steps_suggestions:
            st.markdown("### 🗺️ Suggested Next Steps")
            for i, step in enumerate(enhanced_summary.next_steps_suggestions, 1):
                st.markdown(f"**{i}.** {step}")
        
        # Detailed Content Analysis (NEW)
        if hasattr(enhanced_summary, 'detailed_content_analysis') and enhanced_summary.detailed_content_analysis:
            st.markdown("---")
            st.markdown("## 📖 Detailed Content Analysis")
            st.markdown("*Showing exactly what's in your document*")
            
            content_analysis = enhanced_summary.detailed_content_analysis
            
            # Create tabs for different views of the content
            content_tab1, content_tab2, content_tab3, content_tab4 = st.tabs([
                "📄 Worksheets", "🔍 Content Samples", "🏷️ Key Terms", "📊 Statistics"
            ])
            
            with content_tab1:
                st.markdown("### 📄 Worksheet Breakdown")
                if content_analysis.get('worksheet_breakdown'):
                    for ws in content_analysis['worksheet_breakdown']:
                        with st.expander(f"📋 Sheet: **{ws['name']}**", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Data Cells", ws.get('cell_count', 0))
                            with col2:
                                st.metric("Formulas", ws.get('formula_count', 0))
                            with col3:
                                features = []
                                if ws.get('has_charts'):
                                    features.append("📈 Charts")
                                if ws.get('has_pivot_tables'):
                                    features.append("📊 Pivot Tables")
                                st.write("Features: " + ", ".join(features) if features else "📋 Data Only")
                            
                            # Show sample content from this sheet
                            if ws.get('sample_content'):
                                st.markdown("**Sample Content:**")
                                for i, content in enumerate(ws['sample_content'][:5], 1):
                                    st.markdown(f"`{i}.` {content}")
            
            with content_tab2:
                st.markdown("### 🔍 Document Content Samples")
                if content_analysis.get('content_samples'):
                    st.markdown("Here are actual data samples found in your document:")
                    for i, sample in enumerate(content_analysis['content_samples'][:10], 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>Sample {i}:</strong> {sample}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No text content samples found in the document.")
            
            with content_tab3:
                st.markdown("### 🏷️ Key Terms Identified")
                key_terms = content_analysis.get('key_terms_found', {})
                if key_terms:
                    for category, terms in key_terms.items():
                        if terms:
                            category_name = category.replace('_', ' ').title()
                            st.markdown(f"**{category_name}:**")
                            terms_text = ", ".join(terms)
                            st.markdown(f"```{terms_text}```")
                else:
                    st.info("No specific business terms identified in the document.")
                
                # Most common words
                if content_analysis.get('most_common_words'):
                    st.markdown("**Most Common Words:**")
                    words_text = ", ".join(content_analysis['most_common_words'])
                    st.markdown(f"```{words_text}```")
            
            with content_tab4:
                st.markdown("### 📊 Content Statistics")
                stats = content_analysis.get('content_statistics', {})
                data_structure = content_analysis.get('data_structure', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Worksheets", data_structure.get('total_worksheets', 0))
                with col2:
                    st.metric("Total Data Cells", data_structure.get('total_data_cells', 0))
                with col3:
                    st.metric("Total Formulas", data_structure.get('total_formulas', 0))
                with col4:
                    st.metric("Text Items", stats.get('text_sample_count', 0))
                
                # Additional statistics
                if stats:
                    st.markdown("**Content Details:**")
                    st.write(f"• Total text length: {stats.get('total_text_length', 0)} characters")
                    st.write(f"• Unique text items: {stats.get('unique_text_items', 0)}")
                    st.write(f"• Complexity score: {data_structure.get('complexity_score', 0):.1f}")
        
        # Technical note
        st.markdown("---")
        st.caption(f"This summary was generated using {enhanced_summary.generation_method} analysis to provide business-focused, non-technical insights.")
    
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
    
    def requirements_agent_tab(self):
        """Requirements Agent conversational interface"""
        
        st.header("🤖 Requirements Agent")
        st.markdown("**Have a conversation with our AI agent to get tailored document analysis based on your specific needs.**")
        
        # Initialize session state for requirements agent
        if 'requirements_agent' not in st.session_state:
            st.session_state.requirements_agent = RequirementsAgent(self.intelligent_analyzer)
        
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
        
        if 'conversation_messages' not in st.session_state:
            st.session_state.conversation_messages = []
        
        if 'requirements_analysis_ready' not in st.session_state:
            st.session_state.requirements_analysis_ready = False
        
        # File upload section
        st.subheader("📁 Upload Your Document")
        uploaded_file = st.file_uploader(
            "Choose a file for requirements-based analysis",
            type=['xlsx', 'xlsm', 'xls'],
            help="Upload your document and describe your requirements to get tailored analysis"
        )
        
        if uploaded_file is not None:
            # File info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Filename", uploaded_file.name)
            with col2:
                st.metric("📊 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Store file in session state
            if 'uploaded_file_data' not in st.session_state:
                st.session_state.uploaded_file_data = uploaded_file.getbuffer()
                st.session_state.uploaded_filename = uploaded_file.name
            
            # Initial requirements input
            if not st.session_state.conversation_started:
                st.subheader("🎯 Tell me about your requirements")
                
                user_description = st.text_area(
                    "Describe what you need from this analysis:",
                    placeholder="Example: I need to analyze this financial model for budget planning. I want to understand if the formulas are accurate and identify any risks before presenting to management.",
                    height=100,
                    help="Be as specific as possible about your goals, audience, and what you hope to achieve"
                )
                
                if st.button("🚀 Start Conversation", type="primary", disabled=not user_description.strip()):
                    if user_description.strip():
                        # Start conversation
                        file_type = uploaded_file.name.split('.')[-1].upper()
                        response = st.session_state.requirements_agent.start_conversation(
                            user_description, 
                            file_type
                        )
                        
                        st.session_state.conversation_started = True
                        st.session_state.conversation_messages = [
                            {"sender": "user", "message": user_description, "type": "initial"},
                            {"sender": "agent", "message": response, "type": "response"}
                        ]
                        st.rerun()
            
            # Conversation interface
            if st.session_state.conversation_started:
                self._display_conversation_interface()
        
        else:
            st.info("👆 Please upload a document to start the requirements conversation.")
            
            # Show demo conversation
            with st.expander("💡 See Example Conversation"):
                st.markdown("""
                **Example interaction:**
                
                **You:** "I have an Excel financial model and need help understanding if it's suitable for budget planning"
                
                **Agent:** "Thanks for providing that context about your Excel document. I can see this is related to financial analysis. Let me ask a few targeted questions so I can tailor my analysis to your specific needs. What is the main business problem this document is meant to solve?"
                
                **You:** "I need to present this to senior management for quarterly budget approval"
                
                **Agent:** "That makes sense. Building on what you said, who is your target audience for this analysis? Are we talking about C-level executives, finance team, or a broader audience?"
                
                And so on... The agent will ask 3-5 targeted questions to understand your specific needs.
                """)
    
    def _display_conversation_interface(self):
        """Display the conversational interface"""
        
        # Conversation history
        st.subheader("💬 Conversation")
        
        # Display conversation messages
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.conversation_messages:
                if msg["sender"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {msg["message"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="agent-message">
                        <strong>🤖 Agent:</strong> {msg["message"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show conversation state
        agent_state = st.session_state.requirements_agent.state.value
        st.markdown(f"""
        <div class="conversation-state">
            <strong>Status:</strong> {agent_state.replace('_', ' ').title()}
        </div>
        """, unsafe_allow_html=True)
        
        # Input for next response (if conversation not completed)
        if agent_state != "completed":
            with st.form("user_response_form", clear_on_submit=True):
                user_response = st.text_input(
                    "Your response:",
                    placeholder="Type your answer here...",
                    key="user_input_field"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    send_button = st.form_submit_button("📤 Send")
                
                with col2:
                    skip_button = st.form_submit_button("⏭️ Skip Question")
                
                if send_button and user_response.strip():
                    # Process user response
                    agent_response = st.session_state.requirements_agent.process_user_response(user_response)
                    
                    # Add messages to conversation
                    st.session_state.conversation_messages.extend([
                        {"sender": "user", "message": user_response, "type": "response"},
                        {"sender": "agent", "message": agent_response, "type": "response"}
                    ])
                    st.rerun()
                
                elif skip_button:
                    # Skip current question
                    agent_response = st.session_state.requirements_agent.process_user_response("I'd prefer to skip this question")
                    
                    st.session_state.conversation_messages.append({
                        "sender": "agent", 
                        "message": agent_response, 
                        "type": "response"
                    })
                    st.rerun()
        
        # Analysis button (when conversation is complete)
        if agent_state == "completed":
            st.success("🎉 Conversation completed! Ready for analysis.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Generate Tailored Analysis", type="primary", use_container_width=True):
                    self._generate_requirements_based_analysis()
            
            with col2:
                if st.button("🔄 Start New Conversation", use_container_width=True):
                    self._reset_requirements_conversation()
        
        # Show gathered requirements summary
        if agent_state == "completed":
            st.subheader("📋 Gathered Requirements Summary")
            context = st.session_state.requirements_agent.context
            
            col1, col2 = st.columns(2)
            
            with col1:
                if context.primary_purpose:
                    st.markdown(f"**🎯 Purpose:** {context.primary_purpose}")
                if context.target_audience:
                    st.markdown(f"**👥 Audience:** {context.target_audience}")
                if context.business_domain:
                    st.markdown(f"**🏢 Domain:** {context.business_domain.title()}")
            
            with col2:
                if context.priorities:
                    priorities_text = "; ".join(context.priorities[:2])
                    st.markdown(f"**⭐ Priorities:** {priorities_text}")
                if context.timeline:
                    st.markdown(f"**⏰ Timeline:** {context.timeline}")
                if context.analysis_scope:
                    st.markdown(f"**🔍 Scope:** {context.analysis_scope}")
        
        # Display requirements-based analysis if ready
        if st.session_state.requirements_analysis_ready and 'requirements_analysis' in st.session_state:
            self._display_requirements_analysis()
    
    def _generate_requirements_based_analysis(self):
        """Generate analysis based on gathered requirements"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.uploaded_filename.split('.')[-1]}") as tmp_file:
                tmp_file.write(st.session_state.uploaded_file_data)
                tmp_path = tmp_file.name
            
            # Step 1: Parse document
            status_text.text("🔄 Parsing document...")
            progress_bar.progress(25)
            
            excel_analysis = self.excel_parser.parse(tmp_path)
            
            # Step 2: Standard intelligent analysis
            status_text.text("🧠 Performing standard analysis...")
            progress_bar.progress(50)
            
            intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(excel_analysis)
            
            # Step 3: Requirements-based analysis
            status_text.text("🎯 Generating tailored analysis based on your requirements...")
            progress_bar.progress(75)
            
            requirements_analysis = st.session_state.requirements_agent.generate_requirements_analysis(
                intelligent_analysis
            )
            
            # Step 4: Complete
            status_text.text("✅ Requirements-based analysis complete!")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.excel_analysis = excel_analysis
            st.session_state.intelligent_analysis = intelligent_analysis
            st.session_state.requirements_analysis = requirements_analysis
            st.session_state.requirements_analysis_ready = True
            
            # Clean up
            Path(tmp_path).unlink()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            status_text.text("❌ Analysis failed")
    
    def _display_requirements_analysis(self):
        """Display the requirements-based analysis results"""
        
        st.header("🎯 Your Tailored Analysis")
        analysis = st.session_state.requirements_analysis
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Analysis Type", "Requirements-Based")
        with col2:
            st.metric("🎯 Confidence", f"{analysis.confidence_score:.1%}")
        with col3:
            st.metric("💬 Questions Asked", len(st.session_state.requirements_agent.questions_asked))
        
        # Tabbed results
        req_tab1, req_tab2, req_tab3, req_tab4, req_tab5 = st.tabs([
            "📋 Summary", 
            "💡 Tailored Insights", 
            "🛠️ Recommendations", 
            "⚠️ Risk Assessment",
            "🗺️ Implementation Roadmap"
        ])
        
        with req_tab1:
            st.markdown("### 📋 Requirements Summary")
            st.markdown(analysis.requirements_summary)
            
            st.markdown("### 🎯 Analysis Overview")
            st.write(analysis.conversation_context.user_description)
        
        with req_tab2:
            st.markdown("### 💡 Tailored Insights")
            for i, insight in enumerate(analysis.tailored_insights, 1):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>💡 Insight {i}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        with req_tab3:
            st.markdown("### 🛠️ Specific Recommendations")
            for i, rec in enumerate(analysis.specific_recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong>✅ Recommendation {i}:</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
        
        with req_tab4:
            st.markdown("### ⚠️ Risk Assessment")
            if analysis.risk_assessment:
                for i, risk in enumerate(analysis.risk_assessment, 1):
                    st.markdown(f"""
                    <div class="risk-box">
                        <strong>⚠️ Risk {i}:</strong> {risk}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 No significant risks identified based on your requirements!")
        
        with req_tab5:
            st.markdown("### 🗺️ Implementation Roadmap")
            for i, step in enumerate(analysis.implementation_roadmap, 1):
                st.markdown(f"**{i}.** {step}")
            
            if analysis.success_metrics:
                st.markdown("### 📈 Success Metrics")
                for metric in analysis.success_metrics:
                    st.write(f"• {metric}")
        
        # Export functionality
        st.markdown("### 📤 Export Results")
        
        if st.button("💾 Download Requirements Analysis Report", type="secondary"):
            report_data = {
                'requirements_summary': analysis.requirements_summary,
                'conversation_context': {
                    'user_description': analysis.conversation_context.user_description,
                    'primary_purpose': analysis.conversation_context.primary_purpose,
                    'target_audience': analysis.conversation_context.target_audience,
                    'business_domain': analysis.conversation_context.business_domain,
                    'priorities': analysis.conversation_context.priorities,
                    'timeline': analysis.conversation_context.timeline,
                },
                'tailored_insights': analysis.tailored_insights,
                'specific_recommendations': analysis.specific_recommendations,
                'risk_assessment': analysis.risk_assessment,
                'implementation_roadmap': analysis.implementation_roadmap,
                'success_metrics': analysis.success_metrics,
                'confidence_score': analysis.confidence_score,
                'conversation_history': [
                    {'sender': msg['sender'], 'message': msg['message']} 
                    for msg in st.session_state.conversation_messages
                ]
            }
            
            st.download_button(
                label="📄 Download JSON Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"requirements_analysis_{st.session_state.uploaded_filename.split('.')[0]}.json",
                mime="application/json"
            )
    
    def _reset_requirements_conversation(self):
        """Reset the requirements conversation"""
        
        # Reset agent state
        st.session_state.requirements_agent.reset_conversation()
        
        # Clear session state
        st.session_state.conversation_started = False
        st.session_state.conversation_messages = []
        st.session_state.requirements_analysis_ready = False
        
        if 'requirements_analysis' in st.session_state:
            del st.session_state.requirements_analysis
        
        st.rerun()
    
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
        - **Word**: .docx *(coming soon)*
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

def main():
    """Main application entry point"""
    app = StreamlitXDPAnalyzer()
    app.run()

if __name__ == "__main__":
    main()