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
        """Generate intelligent, LLM-like Excel document summary"""
        
        # Safe data extraction
        file_name = getattr(excel_analysis, 'filename', 'Unknown file')
        worksheets = getattr(excel_analysis, 'worksheets', [])
        if isinstance(worksheets, bool):
            worksheets = []
        worksheet_count = len(worksheets) if worksheets else 1
        
        formula_analysis = getattr(intelligent_analysis, 'formula_analysis', {})
        formula_count = formula_analysis.get('total_formulas', 0)
        if isinstance(formula_count, bool):
            formula_count = 0
            
        insights = getattr(intelligent_analysis, 'key_insights', [])
        themes = getattr(intelligent_analysis, 'themes', [])
        risks = getattr(intelligent_analysis, 'risk_indicators', [])
        
        # Intelligent document purpose detection
        purpose_analysis = self._analyze_excel_purpose(worksheets, formula_count, insights, themes)
        
        # Business context understanding
        business_intelligence = self._extract_business_intelligence(worksheets, formula_analysis, insights)
        
        # Generate human-like narrative summary
        if formula_count > 100 and worksheet_count > 5:
            summary = f"""Looking at "{file_name}", this appears to be a sophisticated financial model or comprehensive business analysis system. The workbook is quite complex with {worksheet_count} interconnected worksheets containing {formula_count} formulas, suggesting it's used for {purpose_analysis['primary_purpose']}.

{business_intelligence['narrative']} The level of complexity indicates this is likely a critical business tool used by {purpose_analysis['likely_users']} for {purpose_analysis['use_cases']}.

{self._generate_risk_narrative(risks)} This spreadsheet represents {business_intelligence['business_value']} and appears to be {purpose_analysis['maintenance_level']}."""
        
        elif formula_count > 20:
            summary = f""""{file_name}" is a {purpose_analysis['document_type']} with {worksheet_count} worksheet{"s" if worksheet_count != 1 else ""} and {formula_count} formulas. This suggests it's actively used for {purpose_analysis['primary_purpose']}.

{business_intelligence['narrative']} The structure indicates it's designed for {purpose_analysis['target_audience']} who need to {purpose_analysis['main_function']}.

{self._generate_efficiency_insights(formula_count, worksheet_count)} Overall, this appears to be {purpose_analysis['criticality_assessment']} that {business_intelligence['operational_role']}."""
        
        elif worksheet_count > 3:
            summary = f""""{file_name}" is organized as a multi-sheet workbook with {worksheet_count} sections, containing {formula_count} calculation{"s" if formula_count != 1 else ""}. This structure suggests it's used for {purpose_analysis['organizational_purpose']}.

{business_intelligence['data_insights']} The layout indicates this workbook serves as {purpose_analysis['functional_role']} for {purpose_analysis['department_focus']}.

This type of organization is typical of {purpose_analysis['industry_pattern']} and appears to be {purpose_analysis['usage_frequency']}."""
        
        else:
            summary = f""""{file_name}" is a focused spreadsheet with {formula_count} calculation{"s" if formula_count != 1 else ""} {"across " + str(worksheet_count) + " sheet" + ("s" if worksheet_count != 1 else "") if worksheet_count > 1 else ""}. {purpose_analysis['simple_analysis']}.

{business_intelligence['basic_insights']} This appears to be {purpose_analysis['utility_type']} that {purpose_analysis['practical_use']}.

The straightforward structure suggests it's designed for {purpose_analysis['accessibility']} and {purpose_analysis['maintenance_ease']}."""
        
        return summary.strip()
    
    def _analyze_excel_purpose(self, worksheets, formula_count, insights, themes):
        """Analyze the purpose and use case of an Excel document"""
        
        # Extract worksheet names for context
        sheet_names = [sheet.name.lower() if hasattr(sheet, 'name') else f'sheet{i+1}' 
                      for i, sheet in enumerate(worksheets)] if worksheets else ['sheet1']
        
        # Purpose detection based on sheet names and content
        purpose_indicators = {
            'financial': ['budget', 'finance', 'revenue', 'cost', 'profit', 'income', 'expense', 'cash', 'forecast'],
            'project': ['project', 'task', 'timeline', 'gantt', 'milestone', 'tracker', 'status'],
            'inventory': ['inventory', 'stock', 'products', 'items', 'warehouse', 'supply'],
            'hr': ['employee', 'staff', 'payroll', 'hr', 'human', 'performance', 'attendance'],
            'sales': ['sales', 'customer', 'lead', 'crm', 'pipeline', 'target', 'quota'],
            'reporting': ['report', 'dashboard', 'metrics', 'kpi', 'analysis', 'summary'],
            'planning': ['plan', 'strategy', 'goal', 'objective', 'roadmap', 'schedule']
        }
        
        detected_purposes = []
        for purpose, keywords in purpose_indicators.items():
            if any(keyword in ' '.join(sheet_names) for keyword in keywords):
                detected_purposes.append(purpose)
        
        # Complexity-based analysis
        if formula_count > 100:
            complexity_level = "enterprise-grade"
            likely_users = "finance teams, analysts, or executives"
            maintenance_level = "professionally maintained with regular updates"
        elif formula_count > 50:
            complexity_level = "advanced"
            likely_users = "business analysts or department managers"  
            maintenance_level = "regularly maintained by power users"
        elif formula_count > 10:
            complexity_level = "intermediate"
            likely_users = "team leads or specialized staff"
            maintenance_level = "periodically updated as needed"
        else:
            complexity_level = "basic"
            likely_users = "general office workers"
            maintenance_level = "simple and easy to maintain"
        
        # Primary purpose determination
        if 'financial' in detected_purposes:
            primary_purpose = "financial planning, budgeting, or economic analysis"
            document_type = "financial planning workbook"
        elif 'project' in detected_purposes:
            primary_purpose = "project management and progress tracking"
            document_type = "project tracking system"
        elif 'sales' in detected_purposes:
            primary_purpose = "sales performance monitoring and customer relationship management"
            document_type = "sales management tool"
        elif 'reporting' in detected_purposes:
            primary_purpose = "business reporting and performance analytics"
            document_type = "reporting dashboard"
        elif len(detected_purposes) > 1:
            primary_purpose = f"integrated business operations covering {', '.join(detected_purposes[:2])}"
            document_type = "multi-purpose business workbook"
        else:
            primary_purpose = "data organization and basic calculations"
            document_type = "general spreadsheet tool"
        
        return {
            'primary_purpose': primary_purpose,
            'document_type': document_type,
            'likely_users': likely_users,
            'use_cases': f"{primary_purpose} with {complexity_level} complexity",
            'maintenance_level': maintenance_level,
            'target_audience': likely_users.split(' or ')[0] if ' or ' in likely_users else likely_users,
            'main_function': primary_purpose.split(' and ')[0] if ' and ' in primary_purpose else primary_purpose,
            'criticality_assessment': f"a {complexity_level}-level business tool",
            'organizational_purpose': primary_purpose,
            'functional_role': f"a {document_type.replace('workbook', '').replace('tool', '').replace('system', '').strip()} resource",
            'department_focus': detected_purposes[0] if detected_purposes else 'general business',
            'industry_pattern': f"{detected_purposes[0] if detected_purposes else 'business'} workflows",
            'usage_frequency': 'regularly used' if formula_count > 10 else 'occasionally referenced',
            'simple_analysis': f"This represents a {complexity_level} tool for {primary_purpose}",
            'utility_type': f"a practical {document_type}",
            'practical_use': f"supports {primary_purpose}",
            'accessibility': 'easy use' if formula_count < 20 else 'specialized knowledge',
            'maintenance_ease': 'straightforward maintenance' if formula_count < 20 else 'requires expertise'
        }
    
    def _extract_business_intelligence(self, worksheets, formula_analysis, insights):
        """Extract business intelligence from Excel analysis"""
        
        # Analyze business value
        if formula_analysis.get('total_formulas', 0) > 50:
            business_value = "significant business value as a critical analytical tool"
            operational_role = "plays a key role in business decision-making processes"
            narrative = "The extensive use of formulas suggests sophisticated business logic for automated calculations and scenario analysis."
        elif formula_analysis.get('total_formulas', 0) > 10:
            business_value = "meaningful business utility for operational tasks"
            operational_role = "supports important business functions"
            narrative = "The calculation complexity indicates this workbook automates key business processes and reduces manual effort."
        else:
            business_value = "basic business utility for simple tasks"
            operational_role = "serves as a simple tool for data organization"
            narrative = "The straightforward design focuses on data storage and basic calculations rather than complex analysis."
        
        # Data insights based on structure
        if len(worksheets) > 5:
            data_insights = "The multi-sheet organization suggests comprehensive data management with clear separation of concerns across different business areas."
        elif len(worksheets) > 2:
            data_insights = "The structured approach with multiple sheets indicates organized data flow and logical business process separation."
        else:
            data_insights = "The single-sheet design prioritizes simplicity and ease of use over complex data organization."
        
        # Basic insights for simple documents
        basic_insights = narrative.replace('sophisticated business logic', 'practical functionality').replace('extensive use', 'use')
        
        return {
            'business_value': business_value,
            'operational_role': operational_role,
            'narrative': narrative,
            'data_insights': data_insights,
            'basic_insights': basic_insights
        }
    
    def _generate_risk_narrative(self, risks):
        """Generate narrative about risks"""
        if not risks:
            return "No significant risks were identified in the current analysis."
        elif len(risks) > 3:
            return f"Analysis identified {len(risks)} potential risk areas that should be reviewed for business continuity and data integrity."
        else:
            return f"There are {len(risks)} areas flagged for attention, though they appear manageable with proper oversight."
    
    def _generate_efficiency_insights(self, formula_count, worksheet_count):
        """Generate insights about efficiency and optimization"""
        if formula_count > 50 and worksheet_count > 3:
            return "The complexity suggests opportunities for automation and process optimization could yield significant time savings."
        elif formula_count > 20:
            return "The structured calculations indicate this workbook already provides good efficiency gains over manual processes."
        else:
            return "The straightforward design prioritizes usability and maintenance simplicity."

    def _generate_word_summary(self, word_analysis, intelligent_analysis):
        """Generate intelligent, LLM-like Word document summary"""
        
        # Safe data extraction
        file_name = getattr(word_analysis, 'filename', 'Unknown document')
        word_count = getattr(word_analysis, 'total_words', 0)
        page_count = getattr(word_analysis, 'page_count', 1)
        headings = getattr(word_analysis, 'headings', [])
        if isinstance(headings, bool):
            headings = []
        headings_count = len(headings) if headings else 0
        tables_count = getattr(word_analysis, 'tables_count', 0)
        
        themes = getattr(intelligent_analysis, 'themes', [])
        if isinstance(themes, bool):
            themes = []
        insights = getattr(intelligent_analysis, 'key_insights', [])
        risks = getattr(intelligent_analysis, 'risk_indicators', [])
        
        # Intelligent document analysis
        doc_purpose = self._analyze_word_purpose(headings, word_count, themes, tables_count)
        content_intelligence = self._extract_content_intelligence(word_count, headings_count, themes, insights)
        
        # Generate human-like narrative based on document characteristics
        if word_count > 5000 and headings_count > 10:
            summary = f""""{file_name}" is a comprehensive professional document that demonstrates significant depth and structured thinking. With {word_count:,} words across {page_count} pages and {headings_count} organized sections, this represents {doc_purpose['document_significance']}.

{content_intelligence['professional_assessment']} The substantial length and detailed organization suggest this is {doc_purpose['intended_use']} designed for {doc_purpose['target_readers']}.

{self._analyze_document_sophistication(headings_count, tables_count, themes)} This level of detail indicates the document serves as {doc_purpose['business_function']} and likely requires {content_intelligence['expertise_level']} to fully utilize."""

        elif word_count > 2000 and headings_count > 5:
            summary = f""""{file_name}" is a well-structured business document spanning {page_count} pages with {word_count:,} words. The {headings_count} headings indicate {doc_purpose['organizational_approach']} and suggest this is used for {doc_purpose['primary_function']}.

{content_intelligence['structural_analysis']} The document appears to be {doc_purpose['formality_level']} that {doc_purpose['practical_application']}.

{self._assess_document_utility(word_count, themes)} Overall, this represents {doc_purpose['business_value']} that would be most valuable to {doc_purpose['key_stakeholders']}."""

        elif word_count > 500:
            summary = f""""{file_name}" is a focused document with {word_count:,} words {"and " + str(headings_count) + " section" + ("s" if headings_count != 1 else "") if headings_count > 0 else ""}. {doc_purpose['concise_assessment']}.

{content_intelligence['content_evaluation']} The structure suggests this document is {doc_purpose['usability_focus']} and designed for {doc_purpose['accessibility_level']}.

{self._evaluate_document_efficiency(word_count, page_count)} This appears to be {doc_purpose['document_category']} that {doc_purpose['operational_purpose']}."""

        else:
            summary = f""""{file_name}" is a brief document with {word_count} words. {doc_purpose['brevity_analysis']}.

{content_intelligence['simple_evaluation']} This type of document typically serves as {doc_purpose['simple_function']} for {doc_purpose['quick_reference_use']}.

The concise format suggests it's designed for {doc_purpose['efficiency_focus']} and {doc_purpose['ease_of_use']}."""
        
        return summary.strip()
    
    def _analyze_word_purpose(self, headings, word_count, themes, tables_count):
        """Analyze the purpose and business context of a Word document"""
        
        # Extract heading content for analysis
        heading_text = ' '.join([h.get('text', '').lower() for h in headings if isinstance(h, dict)])
        
        # Purpose indicators
        purpose_patterns = {
            'report': ['report', 'analysis', 'findings', 'summary', 'results', 'conclusion'],
            'proposal': ['proposal', 'recommendation', 'suggestion', 'plan', 'strategy', 'approach'],
            'policy': ['policy', 'procedure', 'guidelines', 'standards', 'requirements', 'compliance'],
            'manual': ['manual', 'guide', 'instructions', 'tutorial', 'how-to', 'process'],
            'contract': ['agreement', 'contract', 'terms', 'conditions', 'legal', 'binding'],
            'presentation': ['presentation', 'overview', 'introduction', 'executive', 'brief']
        }
        
        detected_type = 'general business document'
        for doc_type, keywords in purpose_patterns.items():
            if any(keyword in heading_text for keyword in keywords):
                detected_type = f"{doc_type} document"
                break
        
        # Complexity and significance assessment
        if word_count > 5000:
            document_significance = "a substantial piece of business documentation requiring significant investment of time and expertise"
            intended_use = "a comprehensive resource for strategic decision-making or detailed reference"
            business_function = "a critical business reference or decision-making tool"
            formality_level = "a formal, professional document"
        elif word_count > 2000:
            document_significance = "a significant business document with considerable detail"
            intended_use = "an important reference for business operations or planning"
            business_function = "a key business communication or planning document" 
            formality_level = "a structured business document"
        else:
            document_significance = "a focused communication piece"
            intended_use = "practical business communication"
            business_function = "a straightforward business tool"
            formality_level = "an accessible business document"
        
        # Target audience based on complexity
        if word_count > 3000 and len(headings) > 8:
            target_readers = "executives, specialists, or stakeholders requiring detailed information"
            key_stakeholders = "senior management or subject matter experts"
            expertise_level = "specialized knowledge or significant time investment"
        elif word_count > 1000:
            target_readers = "business professionals or team members"
            key_stakeholders = "department managers or project teams"
            expertise_level = "business familiarity and moderate attention"
        else:
            target_readers = "general business audience"
            key_stakeholders = "team members or general staff"
            expertise_level = "basic business understanding"
        
        return {
            'document_significance': document_significance,
            'intended_use': intended_use,
            'target_readers': target_readers,
            'business_function': business_function,
            'formality_level': formality_level,
            'key_stakeholders': key_stakeholders,
            'organizational_approach': f"careful organization and structured presentation",
            'primary_function': f"business communication and information management",
            'practical_application': f"serves as a {detected_type} for business purposes",
            'business_value': f"a valuable {detected_type}",
            'concise_assessment': f"This represents a {detected_type.replace('document', '').strip()} with focused content",
            'usability_focus': "designed for practical business use",
            'accessibility_level': "straightforward access and understanding",
            'document_category': f"a practical {detected_type}",
            'operational_purpose': "supports specific business needs",
            'brevity_analysis': f"This appears to be a concise {detected_type.replace('general business document', 'communication')}",
            'simple_function': f"quick reference or brief communication",
            'quick_reference_use': "immediate business needs",
            'efficiency_focus': "quick consumption",
            'ease_of_use': "minimal time investment"
        }
    
    def _extract_content_intelligence(self, word_count, headings_count, themes, insights):
        """Extract intelligence about document content and structure"""
        
        if word_count > 3000:
            professional_assessment = "The extensive content demonstrates thorough research and comprehensive coverage of the subject matter."
            expertise_level = "significant expertise or substantial time"
        elif word_count > 1000:
            professional_assessment = "The content depth indicates solid preparation and structured thinking."
            expertise_level = "moderate expertise and focused attention"
        else:
            professional_assessment = "The concise approach suggests efficient communication and clear priorities."
            expertise_level = "basic familiarity"
        
        if headings_count > 8:
            structural_analysis = "The detailed organization with multiple sections indicates comprehensive planning and systematic approach to the subject."
        elif headings_count > 3:
            structural_analysis = "The structured layout suggests organized thinking and clear communication objectives."
        else:
            structural_analysis = "The straightforward structure prioritizes clarity and ease of reading."
        
        # Content evaluation
        if themes and len(themes) > 2:
            content_evaluation = f"The document covers multiple business areas including {', '.join(themes[:3])}, indicating broad scope and integrated thinking."
        elif themes and len(themes) > 0:
            content_evaluation = f"The content focuses on {themes[0]} with clear business relevance and practical applications."
        else:
            content_evaluation = "The content appears focused on practical business communication."
        
        simple_evaluation = "The brief format suggests targeted communication for specific business needs."
        
        return {
            'professional_assessment': professional_assessment,
            'expertise_level': expertise_level,
            'structural_analysis': structural_analysis,
            'content_evaluation': content_evaluation,
            'simple_evaluation': simple_evaluation
        }
    
    def _analyze_document_sophistication(self, headings_count, tables_count, themes):
        """Analyze document sophistication level"""
        sophistication_indicators = []
        
        if headings_count > 10:
            sophistication_indicators.append("highly organized structure")
        if tables_count > 2:
            sophistication_indicators.append("data-driven content")
        if len(themes) > 3:
            sophistication_indicators.append("multi-faceted business analysis")
        
        if sophistication_indicators:
            return f"The document demonstrates {', '.join(sophistication_indicators)}, indicating professional preparation and strategic thinking."
        else:
            return "The document shows clear organization and professional presentation."
    
    def _assess_document_utility(self, word_count, themes):
        """Assess the practical utility of the document"""
        if word_count > 2000 and themes:
            return f"Given its scope and {len(themes)} thematic areas, this document provides comprehensive value for business planning and decision-making."
        elif word_count > 1000:
            return "The substantial content provides good depth for business reference and operational guidance."
        else:
            return "The focused content delivers targeted value for specific business needs."
    
    def _evaluate_document_efficiency(self, word_count, page_count):
        """Evaluate document efficiency and readability"""
        words_per_page = word_count / max(page_count, 1)
        
        if words_per_page > 400:
            return "The information density suggests comprehensive coverage with efficient use of space."
        elif words_per_page > 200:
            return "The balanced content density provides good information value with readable formatting."
        else:
            return "The comfortable information density prioritizes readability and ease of consumption."

def main():
    """Main application entry point"""
    app = StreamlitXDPAnalyzer()
    app.run()

if __name__ == "__main__":
    main()