"""
XDP Analyzer - Simple Streamlit Web Interface
Uses the reliable simple Excel parser for comprehensive analysis
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

from parsers.simple_excel_parser import SimpleExcelParser
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer

# Page configuration
st.set_page_config(
    page_title="XDP Analyzer - Simple Excel Intelligence",
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

class SimpleStreamlitXDPAnalyzer:
    """Simple Streamlit interface for XDP Analyzer"""
    
    def __init__(self):
        self.excel_parser = SimpleExcelParser()
        self.intelligent_analyzer = FreeIntelligentAnalyzer()
    
    def run(self):
        """Main Streamlit app"""
        
        # Header
        st.markdown('<h1 class="main-header">📊 XDP Analyzer - Simple</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">🆓 <strong>Reliable</strong> Excel Analysis • No Complex Dependencies • Fast & Accurate</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("🚀 Getting Started")
            st.markdown("""
            **Supported Files:**
            - 📊 Excel: .xlsx, .xlsm, .xls, .xlsb
            
            **Features:**
            - ✅ Multi-sheet Analysis
            - ✅ Data Type Detection  
            - ✅ Statistical Summary
            - ✅ Formula Detection
            - ✅ Data Quality Scoring
            - ✅ Intelligent Insights
            """)
            
            st.header("🛡️ Reliable Analysis")
            st.success("Uses proven pandas-based parsing for accurate results!")
        
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
        
        st.header("📁 Upload Your Excel File")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file to analyze",
            type=['xlsx', 'xlsm', 'xls', 'xlsb'],
            help="Upload Excel files for comprehensive analysis"
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
            if st.button("🔍 Analyze Excel File", type="primary", use_container_width=True):
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
            status_text.text("🔄 Parsing Excel structure...")
            progress_bar.progress(25)
            
            excel_analysis = self.excel_parser.parse(tmp_path)
            st.session_state.excel_analysis = excel_analysis
            
            # Step 2: Intelligent analysis
            status_text.text("🧠 Performing intelligent analysis...")
            progress_bar.progress(50)
            
            try:
                # Adapt analysis for intelligent analyzer
                adapted_analysis = self._adapt_for_intelligent_analyzer(excel_analysis)
                intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(adapted_analysis)
                st.session_state.intelligent_analysis = intelligent_analysis
            except Exception as e:
                st.warning(f"Intelligent analysis failed: {e}")
                st.session_state.intelligent_analysis = None
            
            # Step 3: Generate insights
            status_text.text("💡 Generating insights...")
            progress_bar.progress(75)
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Display results
            self.display_analysis_results(excel_analysis, st.session_state.intelligent_analysis)
            
            # Clean up
            Path(tmp_path).unlink()
            
        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
            status_text.text("❌ Analysis failed")
    
    def _adapt_for_intelligent_analyzer(self, simple_analysis):
        """Adapt simple analysis to work with intelligent analyzer"""
        
        # Create a mock object that mimics the expected structure
        class AdaptedAnalysis:
            def __init__(self, simple_analysis):
                self.worksheets = []
                self.vba_code = {}
                self.external_references = []
                self.defined_names = []
                self.custom_properties = {}
                self.formulas_summary = {}
                self.business_logic = {}
                self.data_flow = {}
                self.complexity_metrics = {
                    'complexity_score': simple_analysis.summary.get('data_quality_score', 0)
                }
                
                # Convert sheets to expected format
                for sheet in simple_analysis.sheets:
                    worksheet = AdaptedWorksheet(sheet)
                    self.worksheets.append(worksheet)
                
                # Set formulas summary
                self.formulas_summary = {
                    'total_formulas': simple_analysis.summary.get('total_formulas', 0),
                    'function_usage': {},
                    'complexity_scores': []
                }
        
        class AdaptedWorksheet:
            def __init__(self, sheet):
                self.name = sheet.name
                self.cells = []
                self.formulas = []
                self.charts = []
                self.pivot_tables = []
                self.data_validation = []
                self.conditional_formatting = []
                self.named_ranges = []
                self.vba_references = []
                self.hidden = False
                self.protected = False
                self.dimensions = {'max_row': sheet.shape[0], 'max_column': sheet.shape[1]}
                
                # Convert sample data to cells
                for i, row_data in enumerate(sheet.sample_data):
                    for col_name, value in row_data.items():
                        if value is not None:
                            cell = AdaptedCell(f"{col_name}{i+1}", value)
                            self.cells.append(cell)
                
                # Add formula info
                if sheet.has_formulas:
                    self.formulas = [{'formula': '=FORMULA', 'functions': ['SUM'], 'complexity': {'score': 1}}]
        
        class AdaptedCell:
            def __init__(self, address, value):
                self.address = address
                self.value = value
                self.formula = None
                self.data_type = str(type(value).__name__)
                self.number_format = None
                self.comment = None
                self.hyperlink = None
                self.font_info = {}
                self.fill_info = {}
                self.is_merged = False
                self.merge_range = None
        
        return AdaptedAnalysis(simple_analysis)
    
    def display_analysis_results(self, excel_analysis, intelligent_analysis):
        """Display comprehensive analysis results"""
        
        st.success("🎉 Analysis completed successfully!")
        
        # Overview metrics
        st.subheader("📈 File Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Worksheets", 
                excel_analysis.sheet_count,
                help="Number of worksheets in the Excel file"
            )
        
        with col2:
            st.metric(
                "📏 Total Rows", 
                excel_analysis.total_rows,
                help="Total number of rows across all sheets"
            )
        
        with col3:
            st.metric(
                "📐 Total Columns", 
                excel_analysis.total_columns,
                help="Maximum number of columns in any sheet"
            )
        
        with col4:
            st.metric(
                "⭐ Data Quality", 
                f"{excel_analysis.summary['data_quality_score']}/100",
                help="Overall data quality score"
            )
        
        # Tabs for detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Sheet Details", "📊 Statistics", "💡 Insights", "📤 Export"])
        
        with tab1:
            self.display_sheet_details_tab(excel_analysis)
        
        with tab2:
            self.display_statistics_tab(excel_analysis)
        
        with tab3:
            self.display_insights_tab(intelligent_analysis)
        
        with tab4:
            self.display_export_tab(excel_analysis, intelligent_analysis)
    
    def display_sheet_details_tab(self, excel_analysis):
        """Display detailed sheet information"""
        
        st.subheader("📋 Worksheet Analysis")
        
        # Create summary table
        sheet_data = []
        for sheet in excel_analysis.sheets:
            sheet_data.append({
                'Sheet Name': sheet.name,
                'Rows': sheet.shape[0],
                'Columns': sheet.shape[1],
                'Total Cells': sheet.shape[0] * sheet.shape[1],
                'Formulas': sheet.formula_count,
                'Data Types': len(set(sheet.data_types.values())),
                'Null Cells': sum(sheet.null_counts.values())
            })
        
        df_sheets = pd.DataFrame(sheet_data)
        st.dataframe(df_sheets, use_container_width=True)
        
        # Sheet size visualization
        if len(sheet_data) > 1:
            fig = px.bar(
                df_sheets, 
                x='Sheet Name', 
                y='Total Cells',
                title="Sheet Size Comparison",
                color='Total Cells',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual sheet details
        selected_sheet = st.selectbox("Select sheet for detailed view:", excel_analysis.sheet_names)
        
        if selected_sheet:
            sheet = next(s for s in excel_analysis.sheets if s.name == selected_sheet)
            
            st.subheader(f"📄 {selected_sheet} Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Types:**")
                for col, dtype in sheet.data_types.items():
                    non_null = sheet.non_null_counts[col]
                    null = sheet.null_counts[col]
                    st.write(f"• {col}: {dtype} ({non_null} non-null, {null} null)")
            
            with col2:
                if sheet.unique_values:
                    st.markdown("**Unique Values (≤20):**")
                    for col, values in sheet.unique_values.items():
                        st.write(f"• {col}: {values}")
            
            # Sample data
            if sheet.sample_data:
                st.markdown("**Sample Data (First 3 rows):**")
                sample_df = pd.DataFrame(sheet.sample_data)
                st.dataframe(sample_df, use_container_width=True)
    
    def display_statistics_tab(self, excel_analysis):
        """Display statistical analysis"""
        
        st.subheader("📊 Statistical Analysis")
        
        # Data quality metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Sheet Size", f"{excel_analysis.summary['average_sheet_size']:.0f} cells")
        
        with col2:
            st.metric("Sheets with Formulas", excel_analysis.summary['sheets_with_formulas'])
        
        with col3:
            st.metric("Most Complex Sheet", excel_analysis.summary['most_complex_sheet'])
        
        # Formula analysis
        if excel_analysis.summary['total_formulas'] > 0:
            st.subheader("🧮 Formula Analysis")
            
            formula_data = []
            for sheet in excel_analysis.sheets:
                if sheet.formula_count > 0:
                    formula_data.append({
                        'Sheet': sheet.name,
                        'Formulas': sheet.formula_count,
                        'Formula Density': sheet.formula_count / (sheet.shape[0] * sheet.shape[1]) * 100
                    })
            
            if formula_data:
                df_formulas = pd.DataFrame(formula_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(df_formulas, x='Sheet', y='Formulas', title="Formulas per Sheet")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df_formulas, x='Sheet', y='Formula Density', title="Formula Density (%)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data type distribution
        st.subheader("📊 Data Type Distribution")
        
        all_data_types = {}
        for sheet in excel_analysis.sheets:
            for col, dtype in sheet.data_types.items():
                all_data_types[dtype] = all_data_types.get(dtype, 0) + 1
        
        if all_data_types:
            fig = px.pie(
                values=list(all_data_types.values()),
                names=list(all_data_types.keys()),
                title="Data Type Distribution Across All Sheets"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_insights_tab(self, intelligent_analysis):
        """Display intelligent analysis insights"""
        
        if intelligent_analysis is None:
            st.info("Intelligent analysis not available. Basic analysis completed successfully.")
            return
        
        st.subheader("💡 Intelligent Insights")
        
        # Summary
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Executive Summary")
        st.write(intelligent_analysis.summary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        for i, insight in enumerate(intelligent_analysis.key_insights, 1):
            st.markdown(f"""
            <div class="insight-box">
                <strong>💡 Insight {i}:</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Business logic
        st.markdown("### 🎯 Business Logic")
        st.write(intelligent_analysis.business_logic)
        
        # Recommendations
        st.markdown("### 🛠️ Recommendations")
        for i, recommendation in enumerate(intelligent_analysis.recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-box">
                <strong>✅ Recommendation {i}:</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        # Risk indicators
        if intelligent_analysis.risk_indicators:
            st.markdown("### ⚠️ Risk Indicators")
            for i, risk in enumerate(intelligent_analysis.risk_indicators, 1):
                st.markdown(f"""
                <div class="risk-box">
                    <strong>⚠️ Risk {i}:</strong> {risk}
                </div>
                """, unsafe_allow_html=True)
        
        # Automation opportunities
        if intelligent_analysis.automation_opportunities:
            st.markdown("### 🤖 Automation Opportunities")
            for i, opportunity in enumerate(intelligent_analysis.automation_opportunities, 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>🚀 Opportunity {i}:</strong> {opportunity}
                </div>
                """, unsafe_allow_html=True)
    
    def display_export_tab(self, excel_analysis, intelligent_analysis):
        """Display export options"""
        
        st.subheader("📤 Export Analysis Results")
        
        # Create comprehensive report
        report = {
            'file_info': {
                'filename': excel_analysis.filename,
                'file_type': excel_analysis.file_type,
                'sheet_count': excel_analysis.sheet_count,
                'total_rows': excel_analysis.total_rows,
                'total_columns': excel_analysis.total_columns,
                'analysis_timestamp': excel_analysis.analysis_timestamp
            },
            'summary': excel_analysis.summary,
            'sheets': [
                {
                    'name': sheet.name,
                    'shape': sheet.shape,
                    'columns': sheet.columns,
                    'data_types': sheet.data_types,
                    'null_counts': sheet.null_counts,
                    'non_null_counts': sheet.non_null_counts,
                    'formula_count': sheet.formula_count,
                    'has_formulas': sheet.has_formulas
                }
                for sheet in excel_analysis.sheets
            ]
        }
        
        if intelligent_analysis:
            report['intelligent_analysis'] = {
                'summary': intelligent_analysis.summary,
                'key_insights': intelligent_analysis.key_insights,
                'business_logic': intelligent_analysis.business_logic,
                'recommendations': intelligent_analysis.recommendations,
                'risk_indicators': intelligent_analysis.risk_indicators,
                'automation_opportunities': intelligent_analysis.automation_opportunities,
                'confidence_score': intelligent_analysis.confidence_score
            }
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Full Report (JSON)",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"{excel_analysis.filename}_analysis_report.json",
                mime="application/json"
            )
        
        with col2:
            # Create CSV summary
            csv_data = []
            for sheet in excel_analysis.sheets:
                csv_data.append({
                    'Sheet Name': sheet.name,
                    'Rows': sheet.shape[0],
                    'Columns': sheet.shape[1],
                    'Formulas': sheet.formula_count,
                    'Null Cells': sum(sheet.null_counts.values()),
                    'Data Quality': excel_analysis.summary['data_quality_score']
                })
            
            csv_df = pd.DataFrame(csv_data)
            csv_string = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Summary (CSV)",
                data=csv_string,
                file_name=f"{excel_analysis.filename}_summary.csv",
                mime="text/csv"
            )
    
    def dashboard_tab(self):
        """Dashboard overview tab"""
        
        st.header("📊 Analysis Dashboard")
        
        if 'excel_analysis' not in st.session_state:
            st.info("👆 Upload and analyze an Excel file in the first tab to see the dashboard!")
            return
        
        # Display cached analysis
        excel_analysis = st.session_state.excel_analysis
        intelligent_analysis = st.session_state.get('intelligent_analysis')
        
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Worksheets", excel_analysis.sheet_count)
        
        with col2:
            st.metric("📏 Total Rows", excel_analysis.total_rows)
        
        with col3:
            st.metric("⭐ Data Quality", f"{excel_analysis.summary['data_quality_score']}/100")
        
        # Quick insights
        if intelligent_analysis:
            st.subheader("⚡ Quick Insights")
            for insight in intelligent_analysis.key_insights[:3]:
                st.info(f"💡 {insight}")
        else:
            st.info("💡 Basic analysis completed. Upload file for intelligent insights.")
    
    def about_tab(self):
        """About and help tab"""
        
        st.header("ℹ️ About XDP Analyzer - Simple")
        
        st.markdown("""
        ### 🎯 What is XDP Analyzer - Simple?
        
        This is a **reliable, simplified version** of XDP Analyzer that uses proven pandas-based 
        Excel parsing for accurate and fast analysis of Excel files.
        
        ### ✨ Key Features
        
        - 🚀 **Fast & Reliable** - Uses proven pandas ExcelFile for accurate parsing
        - 📊 **Multi-sheet Support** - Analyzes all worksheets in your Excel file
        - 🔍 **Comprehensive Analysis** - Data types, statistics, formulas, quality scores
        - 💡 **Intelligent Insights** - Business logic detection and recommendations
        - 📤 **Export Options** - Download detailed reports in JSON or CSV format
        - 🆓 **100% Free** - No external APIs, no costs, complete privacy
        
        ### 🛠️ Supported File Types
        
        - **Excel**: .xlsx, .xlsm, .xls, .xlsb
        
        ### 🚀 How It Works
        
        1. **Upload** your Excel file using the file uploader
        2. **Parse** - Our reliable parser examines structure and content
        3. **Analyze** - Get comprehensive statistics and insights
        4. **Export** - Download detailed analysis reports
        
        ### 🔧 Technology Stack
        
        - **Frontend**: Streamlit (Free)
        - **Excel Parsing**: pandas + openpyxl (Proven & Reliable)
        - **Analysis**: Custom algorithms + NLTK (Free)
        - **Visualization**: Plotly, Matplotlib (Free)
        
        ### 📞 Support
        
        This simplified version focuses on reliability and accuracy over complex features.
        For questions or issues, check the analysis results and export options.
        
        ---
        
        **Built with ❤️ for reliable Excel analysis**
        """)

def main():
    """Main application entry point"""
    app = SimpleStreamlitXDPAnalyzer()
    app.run()

if __name__ == "__main__":
    main()
