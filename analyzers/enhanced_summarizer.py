"""
Enhanced Document Summarizer with Free LLM Integration
Provides detailed, intuitive, non-technical summaries using free/open LLM APIs
"""

import json
import logging
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import requests
import time
from datetime import datetime

# For local fallback summarization
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, automated_readability_index

@dataclass
class EnhancedSummary:
    """Enhanced document summary with business context"""
    executive_summary: str
    what_this_document_does: str
    key_business_purpose: str
    main_findings: List[str]
    business_implications: List[str]
    target_audience_insights: str
    document_complexity_plain_english: str
    actionable_insights: List[str]
    potential_concerns: List[str]
    next_steps_suggestions: List[str]
    confidence_score: float
    generation_method: str  # 'llm', 'hybrid', 'local'

class EnhancedDocumentSummarizer:
    """
    Enhanced document summarizer using free LLMs with intelligent fallbacks
    """
    
    def __init__(self, use_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize NLTK components for fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        # Free LLM options (in order of preference)
        self.llm_options = [
            {
                'name': 'huggingface_inference',
                'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
                'headers': {},  # No API key needed for basic usage
                'max_tokens': 1000
            },
            {
                'name': 'ollama_local',
                'url': 'http://localhost:11434/api/generate',
                'model': 'llama3.1:8b',  # If user has Ollama installed
                'max_tokens': 2000
            }
        ]
        
        # Business context templates
        self.business_context_patterns = {
            'financial_modeling': {
                'keywords': ['var', 'value at risk', 'portfolio', 'returns', 'volatility', 'correlation', 'monte carlo', 'simulation'],
                'purpose': 'financial risk assessment and portfolio management',
                'audience': 'risk managers, portfolio managers, and senior executives',
                'insights_focus': ['risk levels', 'potential losses', 'diversification', 'regulatory compliance']
            },
            'budgeting_planning': {
                'keywords': ['budget', 'forecast', 'revenue', 'expenses', 'profit', 'margin', 'cash flow'],
                'purpose': 'financial planning and budget management',
                'audience': 'finance teams, department heads, and executives',
                'insights_focus': ['spending patterns', 'revenue projections', 'cost optimization', 'resource allocation']
            },
            'data_analysis': {
                'keywords': ['analysis', 'metrics', 'kpi', 'dashboard', 'trends', 'correlation', 'regression'],
                'purpose': 'business intelligence and data-driven decision making',
                'audience': 'analysts, managers, and decision makers',
                'insights_focus': ['performance trends', 'key drivers', 'opportunities', 'data quality']
            },
            'operations_tracking': {
                'keywords': ['inventory', 'production', 'efficiency', 'utilization', 'capacity', 'throughput'],
                'purpose': 'operational performance monitoring and optimization',
                'audience': 'operations managers, production teams, and executives',
                'insights_focus': ['operational efficiency', 'bottlenecks', 'capacity planning', 'process improvements']
            },
            'sales_performance': {
                'keywords': ['sales', 'revenue', 'customers', 'pipeline', 'conversion', 'territory', 'quota'],
                'purpose': 'sales performance tracking and revenue management',
                'audience': 'sales teams, sales managers, and revenue executives',
                'insights_focus': ['sales trends', 'customer behavior', 'territory performance', 'pipeline health']
            }
        }
    
    async def generate_enhanced_summary(self, excel_analysis, formula_analysis) -> EnhancedSummary:
        """Generate comprehensive business-focused summary"""
        
        # Extract comprehensive context
        document_context = self._extract_comprehensive_context(excel_analysis, formula_analysis)
        
        # Determine business domain and purpose
        business_context = self._identify_business_context(document_context)
        
        # Try LLM summarization first, fallback to enhanced local if needed
        if self.use_llm:
            try:
                llm_summary = await self._generate_llm_summary(document_context, business_context)
                if llm_summary:
                    return llm_summary
            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {e}, falling back to enhanced local")
        
        # Enhanced local summarization
        return self._generate_enhanced_local_summary(document_context, business_context)
    
    def _extract_comprehensive_context(self, excel_analysis, formula_analysis) -> Dict[str, Any]:
        """Extract comprehensive context from Excel analysis"""
        
        context = {
            'basic_info': {
                'filename': excel_analysis.filename,
                'worksheets': len(excel_analysis.worksheets),
                'total_formulas': formula_analysis.get('total_formulas', 0),
                'complexity_score': formula_analysis.get('average_complexity', 0)
            },
            'content_analysis': {},
            'formula_patterns': {},
            'business_indicators': {},
            'data_structure': {},
            'text_content': []
        }
        
        # Extract worksheet names and their content
        worksheet_info = []
        all_text_content = []
        
        for ws in excel_analysis.worksheets:
            ws_info = {
                'name': ws.name,
                'cell_count': len(ws.cells),
                'formula_count': len(ws.formulas),
                'has_charts': len(ws.charts) > 0,
                'has_pivot_tables': len(ws.pivot_tables) > 0
            }
            worksheet_info.append(ws_info)
            
            # Extract text content from cells
            text_values = []
            for cell in ws.cells:
                if isinstance(cell.value, str) and len(cell.value.strip()) > 2:
                    text_values.append(cell.value.strip())
                    all_text_content.append(cell.value.strip())
                
                # Extract comments
                if cell.comment:
                    all_text_content.append(cell.comment)
            
            ws_info['sample_text'] = text_values[:10]  # First 10 text values
        
        context['worksheets_detail'] = worksheet_info
        context['text_content'] = all_text_content
        
        # Analyze formula functions and patterns
        if 'function_usage' in formula_analysis:
            context['formula_patterns'] = {
                'top_functions': list(formula_analysis['function_usage'].keys())[:10],
                'function_counts': formula_analysis['function_usage'],
                'business_categories': formula_analysis.get('business_categories', {})
            }
        
        # Extract business indicators from content
        context['business_indicators'] = self._extract_business_indicators(all_text_content)
        
        return context
    
    def _extract_business_indicators(self, text_content: List[str]) -> Dict[str, Any]:
        """Extract business-relevant indicators from text content"""
        
        all_text = ' '.join(text_content).lower()
        
        indicators = {
            'financial_terms': [],
            'business_processes': [],
            'metrics_mentioned': [],
            'time_periods': [],
            'departments': [],
            'document_type_clues': []
        }
        
        # Financial terms
        financial_patterns = [
            r'\b(revenue|profit|loss|cost|expense|budget|forecast|roi|margin|ebitda|cash\s+flow)\b',
            r'\b(portfolio|investment|return|yield|dividend|interest|principal)\b',
            r'\b(var|value\s+at\s+risk|volatility|correlation|beta|alpha|sharpe)\b'
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, all_text)
            indicators['financial_terms'].extend(matches)
        
        # Business processes
        process_patterns = [
            r'\b(analysis|planning|forecasting|modeling|simulation|optimization)\b',
            r'\b(tracking|monitoring|reporting|dashboard|scorecard)\b',
            r'\b(inventory|production|sales|marketing|operations|hr)\b'
        ]
        
        for pattern in process_patterns:
            matches = re.findall(pattern, all_text)
            indicators['business_processes'].extend(matches)
        
        # Time periods
        time_patterns = [
            r'\b(daily|weekly|monthly|quarterly|yearly|annual)\b',
            r'\b(q[1-4]|fy\d+|20\d{2})\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, all_text)
            indicators['time_periods'].extend(matches)
        
        # Clean and deduplicate
        for key in indicators:
            indicators[key] = list(set(indicators[key]))[:10]  # Top 10 unique items
        
        return indicators
    
    def _identify_business_context(self, document_context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify business context and purpose"""
        
        # Combine all text for analysis
        all_text = ' '.join(document_context['text_content']).lower()
        filename = document_context['basic_info']['filename'].lower()
        
        # Score against business context patterns
        context_scores = {}
        for context_type, patterns in self.business_context_patterns.items():
            score = 0
            
            # Check keywords in content
            for keyword in patterns['keywords']:
                if keyword in all_text:
                    score += 2
                if keyword in filename:
                    score += 3
            
            # Check in business indicators
            indicators = document_context.get('business_indicators', {})
            for indicator_list in indicators.values():
                for indicator in indicator_list:
                    if any(keyword in indicator for keyword in patterns['keywords']):
                        score += 1
            
            context_scores[context_type] = score
        
        # Get best matching context
        if context_scores:
            best_context = max(context_scores.items(), key=lambda x: x[1])[0]
            context_info = self.business_context_patterns[best_context].copy()
            context_info['confidence'] = context_scores[best_context] / 10  # Normalize
        else:
            best_context = 'general_analysis'
            context_info = {
                'purpose': 'general data analysis and business intelligence',
                'audience': 'business stakeholders and decision makers',
                'insights_focus': ['data patterns', 'business metrics', 'performance indicators'],
                'confidence': 0.3
            }
        
        context_info['type'] = best_context
        context_info['all_scores'] = context_scores
        
        return context_info
    
    async def _generate_llm_summary(self, document_context: Dict[str, Any], business_context: Dict[str, Any]) -> Optional[EnhancedSummary]:
        """Generate summary using free LLM APIs"""
        
        # Prepare context for LLM
        context_summary = self._prepare_llm_context(document_context, business_context)
        
        # Try each LLM option
        for llm_config in self.llm_options:
            try:
                if llm_config['name'] == 'huggingface_inference':
                    result = await self._query_huggingface(context_summary, llm_config)
                elif llm_config['name'] == 'ollama_local':
                    result = await self._query_ollama(context_summary, llm_config)
                else:
                    continue
                
                if result:
                    return self._parse_llm_response(result, business_context, 'llm')
                    
            except Exception as e:
                self.logger.warning(f"LLM {llm_config['name']} failed: {e}")
                continue
        
        return None
    
    def _prepare_llm_context(self, document_context: Dict[str, Any], business_context: Dict[str, Any]) -> str:
        """Prepare context for LLM analysis"""
        
        context_parts = [
            f"Document Analysis Context:",
            f"Filename: {document_context['basic_info']['filename']}",
            f"Worksheets: {document_context['basic_info']['worksheets']}",
            f"Total Formulas: {document_context['basic_info']['total_formulas']}",
            f"Business Context: {business_context['purpose']}",
            f"Target Audience: {business_context['audience']}"
        ]
        
        # Add worksheet details
        if document_context.get('worksheets_detail'):
            context_parts.append("Worksheet Details:")
            for ws in document_context['worksheets_detail'][:5]:  # Top 5 worksheets
                context_parts.append(f"- {ws['name']}: {ws['cell_count']} cells, {ws['formula_count']} formulas")
        
        # Add business indicators
        if document_context.get('business_indicators'):
            indicators = document_context['business_indicators']
            if indicators.get('financial_terms'):
                context_parts.append(f"Financial Terms Found: {', '.join(indicators['financial_terms'][:5])}")
            if indicators.get('business_processes'):
                context_parts.append(f"Business Processes: {', '.join(indicators['business_processes'][:5])}")
        
        # Add formula patterns
        if document_context.get('formula_patterns', {}).get('top_functions'):
            top_functions = document_context['formula_patterns']['top_functions'][:5]
            context_parts.append(f"Main Excel Functions Used: {', '.join(top_functions)}")
        
        # Add sample text content
        if document_context.get('text_content'):
            sample_text = ' '.join(document_context['text_content'][:20])  # First 20 text items
            if len(sample_text) > 500:
                sample_text = sample_text[:500] + "..."
            context_parts.append(f"Sample Content: {sample_text}")
        
        return '\n'.join(context_parts)
    
    async def _query_huggingface(self, context: str, config: Dict) -> Optional[str]:
        """Query Hugging Face Inference API (free tier)"""
        
        prompt = f"""
        Analyze this Excel document and provide a business-focused summary:

        {context}

        Please provide:
        1. Executive Summary (what this document is for)
        2. Main Business Purpose
        3. Key Findings (3-5 points)
        4. Business Implications
        5. Who should use this document
        6. Complexity level in plain English
        7. Actionable insights
        8. Potential concerns
        9. Suggested next steps
        """
        
        try:
            # Note: Using a free model that doesn't require API key
            payload = {
                "inputs": prompt[:1000],  # Limit input size for free tier
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
                headers=config['headers'],
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
            
        except Exception as e:
            self.logger.warning(f"Hugging Face API error: {e}")
        
        return None
    
    async def _query_ollama(self, context: str, config: Dict) -> Optional[str]:
        """Query local Ollama instance if available"""
        
        prompt = f"""
        You are a business analyst. Analyze this Excel document data and provide a comprehensive, non-technical summary:

        {context}

        Provide a detailed analysis including:
        - Executive summary
        - Business purpose
        - Key findings
        - Implications
        - Target audience
        - Complexity assessment
        - Actionable insights
        - Concerns
        - Next steps
        """
        
        try:
            payload = {
                "model": config.get('model', 'llama3.1:8b'),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": config.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(
                config['url'],
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            
        except Exception as e:
            self.logger.debug(f"Ollama not available: {e}")
        
        return None
    
    def _parse_llm_response(self, llm_response: str, business_context: Dict, method: str) -> EnhancedSummary:
        """Parse and structure LLM response"""
        
        # Basic parsing - in production, this would be more sophisticated
        sections = {
            'executive_summary': '',
            'business_purpose': '',
            'key_findings': [],
            'business_implications': [],
            'target_audience': '',
            'complexity_assessment': '',
            'actionable_insights': [],
            'concerns': [],
            'next_steps': []
        }
        
        # Simple section extraction
        lines = llm_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            lower_line = line.lower()
            if 'executive summary' in lower_line or 'summary' in lower_line:
                current_section = 'executive_summary'
            elif 'purpose' in lower_line or 'business purpose' in lower_line:
                current_section = 'business_purpose'
            elif 'findings' in lower_line or 'key points' in lower_line:
                current_section = 'key_findings'
            elif 'implications' in lower_line:
                current_section = 'business_implications'
            elif 'audience' in lower_line or 'users' in lower_line:
                current_section = 'target_audience'
            elif 'complexity' in lower_line:
                current_section = 'complexity_assessment'
            elif 'insights' in lower_line or 'actionable' in lower_line:
                current_section = 'actionable_insights'
            elif 'concerns' in lower_line or 'risks' in lower_line:
                current_section = 'concerns'
            elif 'next steps' in lower_line or 'recommendations' in lower_line:
                current_section = 'next_steps'
            else:
                # Add content to current section
                if current_section and line:
                    if current_section in ['key_findings', 'business_implications', 'actionable_insights', 'concerns', 'next_steps']:
                        sections[current_section].append(line)
                    else:
                        sections[current_section] += line + ' '
        
        return EnhancedSummary(
            executive_summary=sections['executive_summary'].strip() or self._generate_fallback_executive_summary(business_context),
            what_this_document_does=sections['business_purpose'].strip() or f"This document supports {business_context['purpose']}",
            key_business_purpose=business_context['purpose'],
            main_findings=sections['key_findings'][:5] or ['Analysis findings available in detailed view'],
            business_implications=sections['business_implications'][:5] or ['Business impact assessment included'],
            target_audience_insights=sections['target_audience'].strip() or business_context['audience'],
            document_complexity_plain_english=sections['complexity_assessment'].strip() or 'Moderate complexity business document',
            actionable_insights=sections['actionable_insights'][:5] or ['Actionable recommendations provided'],
            potential_concerns=sections['concerns'][:3] or ['Risk assessment included'],
            next_steps_suggestions=sections['next_steps'][:5] or ['Next steps identified in analysis'],
            confidence_score=0.8,
            generation_method=method
        )
    
    def _generate_enhanced_local_summary(self, document_context: Dict[str, Any], business_context: Dict[str, Any]) -> EnhancedSummary:
        """Generate enhanced summary using local intelligence"""
        
        basic_info = document_context['basic_info']
        business_indicators = document_context.get('business_indicators', {})
        formula_patterns = document_context.get('formula_patterns', {})
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(basic_info, business_context, business_indicators)
        
        # What this document does
        what_it_does = self._generate_document_purpose(business_context, formula_patterns, business_indicators)
        
        # Main findings
        main_findings = self._generate_main_findings(document_context, business_context)
        
        # Business implications
        business_implications = self._generate_business_implications(business_context, formula_patterns, basic_info)
        
        # Target audience insights
        audience_insights = self._generate_audience_insights(business_context, basic_info)
        
        # Complexity assessment
        complexity_assessment = self._generate_complexity_assessment(basic_info, formula_patterns)
        
        # Actionable insights
        actionable_insights = self._generate_actionable_insights(business_context, document_context)
        
        # Potential concerns
        concerns = self._generate_potential_concerns(basic_info, formula_patterns, business_context)
        
        # Next steps
        next_steps = self._generate_next_steps(business_context, basic_info)
        
        # Calculate confidence based on available data
        confidence = self._calculate_enhanced_confidence(basic_info, business_context, document_context)
        
        return EnhancedSummary(
            executive_summary=executive_summary,
            what_this_document_does=what_it_does,
            key_business_purpose=business_context['purpose'],
            main_findings=main_findings,
            business_implications=business_implications,
            target_audience_insights=audience_insights,
            document_complexity_plain_english=complexity_assessment,
            actionable_insights=actionable_insights,
            potential_concerns=concerns,
            next_steps_suggestions=next_steps,
            confidence_score=confidence,
            generation_method='enhanced_local'
        )
    
    def _generate_executive_summary(self, basic_info: Dict, business_context: Dict, indicators: Dict) -> str:
        """Generate executive summary"""
        
        filename = basic_info['filename'].replace('.xlsx', '').replace('.xls', '').replace('_', ' ').title()
        ws_count = basic_info['worksheets']
        formula_count = basic_info['total_formulas']
        
        # Build narrative based on business context
        context_type = business_context['type']
        
        if context_type == 'financial_modeling':
            if formula_count > 0:
                return f"This is a sophisticated financial analysis workbook ({filename}) containing {ws_count} worksheets with {formula_count} calculations. It appears to be designed for risk assessment and portfolio management, enabling stakeholders to understand financial exposures and make informed investment decisions. The document shows evidence of quantitative modeling with financial metrics and risk calculations."
            else:
                return f"This is a financial data workbook ({filename}) containing {ws_count} worksheets with structured financial data. It appears to be designed for risk assessment and portfolio management, providing foundational data for financial analysis and decision-making. The document serves as a data repository for financial metrics and risk-related information."
        
        elif context_type == 'budgeting_planning':
            if formula_count > 0:
                return f"{filename} is a comprehensive budgeting and planning tool with {ws_count} worksheets containing {formula_count} financial calculations. This workbook enables finance teams to track expenses, forecast revenue, and plan resource allocation. It serves as a central financial planning document for budget management and strategic financial decision-making."
            else:
                return f"{filename} is a budgeting and planning data repository with {ws_count} worksheets containing structured financial data. This workbook serves as a foundation for budget analysis, providing organized data for expense tracking, revenue planning, and resource allocation decisions."
        
        elif context_type == 'data_analysis':
            if formula_count > 0:
                return f"This analytical workbook ({filename}) contains {ws_count} data worksheets with {formula_count} analytical calculations. It's designed to provide business intelligence insights through data analysis, helping stakeholders identify trends, patterns, and performance metrics. The document supports data-driven decision making across the organization."
            else:
                return f"This data repository workbook ({filename}) contains {ws_count} worksheets with structured business data. It's designed as a foundational dataset for business intelligence analysis, providing organized information for identifying trends, patterns, and performance metrics to support data-driven decision making."
        
        else:
            # General business document
            financial_terms = len(indicators.get('financial_terms', []))
            if financial_terms > 3:
                focus = "financial analysis and business performance"
            elif indicators.get('business_processes'):
                focus = "business process analysis and operational insights"
            else:
                focus = "data analysis and business intelligence"
            
            return f"{filename} is a business analysis workbook featuring {ws_count} worksheets and {formula_count} calculations focused on {focus}. This document provides stakeholders with analytical insights and data-driven recommendations to support business decision-making and strategic planning."
    
    def _generate_document_purpose(self, business_context: Dict, formula_patterns: Dict, indicators: Dict) -> str:
        """Generate what this document does"""
        
        purpose_base = business_context['purpose']
        top_functions = formula_patterns.get('top_functions', [])
        
        capabilities = []
        
        # Analyze top functions to understand capabilities
        if 'SUM' in top_functions or 'SUMIF' in top_functions:
            capabilities.append("aggregate financial data and totals")
        if 'IF' in top_functions or 'VLOOKUP' in top_functions:
            capabilities.append("perform conditional logic and data lookups")
        if 'AVERAGE' in top_functions or 'COUNT' in top_functions:
            capabilities.append("calculate statistical summaries and metrics")
        if 'NPV' in top_functions or 'IRR' in top_functions:
            capabilities.append("perform advanced financial calculations")
        if any(func in top_functions for func in ['VAR', 'STDEV', 'CORREL']):
            capabilities.append("analyze risk and statistical relationships")
        
        if not capabilities:
            if formula_patterns.get('top_functions'):
                capabilities = ["perform business calculations and analysis"]
            else:
                capabilities = ["store and organize business data"]
        
        capabilities_text = ", ".join(capabilities[:-1]) + f" and {capabilities[-1]}" if len(capabilities) > 1 else capabilities[0]
        
        if formula_patterns.get('top_functions'):
            return f"This workbook serves as a tool for {purpose_base}. It enables users to {capabilities_text}. The document provides a structured framework for analyzing business data, generating insights, and supporting strategic decision-making processes."
        else:
            return f"This workbook serves as a data repository for {purpose_base}. It enables users to {capabilities_text} in an organized manner. The document provides a structured foundation for business data organization, serving as input for analysis tools and supporting data-driven decision-making processes."
    
    def _generate_main_findings(self, document_context: Dict, business_context: Dict) -> List[str]:
        """Generate main findings"""
        
        findings = []
        basic_info = document_context['basic_info']
        formula_patterns = document_context.get('formula_patterns', {})
        business_indicators = document_context.get('business_indicators', {})
        worksheets = document_context.get('worksheets_detail', [])
        
        # Finding about document structure
        if basic_info['worksheets'] > 5:
            findings.append(f"Complex multi-sheet structure with {basic_info['worksheets']} worksheets suggests comprehensive business modeling")
        elif basic_info['worksheets'] > 1:
            findings.append(f"Well-organized {basic_info['worksheets']}-sheet structure enables systematic analysis")
        else:
            findings.append("Focused single-sheet analysis for targeted business insights")
        
        # Finding about computational complexity
        formula_count = basic_info['total_formulas']
        if formula_count > 100:
            findings.append(f"Highly automated with {formula_count} formulas, indicating sophisticated business logic")
        elif formula_count > 20:
            findings.append(f"Moderate automation with {formula_count} calculations supporting business analysis")
        else:
            findings.append("Simple structure with focused calculations for specific business needs")
        
        # Finding about business focus
        financial_terms = business_indicators.get('financial_terms', [])
        if len(financial_terms) > 5:
            top_terms = financial_terms[:3]
            findings.append(f"Strong financial focus with emphasis on {', '.join(top_terms)}")
        
        # Finding about time dimensions
        time_periods = business_indicators.get('time_periods', [])
        if time_periods:
            findings.append(f"Time-based analysis covering {', '.join(time_periods[:3])}")
        
        # Finding about functionality
        top_functions = formula_patterns.get('top_functions', [])[:3]
        if top_functions:
            findings.append(f"Core functionality built around {', '.join(top_functions)} operations")
        
        return findings[:5]  # Return top 5 findings
    
    def _generate_business_implications(self, business_context: Dict, formula_patterns: Dict, basic_info: Dict) -> List[str]:
        """Generate business implications"""
        
        implications = []
        context_type = business_context['type']
        
        # Domain-specific implications
        if context_type == 'financial_modeling':
            implications.extend([
                "Enables quantitative risk assessment for investment decisions",
                "Supports regulatory compliance and risk reporting requirements",
                "Facilitates scenario analysis and stress testing capabilities"
            ])
        elif context_type == 'budgeting_planning':
            implications.extend([
                "Improves financial planning accuracy and accountability",
                "Enables better resource allocation and cost management",
                "Supports strategic planning and financial forecasting"
            ])
        elif context_type == 'data_analysis':
            implications.extend([
                "Enhances data-driven decision making capabilities",
                "Identifies performance trends and business opportunities",
                "Supports evidence-based strategic planning"
            ])
        else:
            implications.extend([
                "Provides structured framework for business analysis",
                "Enables consistent and repeatable analytical processes",
                "Supports informed decision making across the organization"
            ])
        
        # Complexity-based implications
        if basic_info['total_formulas'] > 50:
            implications.append("High automation reduces manual effort and human error")
        
        if basic_info['worksheets'] > 3:
            implications.append("Comprehensive structure enables multi-dimensional analysis")
        
        return implications[:5]
    
    def _generate_audience_insights(self, business_context: Dict, basic_info: Dict) -> str:
        """Generate target audience insights"""
        
        base_audience = business_context['audience']
        complexity = basic_info['complexity_score']
        
        if complexity > 15:
            skill_level = "advanced Excel users and quantitative analysts"
        elif complexity > 8:
            skill_level = "intermediate Excel users with business analysis experience"
        else:
            skill_level = "business professionals with basic Excel knowledge"
        
        return f"This document is designed for {base_audience}, specifically {skill_level}. Users should have understanding of the business domain and comfort with Excel-based analysis tools."
    
    def _generate_complexity_assessment(self, basic_info: Dict, formula_patterns: Dict) -> str:
        """Generate complexity assessment in plain English"""
        
        complexity_score = basic_info['complexity_score']
        formula_count = basic_info['total_formulas']
        
        # Handle 0-formula documents specially
        if formula_count == 0:
            level = "data-focused"
            description = "primarily contains structured data and requires basic Excel skills to navigate"
        elif complexity_score > 20:
            level = "highly sophisticated"
            description = "requires advanced Excel expertise"
        elif complexity_score > 10:
            level = "moderately complex"
            description = "requires good Excel knowledge"
        elif complexity_score > 5:
            level = "moderately simple"
            description = "accessible to most Excel users"
        else:
            level = "straightforward"
            description = "easy to understand and use"
        
        # Function complexity indicators
        advanced_functions = []
        top_functions = formula_patterns.get('top_functions', [])
        
        advanced_indicators = ['VLOOKUP', 'INDEX', 'MATCH', 'INDIRECT', 'OFFSET', 'SUMPRODUCT']
        for func in advanced_indicators:
            if func in top_functions:
                advanced_functions.append(func)
        
        complexity_context = ""
        if advanced_functions:
            complexity_context = f" The use of advanced functions like {', '.join(advanced_functions[:2])} indicates sophisticated analytical capabilities."
        
        return f"This is a {level} business document with {formula_count} calculations that {description}.{complexity_context} The structure is well-organized for its intended business purpose."
    
    def _generate_actionable_insights(self, business_context: Dict, document_context: Dict) -> List[str]:
        """Generate actionable insights"""
        
        insights = []
        context_type = business_context['type']
        basic_info = document_context['basic_info']
        
        # Domain-specific actionable insights
        focus_areas = business_context.get('insights_focus', [])
        for focus in focus_areas[:3]:
            if context_type == 'financial_modeling':
                insights.append(f"Use this model to regularly assess {focus} and adjust investment strategies")
            elif context_type == 'budgeting_planning':
                insights.append(f"Monitor {focus} monthly to ensure budget alignment and identify variances early")
            else:
                insights.append(f"Focus on {focus} trends to identify opportunities for business improvement")
        
        # Structure-based insights
        if basic_info['worksheets'] > 3:
            insights.append("Leverage the multi-sheet structure to create comprehensive reports for different stakeholder groups")
        
        if basic_info['total_formulas'] > 30:
            insights.append("Document the calculation methodology to ensure consistent usage across teams")
        
        return insights[:5]
    
    def _generate_potential_concerns(self, basic_info: Dict, formula_patterns: Dict, business_context: Dict) -> List[str]:
        """Generate potential concerns"""
        
        concerns = []
        
        # Complexity concerns
        if basic_info['complexity_score'] > 20:
            concerns.append("High complexity may require specialized training for new users")
        
        if basic_info['total_formulas'] > 100:
            concerns.append("Large number of formulas may impact performance and maintenance")
        
        # Function-specific concerns
        top_functions = formula_patterns.get('top_functions', [])
        risky_functions = ['INDIRECT', 'OFFSET', 'VOLATILE']
        if any(func in top_functions for func in risky_functions):
            concerns.append("Use of volatile functions may cause performance issues with large datasets")
        
        # Business context concerns
        if business_context['type'] == 'financial_modeling':
            concerns.append("Financial models require regular validation and stress testing")
        elif business_context['type'] == 'budgeting_planning':
            concerns.append("Budget models need periodic review to ensure assumptions remain valid")
        
        if not concerns:
            concerns.append("Standard Excel best practices should be followed for optimal performance")
        
        return concerns[:3]
    
    def _generate_next_steps(self, business_context: Dict, basic_info: Dict) -> List[str]:
        """Generate next steps suggestions"""
        
        steps = []
        context_type = business_context['type']
        
        # Always start with validation
        steps.append("Validate all calculations and assumptions with subject matter experts")
        
        # Domain-specific next steps
        if context_type == 'financial_modeling':
            steps.extend([
                "Perform back-testing with historical data to validate model accuracy",
                "Establish regular review cycles for model parameters and assumptions",
                "Create user documentation and training materials for model users"
            ])
        elif context_type == 'budgeting_planning':
            steps.extend([
                "Establish monthly review process to track actual vs. planned performance",
                "Create variance analysis reports for management review",
                "Set up automated alerts for significant budget deviations"
            ])
        else:
            steps.extend([
                "Create regular reporting schedules to track key metrics",
                "Establish data quality checks and validation procedures",
                "Train users on proper usage and interpretation of results"
            ])
        
        # Technical next steps
        if basic_info['total_formulas'] > 50:
            steps.append("Consider implementing version control and change management processes")
        
        return steps[:5]
    
    def _calculate_enhanced_confidence(self, basic_info: Dict, business_context: Dict, document_context: Dict) -> float:
        """Calculate enhanced confidence score based on available data"""
        
        confidence = 0.5  # Base confidence
        
        # Formula-based confidence
        formula_count = basic_info.get('total_formulas', 0)
        if formula_count > 50:
            confidence += 0.2  # High formula density
        elif formula_count > 10:
            confidence += 0.15  # Moderate formula density
        elif formula_count > 0:
            confidence += 0.05  # Some formulas
        # No penalty for 0 formulas - might be data-only document
        
        # Structure-based confidence
        worksheet_count = basic_info.get('worksheets', 0)
        if worksheet_count > 3:
            confidence += 0.1  # Multi-sheet structure
        elif worksheet_count > 1:
            confidence += 0.05  # Some structure
        
        # Business context confidence
        business_confidence = business_context.get('confidence', 0)
        if business_confidence > 0.5:
            confidence += 0.1  # Good business context detection
        elif business_confidence > 0.2:
            confidence += 0.05  # Some business context
        
        # Content richness
        text_content = document_context.get('text_content', [])
        if len(text_content) > 50:
            confidence += 0.1  # Rich text content
        elif len(text_content) > 10:
            confidence += 0.05  # Some text content
        
        # Business indicators
        business_indicators = document_context.get('business_indicators', {})
        indicator_count = sum(len(indicators) for indicators in business_indicators.values())
        if indicator_count > 10:
            confidence += 0.05  # Rich business indicators
        
        # Special handling for data-only documents
        if formula_count == 0 and worksheet_count > 1 and len(text_content) > 20:
            # This looks like a structured data document
            confidence += 0.1  # Boost for structured data
        
        return min(0.95, confidence)  # Cap at 95%
    
    def _generate_fallback_executive_summary(self, business_context: Dict) -> str:
        """Generate fallback executive summary when LLM fails"""
        return f"This business document supports {business_context['purpose']} and is designed for {business_context['audience']}. It provides analytical capabilities and structured data analysis for informed decision-making."

# Example usage and testing
if __name__ == "__main__":
    summarizer = EnhancedDocumentSummarizer(use_llm=False)  # Test with local mode
    print("Enhanced Document Summarizer initialized for testing")
    print("Ready to generate detailed, business-focused summaries")