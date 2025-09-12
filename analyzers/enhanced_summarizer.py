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
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from textstat import flesch_reading_ease, automated_readability_index
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

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
    # New detailed content analysis fields
    detailed_content_analysis: Dict[str, Any] = None
    # EUDA-style documentation fields
    formula_documentation: Dict[str, Any] = None
    business_process_flows: List[Dict] = None
    calculation_explanations: List[Dict] = None

class EnhancedDocumentSummarizer:
    """
    Enhanced document summarizer using free LLMs with intelligent fallbacks
    """
    
    def __init__(self, use_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize NLTK components for fallback
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        else:
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
            },
            'creative_content': {
                'keywords': ['screenplay', 'script', 'scene', 'character', 'dialogue', 'act', 'fade in', 'fade out', 'int.', 'ext.', 'voice over', 'montage'],
                'purpose': 'creative writing, screenplay, or script development',
                'audience': 'writers, directors, producers, and creative teams',
                'insights_focus': ['story structure', 'character development', 'scene breakdown', 'dialogue patterns']
            },
            'project_management': {
                'keywords': ['task', 'milestone', 'deadline', 'resource', 'timeline', 'schedule', 'status', 'deliverable', 'gantt', 'kanban'],
                'purpose': 'project planning, tracking, and resource management',
                'audience': 'project managers, team leads, and stakeholders',
                'insights_focus': ['project timeline', 'resource allocation', 'risk assessment', 'deliverable status']
            },
            'inventory_logistics': {
                'keywords': ['inventory', 'stock', 'warehouse', 'shipping', 'supplier', 'logistics', 'fulfillment', 'procurement'],
                'purpose': 'inventory management and supply chain operations',
                'audience': 'operations managers, supply chain teams, and procurement staff',
                'insights_focus': ['stock levels', 'supply chain efficiency', 'vendor performance', 'logistics optimization']
            },
            'hr_personnel': {
                'keywords': ['employee', 'payroll', 'benefits', 'performance', 'training', 'recruitment', 'onboarding', 'hr'],
                'purpose': 'human resources and personnel management',
                'audience': 'HR teams, managers, and executives',
                'insights_focus': ['employee metrics', 'compensation analysis', 'performance trends', 'workforce planning']
            },
            'marketing_analytics': {
                'keywords': ['campaign', 'conversion', 'ctr', 'roi', 'impressions', 'clicks', 'engagement', 'social media', 'seo', 'ppc'],
                'purpose': 'marketing performance analysis and campaign optimization',
                'audience': 'marketing teams, digital marketers, and growth analysts',
                'insights_focus': ['campaign performance', 'audience insights', 'channel effectiveness', 'conversion optimization']
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
        
        # Creative content patterns
        creative_patterns = [
            r'\b(screenplay|script|scene|character|dialogue|act|scene)\b',
            r'\b(fade\s+in|fade\s+out|int\.|ext\.|voice\s+over|montage)\b',
            r'\b(story|plot|narrative|character\s+development|story\s+arc)\b'
        ]
        
        # Project management patterns
        project_patterns = [
            r'\b(task|milestone|deadline|deliverable|timeline|schedule)\b',
            r'\b(resource|allocation|gantt|kanban|status|progress)\b',
            r'\b(project|phase|sprint|iteration|backlog)\b'
        ]
        
        for pattern in process_patterns:
            matches = re.findall(pattern, all_text)
            indicators['business_processes'].extend(matches)
        
        # Add creative content indicators
        for pattern in creative_patterns:
            matches = re.findall(pattern, all_text)
            indicators['document_type_clues'].extend(matches)
        
        # Add project management indicators  
        for pattern in project_patterns:
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
            keyword_matches = 0
            
            # Check keywords in content with higher weighting for exact matches
            for keyword in patterns['keywords']:
                if keyword.lower() in all_text:
                    score += 3
                    keyword_matches += 1
                if keyword.lower() in filename:
                    score += 5
                    keyword_matches += 1
            
            # Check in business indicators
            indicators = document_context.get('business_indicators', {})
            for indicator_list in indicators.values():
                for indicator in indicator_list:
                    if any(keyword.lower() in indicator.lower() for keyword in patterns['keywords']):
                        score += 2
                        keyword_matches += 1
            
            # Bonus for multiple keyword matches (indicates stronger domain alignment)
            if keyword_matches >= 2:
                score += keyword_matches * 2
            
            # Penalty for very few matches (reduces false positives)
            if keyword_matches < 2 and score > 0:
                score = max(1, score // 2)
            
            context_scores[context_type] = score
        
        # Get best matching context with improved confidence scoring
        if context_scores and max(context_scores.values()) > 0:
            best_context = max(context_scores.items(), key=lambda x: x[1])[0]
            best_score = context_scores[best_context]
            context_info = self.business_context_patterns[best_context].copy()
            
            # Improved confidence calculation based on content depth
            total_text_length = len(all_text)
            cell_count = sum(len(ws['cells']) for ws in document_context.get('worksheets_detail', []))
            
            # Base confidence from keyword matches
            base_confidence = min(0.95, best_score / 20)  # Scale to max 95%
            
            # Adjust for content depth
            if total_text_length > 1000 and cell_count > 50:
                content_bonus = 0.1
            elif total_text_length > 500 and cell_count > 20:
                content_bonus = 0.05
            else:
                content_bonus = 0
                
            # Penalty if second-highest score is very close (ambiguous)
            scores_list = sorted(context_scores.values(), reverse=True)
            if len(scores_list) > 1 and scores_list[1] / max(scores_list[0], 1) > 0.7:
                ambiguity_penalty = 0.2
            else:
                ambiguity_penalty = 0
                
            final_confidence = max(0.1, min(0.95, base_confidence + content_bonus - ambiguity_penalty))
            context_info['confidence'] = final_confidence
        else:
            best_context = 'general_analysis'
            context_info = {
                'purpose': 'general data analysis and business intelligence',
                'audience': 'business stakeholders and decision makers',
                'insights_focus': ['data patterns', 'business metrics', 'performance indicators'],
                'confidence': 0.2  # Low confidence for unclear content
            }
        
        context_info['type'] = best_context
        context_info['all_scores'] = context_scores
        
        return context_info
    
    def _generate_enhanced_local_summary(self, document_context: Dict[str, Any], business_context: Dict[str, Any]) -> EnhancedSummary:
        """Generate enhanced summary using local intelligence"""
        
        basic_info = document_context['basic_info']
        business_indicators = document_context.get('business_indicators', {})
        formula_patterns = document_context.get('formula_patterns', {})
        
        # Store current context for access by other methods
        self._current_worksheets_detail = document_context.get('worksheets_detail', [])
        self._current_text_content = document_context.get('text_content', [])
        
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
        
        # Generate detailed content analysis
        detailed_content = self._generate_detailed_content_analysis(document_context, business_indicators)
        
        # Generate EUDA-style formula documentation
        formula_docs = self._generate_formula_documentation(document_context)
        business_processes = self._generate_business_process_documentation(document_context)
        calc_explanations = self._generate_calculation_explanations(document_context)
        
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
            generation_method='enhanced_local',
            detailed_content_analysis=detailed_content,
            formula_documentation=formula_docs,
            business_process_flows=business_processes,
            calculation_explanations=calc_explanations
        )
    
    def _generate_executive_summary(self, basic_info: Dict, business_context: Dict, indicators: Dict) -> str:
        """Generate content-specific executive summary"""
        
        filename = basic_info['filename'].replace('.xlsx', '').replace('.xls', '').replace('_', ' ').title()
        ws_count = basic_info['worksheets']
        formula_count = basic_info['total_formulas']
        context_type = business_context['type']
        
        # Get actual content details for more specific summary
        worksheets_detail = getattr(self, '_current_worksheets_detail', [])
        text_content = getattr(self, '_current_text_content', [])
        
        # Build content-aware summary
        content_preview = ""
        if worksheets_detail:
            worksheet_names = [ws['name'] for ws in worksheets_detail]
            content_preview = f" The worksheets ({', '.join(worksheet_names)}) "
            
            # Add sample content if available
            sample_content = []
            for ws in worksheets_detail[:2]:
                if ws.get('sample_text'):
                    sample_content.extend(ws['sample_text'][:2])
            
            if sample_content:
                clean_samples = [text.strip() for text in sample_content if text and len(text.strip()) > 1][:3]
                if clean_samples:
                    content_preview += f"contain data such as '{', '.join(clean_samples)}'. "
        
        # Extract key business terms for more specificity
        key_terms = []
        for category, terms in indicators.items():
            if terms and category != 'document_type_clues':
                key_terms.extend(terms[:2])
        unique_key_terms = list(set(key_terms))[:4]
        
        terms_context = ""
        if unique_key_terms:
            terms_context = f" Key business terms found include: {', '.join(unique_key_terms)}."
        
        # Generate context-specific but content-aware summary
        if context_type == 'creative_content':
            creative_indicators = indicators.get('document_type_clues', [])
            if any(term in creative_indicators for term in ['screenplay', 'script', 'scene', 'character']):
                return f"{filename} is a creative writing document with {ws_count} worksheets focused on screenplay or script development.{content_preview}This workbook organizes story elements, character information, scene breakdowns, or dialogue structure.{terms_context}"
            else:
                return f"{filename} is a creative content planning document with {ws_count} worksheets.{content_preview}This workbook organizes creative project elements and planning details.{terms_context}"
        
        elif context_type == 'financial_modeling':
            if formula_count > 0:
                return f"This is a sophisticated financial analysis workbook ({filename}) with {ws_count} worksheets and {formula_count} calculations.{content_preview}It's designed for risk assessment and portfolio management with quantitative modeling capabilities.{terms_context}"
            else:
                return f"This is a financial data repository ({filename}) with {ws_count} worksheets containing structured financial data.{content_preview}It provides foundational data for financial analysis and decision-making.{terms_context}"
        
        elif context_type == 'budgeting_planning':
            if formula_count > 0:
                return f"{filename} is a comprehensive budgeting tool with {ws_count} worksheets and {formula_count} calculations.{content_preview}This workbook enables finance teams to track expenses, forecast revenue, and plan resource allocation.{terms_context}"
            else:
                return f"{filename} is a budgeting data repository with {ws_count} worksheets.{content_preview}It serves as a foundation for budget analysis and financial planning.{terms_context}"
        
        elif context_type == 'project_management':
            return f"{filename} is a project management workbook with {ws_count} worksheets for tracking progress, resources, and deliverables.{content_preview}This document serves project managers and teams in monitoring timelines and allocating resources.{terms_context}"
        
        elif context_type == 'hr_personnel':
            return f"{filename} is an HR management workbook with {ws_count} worksheets focused on personnel data and analytics.{content_preview}This document supports HR teams in managing employee information and analyzing workforce patterns.{terms_context}"
        
        else:
            # Content-aware general summary
            doc_clues = indicators.get('document_type_clues', [])
            if doc_clues:
                clue_text = f" Document contains indicators of: {', '.join(doc_clues[:3])}."
            else:
                clue_text = ""
            
            if formula_count > 0:
                return f"{filename} is a business analysis workbook with {ws_count} worksheets and {formula_count} calculations.{content_preview}This document provides data analysis and business intelligence capabilities.{terms_context}{clue_text}"
            else:
                return f"{filename} is a data repository with {ws_count} worksheets containing structured business data.{content_preview}This document serves as a foundation for business analysis and reporting.{terms_context}{clue_text}"
    
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
        """Generate main findings based on actual document content"""
        
        findings = []
        basic_info = document_context['basic_info']
        formula_patterns = document_context.get('formula_patterns', {})
        business_indicators = document_context.get('business_indicators', {})
        worksheets = document_context.get('worksheets_detail', [])
        text_content = document_context.get('text_content', [])
        
        # Finding about actual content
        if worksheets:
            worksheet_names = [ws['name'] for ws in worksheets]
            findings.append(f"Document contains worksheets: {', '.join(worksheet_names)}")
            
            # Analyze actual cell content
            total_cells = sum(ws.get('cell_count', 0) for ws in worksheets)
            if total_cells > 0:
                findings.append(f"Contains {total_cells} data cells across all worksheets")
                
                # Show sample content if available
                for ws in worksheets[:2]:  # First 2 worksheets
                    sample_text = ws.get('sample_text', [])
                    if sample_text:
                        clean_sample = [text for text in sample_text[:3] if text and len(text.strip()) > 1]
                        if clean_sample:
                            findings.append(f"'{ws['name']}' sheet contains data like: {', '.join(clean_sample[:3])}")
        
        # Finding about key terms found in content
        all_terms = []
        for category, terms in business_indicators.items():
            if terms and category != 'document_type_clues':
                all_terms.extend(terms[:2])  # Top 2 from each category
        
        if all_terms:
            unique_terms = list(set(all_terms))[:5]
            findings.append(f"Key business terms identified: {', '.join(unique_terms)}")
        
        # Finding about document type clues
        doc_clues = business_indicators.get('document_type_clues', [])
        if doc_clues:
            findings.append(f"Document type indicators: {', '.join(doc_clues[:3])}")
        
        # Finding about computational complexity
        formula_count = basic_info['total_formulas']
        if formula_count > 100:
            findings.append(f"Highly automated with {formula_count} formulas, indicating sophisticated business logic")
        elif formula_count > 20:
            findings.append(f"Moderate automation with {formula_count} calculations supporting business analysis")
        elif formula_count > 0:
            top_functions = formula_patterns.get('top_functions', [])[:3]
            if top_functions:
                findings.append(f"Contains {formula_count} calculations using functions like {', '.join(top_functions)}")
            else:
                findings.append(f"Contains {formula_count} calculations for data processing")
        else:
            findings.append("Data-only document with no calculations - pure data repository")
        
        # Finding about time dimensions
        time_periods = business_indicators.get('time_periods', [])
        if time_periods:
            findings.append(f"Time-based analysis covering {', '.join(time_periods[:3])}")
        
        return findings[:7]  # Return top 7 findings for more detail
    
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
        """Generate content-specific actionable insights"""
        
        insights = []
        context_type = business_context['type']
        basic_info = document_context['basic_info']
        worksheets = document_context.get('worksheets_detail', [])
        business_indicators = document_context.get('business_indicators', {})
        
        # Content-specific insights based on actual data
        if worksheets:
            worksheet_names = [ws['name'] for ws in worksheets]
            insights.append(f"Analyze the data in the '{worksheet_names[0]}' sheet to identify key patterns and trends")
            
            if len(worksheet_names) > 1:
                insights.append(f"Compare data across '{worksheet_names[0]}' and '{worksheet_names[1]}' sheets to identify relationships and correlations")
        
        # Insights based on key business terms found
        key_terms = []
        for category, terms in business_indicators.items():
            if terms and category != 'document_type_clues':
                key_terms.extend(terms[:2])
        
        if key_terms:
            unique_terms = list(set(key_terms))[:3]
            insights.append(f"Focus analysis on the identified key areas: {', '.join(unique_terms)}")
        
        # Domain-specific actionable insights
        focus_areas = business_context.get('insights_focus', [])
        if context_type == 'creative_content':
            insights.append("Use this document to track story development progress and character arcs")
            insights.append("Consider creating visualizations of story structure and character relationships")
        elif context_type == 'financial_modeling':
            if focus_areas:
                insights.append(f"Use this model to regularly assess {focus_areas[0]} and adjust investment strategies")
        elif context_type == 'budgeting_planning':
            if focus_areas:
                insights.append(f"Monitor {focus_areas[0]} monthly to ensure budget alignment and identify variances early")
        else:
            # Generic content-aware insights
            if basic_info['total_formulas'] == 0:
                insights.append("Since this is a data-only document, consider adding calculations to derive insights from the raw data")
                insights.append("Create pivot tables or charts to visualize the patterns in this data repository")
            else:
                insights.append("Review and validate the existing calculations to ensure accuracy and relevance")
        
        # Structure-based insights
        if basic_info['worksheets'] > 3:
            insights.append("Leverage the multi-sheet structure to create comprehensive reports for different stakeholder groups")
        
        if basic_info['total_formulas'] > 30:
            insights.append("Document the calculation methodology to ensure consistent usage across teams")
        elif basic_info['total_formulas'] == 0 and basic_info['worksheets'] > 1:
            insights.append("Consider consolidating related data from multiple sheets for easier analysis")
        
        return insights[:6]  # Return up to 6 insights
    
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
    
    def _generate_detailed_content_analysis(self, document_context: Dict[str, Any], business_indicators: Dict) -> Dict[str, Any]:
        """Generate detailed content analysis showing exactly what's in the document"""
        
        worksheets_detail = document_context.get('worksheets_detail', [])
        text_content = document_context.get('text_content', [])
        basic_info = document_context['basic_info']
        
        analysis = {
            'worksheet_breakdown': [],
            'content_samples': [],
            'key_terms_found': {},
            'data_structure': {},
            'content_statistics': {}
        }
        
        # Detailed worksheet breakdown
        for ws in worksheets_detail:
            ws_analysis = {
                'name': ws['name'],
                'cell_count': ws.get('cell_count', 0),
                'formula_count': ws.get('formula_count', 0),
                'has_charts': ws.get('has_charts', False),
                'has_pivot_tables': ws.get('has_pivot_tables', False),
                'sample_content': []
            }
            
            # Add sample content from this worksheet
            sample_text = ws.get('sample_text', [])
            if sample_text:
                clean_samples = [text.strip() for text in sample_text[:5] if text and len(text.strip()) > 1]
                ws_analysis['sample_content'] = clean_samples
                analysis['content_samples'].extend(clean_samples[:3])
            
            analysis['worksheet_breakdown'].append(ws_analysis)
        
        # Key terms analysis
        for category, terms in business_indicators.items():
            if terms:
                analysis['key_terms_found'][category] = terms[:5]  # Top 5 terms per category
        
        # Data structure analysis
        analysis['data_structure'] = {
            'total_worksheets': basic_info['worksheets'],
            'total_formulas': basic_info['total_formulas'],
            'total_data_cells': sum(ws.get('cell_count', 0) for ws in worksheets_detail),
            'complexity_score': basic_info.get('complexity_score', 0)
        }
        
        # Content statistics
        all_text = ' '.join(text_content)
        analysis['content_statistics'] = {
            'total_text_length': len(all_text),
            'unique_text_items': len(set(text_content)),
            'text_sample_count': len(text_content)
        }
        
        # Extract most common words (excluding common stopwords)
        if text_content:
            word_counts = {}
            for text in text_content:
                words = text.lower().split()
                for word in words:
                    if len(word) > 2 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'has', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way', 'may', 'say']:
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get top 10 most common words
            if word_counts:
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis['most_common_words'] = [word for word, count in top_words if count > 1]
        
        return analysis
    
    def _generate_formula_documentation(self, document_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate EUDA-style formula documentation explaining each calculation"""
        
        formula_docs = {
            'total_formulas': 0,
            'formula_breakdown_by_purpose': {},
            'individual_formulas': [],
            'calculation_complexity': {},
            'data_dependencies': {}
        }
        
        worksheets = document_context.get('worksheets_detail', [])
        all_formulas = []
        
        # Collect all formulas with their context
        for ws in worksheets:
            for formula in ws.get('formulas', []):
                formula_info = {
                    'worksheet': ws['name'],
                    'cell_address': formula.get('address', formula.get('cell', 'Unknown')),
                    'formula': formula.get('formula', ''),
                    'result_value': formula.get('result_value', formula.get('value', 'N/A')),
                    'business_purpose': formula.get('business_purpose', {}),
                    'calculation_type': formula.get('calculation_type', 'unknown'),
                    'input_ranges': formula.get('input_ranges', []),
                    'logical_conditions': formula.get('logical_conditions', []),
                    'math_operations': formula.get('math_operations', []),
                    'data_lookups': formula.get('data_lookups', {}),
                    'explanation': self._create_formula_explanation(formula)
                }
                all_formulas.append(formula_info)
        
        formula_docs['total_formulas'] = len(all_formulas)
        formula_docs['individual_formulas'] = all_formulas
        
        # Group formulas by business purpose
        purpose_groups = {}
        for formula in all_formulas:
            purpose = formula['business_purpose'].get('primary_purpose', 'unknown')
            if purpose not in purpose_groups:
                purpose_groups[purpose] = []
            purpose_groups[purpose].append(formula)
        
        formula_docs['formula_breakdown_by_purpose'] = purpose_groups
        
        # Calculate complexity metrics
        complexity_scores = [f.get('complexity', {}).get('score', 0) for f in all_formulas]
        if complexity_scores:
            formula_docs['calculation_complexity'] = {
                'average_complexity': sum(complexity_scores) / len(complexity_scores),
                'max_complexity': max(complexity_scores),
                'complexity_distribution': {
                    'simple': len([s for s in complexity_scores if s < 5]),
                    'moderate': len([s for s in complexity_scores if 5 <= s < 15]),
                    'complex': len([s for s in complexity_scores if s >= 15])
                }
            }
        
        return formula_docs
    
    def _create_formula_explanation(self, formula: Dict) -> str:
        """Create a plain English explanation of what a formula does"""
        
        business_purpose = formula.get('business_purpose', {})
        calc_type = formula.get('calculation_type', 'unknown')
        formula_text = formula.get('formula', '')
        input_ranges = formula.get('input_ranges', [])
        
        # Base explanation from business purpose
        base_explanation = business_purpose.get('explanation', 'Performs a calculation')
        
        # Add specific details about inputs and operations
        explanation_parts = [base_explanation]
        
        if input_ranges:
            ranges_text = ', '.join(input_ranges[:3])  # First 3 ranges
            explanation_parts.append(f"Uses data from ranges: {ranges_text}")
        
        # Add operation-specific details
        math_ops = formula.get('math_operations', [])
        if math_ops:
            ops_text = ', '.join(math_ops)
            explanation_parts.append(f"Performs: {ops_text}")
        
        logical_conditions = formula.get('logical_conditions', [])
        if logical_conditions:
            explanation_parts.append(f"Applies business logic with {len(logical_conditions)} condition(s)")
        
        data_lookups = formula.get('data_lookups', {})
        if data_lookups.get('lookup_functions'):
            lookup_funcs = ', '.join(data_lookups['lookup_functions'])
            explanation_parts.append(f"Retrieves data using: {lookup_funcs}")
        
        return '. '.join(explanation_parts) + '.'
    
    def _generate_business_process_documentation(self, document_context: Dict[str, Any]) -> List[Dict]:
        """Generate business process flow documentation from calculation patterns"""
        
        processes = []
        worksheets = document_context.get('worksheets_detail', [])
        
        # Analyze each worksheet as a potential business process
        for ws in worksheets:
            if ws.get('formulas') and len(ws['formulas']) > 1:
                process = {
                    'process_name': f"{ws['name']} Calculation Process",
                    'worksheet': ws['name'],
                    'description': self._infer_process_description(ws),
                    'steps': [],
                    'inputs': set(),
                    'outputs': [],
                    'business_value': self._identify_business_value(ws)
                }
                
                # Sort formulas by cell address to show logical flow
                formulas = sorted(ws['formulas'], key=lambda x: x.get('address', x.get('cell', 'A1')))
                
                for i, formula in enumerate(formulas, 1):
                    step = {
                        'step_number': i,
                        'cell': formula.get('address', formula.get('cell', 'Unknown')),
                        'description': self._create_formula_explanation(formula),
                        'business_purpose': formula.get('business_purpose', {}).get('business_function', 'Data Processing'),
                        'inputs': formula.get('input_ranges', []),
                        'calculation_type': formula.get('calculation_type', 'unknown')
                    }
                    
                    # Track inputs and outputs
                    for inp in step['inputs']:
                        process['inputs'].add(inp)
                    
                    process['steps'].append(step)
                    process['outputs'].append(step['cell'])
                
                process['inputs'] = list(process['inputs'])
                processes.append(process)
        
        return processes
    
    def _infer_process_description(self, worksheet_info: Dict) -> str:
        """Infer what business process a worksheet represents"""
        
        ws_name = worksheet_info['name'].lower()
        formulas = worksheet_info.get('formulas', [])
        
        # Analyze worksheet name for clues
        if any(keyword in ws_name for keyword in ['summary', 'dashboard', 'report']):
            return f"This appears to be a reporting/summary process that consolidates data from multiple sources to create executive-level insights."
        
        elif any(keyword in ws_name for keyword in ['calc', 'calculation', 'model']):
            return f"This appears to be a calculation engine that processes input data through mathematical and logical operations to derive business metrics."
        
        elif any(keyword in ws_name for keyword in ['data', 'input', 'source']):
            return f"This appears to be a data processing workflow that cleanses, transforms, and prepares raw data for downstream analysis."
        
        else:
            # Analyze formula types for clues
            purposes = [f.get('business_purpose', {}).get('primary_purpose', 'unknown') for f in formulas]
            if 'financial_calculation' in purposes:
                return f"This appears to be a financial analysis process that computes financial metrics and ratios for business decision-making."
            elif 'data_aggregation' in purposes:
                return f"This appears to be a data aggregation process that summarizes information from various sources into consolidated reports."
            elif 'data_lookup' in purposes:
                return f"This appears to be a data integration process that combines information from multiple reference sources."
            else:
                return f"This appears to be a business calculation process with {len(formulas)} computational steps for data analysis."
    
    def _identify_business_value(self, worksheet_info: Dict) -> str:
        """Identify the business value delivered by this process"""
        
        formulas = worksheet_info.get('formulas', [])
        purposes = [f.get('business_purpose', {}).get('primary_purpose', 'unknown') for f in formulas]
        
        value_descriptions = {
            'financial_calculation': 'Enables data-driven financial decision making and investment analysis',
            'data_aggregation': 'Provides consolidated reporting and KPI monitoring capabilities',
            'data_lookup': 'Ensures data consistency and enables cross-referencing across business systems',
            'conditional_logic': 'Automates business rule application and decision-making processes',
            'text_processing': 'Standardizes data formats and improves data quality for analysis',
            'date_calculation': 'Enables time-based analysis and project management capabilities'
        }
        
        if purposes:
            primary_purpose = max(set(purposes), key=purposes.count)
            return value_descriptions.get(primary_purpose, 'Supports business operations through automated calculations')
        
        return 'Processes business data to support operational decision-making'
    
    def _generate_calculation_explanations(self, document_context: Dict[str, Any]) -> List[Dict]:
        """Generate detailed explanations for each calculation showing business logic"""
        
        explanations = []
        worksheets = document_context.get('worksheets_detail', [])
        
        for ws in worksheets:
            for formula in ws.get('formulas', []):
                explanation = {
                    'worksheet': ws['name'],
                    'cell': formula.get('address', formula.get('cell', 'Unknown')),
                    'formula': formula.get('formula', ''),
                    'result': formula.get('result_value', formula.get('value', 'N/A')),
                    'business_explanation': self._create_detailed_business_explanation(formula),
                    'technical_details': self._extract_technical_details(formula),
                    'dependencies': formula.get('input_ranges', []),
                    'impact': self._assess_calculation_impact(formula)
                }
                explanations.append(explanation)
        
        return explanations
    
    def _create_detailed_business_explanation(self, formula: Dict) -> str:
        """Create a detailed business explanation for a specific calculation with enhanced financial modeling context"""
        
        purpose = formula.get('business_purpose', {})
        formula_text = formula.get('formula', '').upper()
        
        # Start with the general purpose
        explanation = purpose.get('explanation', 'This calculation processes business data')
        
        # Enhanced function explanations with financial context
        if 'PERCENTILE' in formula_text and any(term in formula_text for term in ['VAR', 'RISK']):
            explanation += " by calculating Value at Risk (VaR) using the percentile method to estimate potential losses"
        elif 'NORM.DIST' in formula_text and 'EXP' in formula_text:
            explanation += " by implementing Black-Scholes option pricing model using normal distribution and exponential functions"
        elif 'MMULT' in formula_text and 'CORREL' in formula_text:
            explanation += " by performing matrix multiplication for portfolio risk calculation using correlation matrices"
        elif 'RAND' in formula_text and 'AVERAGE' in formula_text:
            explanation += " by running Monte Carlo simulation with random number generation for probabilistic analysis"
        elif 'SUM' in formula_text:
            explanation += " by adding up values"
        elif 'AVERAGE' in formula_text:
            explanation += " by calculating the average"
        elif 'IF' in formula_text:
            explanation += " by applying conditional business logic"
        elif any(func in formula_text for func in ['VLOOKUP', 'INDEX', 'MATCH']):
            explanation += " by looking up related information from reference data"
        elif any(func in formula_text for func in ['NPV', 'IRR', 'PMT']):
            explanation += " by performing time value of money calculations for investment analysis"
        elif any(func in formula_text for func in ['STDEV', 'VAR']):
            explanation += " by calculating statistical measures of variability and risk"
        
        # Add mathematical framework context
        math_framework = purpose.get('mathematical_framework', '')
        if math_framework:
            explanation += f" using {math_framework}"
        
        # Add context about likely use case
        use_case = purpose.get('likely_use_case', '')
        if use_case:
            explanation += f". This is typically used for {use_case}"
        
        # Add regulatory context if applicable
        model_complexity = purpose.get('model_complexity', '')
        if model_complexity in ['advanced', 'highly_advanced']:
            explanation += ". This sophisticated calculation may require model validation and regulatory oversight"
        
        return explanation + "."
    
    def _extract_technical_details(self, formula: Dict) -> Dict[str, Any]:
        """Extract technical details about the formula structure"""
        
        return {
            'calculation_type': formula.get('calculation_type', 'unknown'),
            'dependency_level': len(formula.get('input_ranges', [])),
            'math_operations': formula.get('math_operations', []),
            'logical_conditions': len(formula.get('logical_conditions', [])),
            'lookup_operations': formula.get('data_lookups', {}).get('lookup_functions', []),
            'complexity_score': formula.get('complexity', {}).get('score', 0)
        }
    
    def _assess_calculation_impact(self, formula: Dict) -> str:
        """Assess the business impact of this calculation with enhanced financial modeling perspective"""
        
        purpose = formula.get('business_purpose', {}).get('primary_purpose', 'unknown')
        model_complexity = formula.get('business_purpose', {}).get('model_complexity', 'basic')
        
        impact_descriptions = {
            'financial_calculation': 'High impact - affects financial reporting and investment decisions',
            'risk_modeling': 'Critical impact - affects regulatory capital, risk limits, and compliance reporting',
            'derivatives_pricing': 'Critical impact - affects trading decisions, hedging strategies, and P&L attribution',
            'portfolio_analysis': 'High impact - affects asset allocation, risk budgeting, and performance attribution',
            'monte_carlo_simulation': 'High impact - affects scenario analysis, stress testing, and capital planning',
            'option_pricing': 'Critical impact - affects derivatives trading, risk management, and fair value reporting',
            'fixed_income_calculation': 'High impact - affects bond pricing, duration hedging, and interest rate risk management',
            'time_series_analysis': 'Medium-High impact - affects forecasting, trend analysis, and strategic planning',
            'data_aggregation': 'Medium impact - affects reporting accuracy and KPI visibility',
            'data_lookup': 'Medium impact - affects data consistency and operational efficiency',
            'conditional_logic': 'High impact - affects automated decision-making processes',
            'text_processing': 'Low impact - affects data presentation and formatting',
            'date_calculation': 'Medium impact - affects time-based analysis and scheduling'
        }
        
        base_impact = impact_descriptions.get(purpose, 'Impact varies based on business context and usage')
        
        # Enhance impact assessment based on model complexity
        if model_complexity in ['highly_advanced', 'advanced']:
            if 'Critical' not in base_impact and 'High' not in base_impact:
                base_impact = 'High impact - ' + base_impact.split(' - ')[1] if ' - ' in base_impact else base_impact
            base_impact += '. Requires model validation and regulatory oversight.'
        
        return base_impact

# Example usage and testing
if __name__ == "__main__":
    summarizer = EnhancedDocumentSummarizer(use_llm=False)  # Test with local mode
    print("Enhanced Document Summarizer initialized for testing")
    print("Ready to generate detailed, business-focused summaries")