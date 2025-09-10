"""
FREE Intelligent Document Analyzer
Uses rule-based analysis, pattern recognition, and NLP libraries (no paid APIs)
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import nltk
import textstat
from collections import Counter, defaultdict
from wordcloud import WordCloud
import numpy as np

# Download required NLTK data (free)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

@dataclass
class IntelligentAnalysis:
    """Free intelligent analysis results"""
    document_type: str
    summary: str
    key_insights: List[str]
    business_logic: str
    recommendations: List[str]
    sentiment_analysis: Dict[str, float]
    readability_scores: Dict[str, float]
    keyword_analysis: Dict[str, Any]
    content_themes: List[str]
    risk_indicators: List[str]
    automation_opportunities: List[str]
    confidence_score: float

class FreeIntelligentAnalyzer:
    """
    100% Free document analyzer using rule-based intelligence
    No external APIs, no costs, completely local processing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.stop_words = set()
            self.sentiment_analyzer = None
        
        # Business vocabulary patterns (rule-based)
        self.business_patterns = {
            'financial': {
                'keywords': ['revenue', 'profit', 'cost', 'budget', 'forecast', 'expense', 'income', 'roi', 'margin', 'cash flow'],
                'functions': ['SUM', 'AVERAGE', 'NPV', 'IRR', 'PMT', 'FV', 'PV'],
                'indicators': ['$', '%', 'revenue', 'profit', 'budget']
            },
            'operational': {
                'keywords': ['process', 'workflow', 'efficiency', 'productivity', 'operations', 'inventory', 'supply', 'quality'],
                'functions': ['COUNT', 'COUNTIF', 'MAX', 'MIN', 'RANK'],
                'indicators': ['process', 'workflow', 'operations']
            },
            'analytical': {
                'keywords': ['analysis', 'data', 'metrics', 'kpi', 'dashboard', 'report', 'insights', 'trends'],
                'functions': ['STDEV', 'VAR', 'CORREL', 'TREND', 'FORECAST'],
                'indicators': ['analysis', 'metrics', 'kpi']
            },
            'strategic': {
                'keywords': ['strategy', 'planning', 'goals', 'objectives', 'vision', 'mission', 'growth', 'market'],
                'functions': ['SCENARIO', 'GOALSEEK', 'SOLVER'],
                'indicators': ['strategy', 'planning', 'goals']
            }
        }
        
        # Risk indicators (rule-based)
        self.risk_indicators = {
            'formula_risks': ['hardcoded values', 'circular references', 'volatile functions', 'complex nesting'],
            'process_risks': ['manual data entry', 'no validation', 'single point of failure', 'no backup'],
            'data_risks': ['external dependencies', 'unvalidated inputs', 'inconsistent formats']
        }
        
        # Automation opportunity patterns
        self.automation_patterns = [
            'repetitive calculations',
            'manual data entry',
            'copy-paste operations',
            'routine reporting',
            'data formatting',
            'regular updates'
        ]
    
    def analyze_excel_content(self, excel_analysis) -> IntelligentAnalysis:
        """Analyze Excel content using free intelligence"""
        
        # Extract text content from Excel analysis
        text_content = self._extract_text_from_excel(excel_analysis)
        
        # Analyze formulas
        formula_analysis = self._analyze_formulas_intelligent(excel_analysis)
        
        # Generate summary
        summary = self._generate_excel_summary(excel_analysis, formula_analysis)
        
        # Extract insights
        insights = self._extract_excel_insights(excel_analysis, formula_analysis)
        
        # Identify business logic
        business_logic = self._identify_business_logic(excel_analysis, formula_analysis)
        
        # Generate recommendations
        recommendations = self._generate_excel_recommendations(excel_analysis, formula_analysis)
        
        # Analyze sentiment (if text available)
        sentiment = self._analyze_sentiment(text_content)
        
        # Calculate readability (if text available)
        readability = self._calculate_readability(text_content)
        
        # Keyword analysis
        keywords = self._analyze_keywords(text_content + ' ' + ' '.join(formula_analysis.get('descriptions', [])))
        
        # Identify themes
        themes = self._identify_content_themes(excel_analysis, text_content)
        
        # Risk assessment
        risks = self._assess_excel_risks(excel_analysis, formula_analysis)
        
        # Automation opportunities
        automation = self._identify_automation_opportunities(excel_analysis)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(excel_analysis, formula_analysis)
        
        return IntelligentAnalysis(
            document_type='excel',
            summary=summary,
            key_insights=insights,
            business_logic=business_logic,
            recommendations=recommendations,
            sentiment_analysis=sentiment,
            readability_scores=readability,
            keyword_analysis=keywords,
            content_themes=themes,
            risk_indicators=risks,
            automation_opportunities=automation,
            confidence_score=confidence
        )
    
    def analyze_word_content(self, word_analysis) -> IntelligentAnalysis:
        """Analyze Word document using free intelligence"""
        
        # Extract text content
        text_content = self._extract_text_from_word(word_analysis)
        
        # Generate summary
        summary = self._generate_text_summary(text_content)
        
        # Extract insights
        insights = self._extract_text_insights(word_analysis, text_content)
        
        # Identify business logic from content
        business_logic = self._identify_text_business_logic(text_content)
        
        # Generate recommendations
        recommendations = self._generate_text_recommendations(word_analysis, text_content)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(text_content)
        
        # Readability analysis
        readability = self._calculate_readability(text_content)
        
        # Keyword analysis
        keywords = self._analyze_keywords(text_content)
        
        # Theme identification
        themes = self._identify_text_themes(text_content)
        
        # Risk assessment
        risks = self._assess_text_risks(text_content)
        
        # Automation opportunities
        automation = self._identify_text_automation_opportunities(text_content)
        
        # Confidence score
        confidence = self._calculate_text_confidence(word_analysis, text_content)
        
        return IntelligentAnalysis(
            document_type='word',
            summary=summary,
            key_insights=insights,
            business_logic=business_logic,
            recommendations=recommendations,
            sentiment_analysis=sentiment,
            readability_scores=readability,
            keyword_analysis=keywords,
            content_themes=themes,
            risk_indicators=risks,
            automation_opportunities=automation,
            confidence_score=confidence
        )
    
    def analyze_powerpoint_content(self, pptx_analysis) -> IntelligentAnalysis:
        """Analyze PowerPoint content using free intelligence"""
        
        # Extract text content
        text_content = self._extract_text_from_powerpoint(pptx_analysis)
        
        # Analyze presentation structure
        structure_analysis = self._analyze_presentation_structure(pptx_analysis)
        
        # Generate summary
        summary = self._generate_presentation_summary(pptx_analysis, structure_analysis)
        
        # Extract insights
        insights = self._extract_presentation_insights(pptx_analysis, structure_analysis)
        
        # Business logic
        business_logic = self._identify_presentation_business_logic(text_content, structure_analysis)
        
        # Recommendations
        recommendations = self._generate_presentation_recommendations(pptx_analysis, structure_analysis)
        
        # Sentiment and readability
        sentiment = self._analyze_sentiment(text_content)
        readability = self._calculate_readability(text_content)
        
        # Keywords and themes
        keywords = self._analyze_keywords(text_content)
        themes = self._identify_presentation_themes(text_content, structure_analysis)
        
        # Risks and automation
        risks = self._assess_presentation_risks(pptx_analysis)
        automation = self._identify_presentation_automation_opportunities(pptx_analysis)
        
        # Confidence
        confidence = self._calculate_presentation_confidence(pptx_analysis, structure_analysis)
        
        return IntelligentAnalysis(
            document_type='powerpoint',
            summary=summary,
            key_insights=insights,
            business_logic=business_logic,
            recommendations=recommendations,
            sentiment_analysis=sentiment,
            readability_scores=readability,
            keyword_analysis=keywords,
            content_themes=themes,
            risk_indicators=risks,
            automation_opportunities=automation,
            confidence_score=confidence
        )
    
    def _extract_text_from_excel(self, excel_analysis) -> str:
        """Extract readable text from Excel analysis"""
        text_parts = []
        
        for worksheet in excel_analysis.worksheets:
            text_parts.append(f"Worksheet: {worksheet.name}")
            
            # Extract cell values that are text
            for cell in worksheet.cells:
                if isinstance(cell.value, str) and len(cell.value) > 2:
                    text_parts.append(cell.value)
            
            # Extract comments
            for cell in worksheet.cells:
                if cell.comment:
                    text_parts.append(cell.comment)
        
        return ' '.join(text_parts)
    
    def _analyze_formulas_intelligent(self, excel_analysis) -> Dict[str, Any]:
        """Intelligent formula analysis using patterns"""
        
        all_formulas = []
        descriptions = []
        
        for worksheet in excel_analysis.worksheets:
            all_formulas.extend(worksheet.formulas)
        
        # Analyze formula patterns
        function_usage = Counter()
        complexity_levels = []
        business_categories = defaultdict(int)
        
        for formula_info in all_formulas:
            formula = formula_info.get('formula', '')
            functions = formula_info.get('functions', [])
            
            # Count function usage
            for func in functions:
                function_usage[func] += 1
            
            # Calculate complexity
            complexity = formula_info.get('complexity', {}).get('score', 0)
            complexity_levels.append(complexity)
            
            # Categorize business logic
            category = self._categorize_formula_business_logic(formula, functions)
            business_categories[category] += 1
            
            # Generate description
            description = self._describe_formula(formula, functions, category)
            descriptions.append(description)
        
        return {
            'total_formulas': len(all_formulas),
            'function_usage': dict(function_usage.most_common(10)),
            'average_complexity': np.mean(complexity_levels) if complexity_levels else 0,
            'max_complexity': max(complexity_levels) if complexity_levels else 0,
            'business_categories': dict(business_categories),
            'descriptions': descriptions
        }
    
    def _categorize_formula_business_logic(self, formula: str, functions: List[str]) -> str:
        """Categorize formula business logic"""
        
        formula_lower = formula.lower()
        function_set = set(functions)
        
        for category, patterns in self.business_patterns.items():
            # Check functions
            if function_set & set(patterns['functions']):
                return category
            
            # Check keywords in formula
            if any(keyword in formula_lower for keyword in patterns['keywords']):
                return category
        
        return 'general'
    
    def _describe_formula(self, formula: str, functions: List[str], category: str) -> str:
        """Generate human-readable formula description"""
        
        descriptions = {
            'financial': f"Financial calculation using {', '.join(functions[:3])}",
            'operational': f"Operational analysis with {', '.join(functions[:3])}",
            'analytical': f"Data analysis using {', '.join(functions[:3])}",
            'strategic': f"Strategic calculation with {', '.join(functions[:3])}",
            'general': f"General calculation using {', '.join(functions[:3])}"
        }
        
        base_desc = descriptions.get(category, 'Formula calculation')
        
        if len(functions) > 5:
            base_desc += " (Complex multi-function formula)"
        elif 'IF' in functions:
            base_desc += " with conditional logic"
        elif any(func in functions for func in ['SUM', 'AVERAGE', 'COUNT']):
            base_desc += " for data aggregation"
        
        return base_desc
    
    def _generate_excel_summary(self, excel_analysis, formula_analysis) -> str:
        """Generate intelligent Excel summary"""
        
        worksheets_count = len(excel_analysis.worksheets)
        formulas_count = formula_analysis['total_formulas']
        avg_complexity = formula_analysis['average_complexity']
        top_category = max(formula_analysis['business_categories'].items(), key=lambda x: x[1])[0] if formula_analysis['business_categories'] else 'general'
        
        summary = f"Excel workbook with {worksheets_count} worksheet{'s' if worksheets_count != 1 else ''} "
        summary += f"containing {formulas_count} formula{'s' if formulas_count != 1 else ''}. "
        
        if avg_complexity > 10:
            summary += "The formulas show high complexity indicating sophisticated business logic. "
        elif avg_complexity > 5:
            summary += "The formulas have moderate complexity. "
        else:
            summary += "The formulas are relatively simple. "
        
        summary += f"Primary focus appears to be {top_category} operations. "
        
        # Add VBA analysis if present
        if excel_analysis.vba_code:
            summary += f"Contains VBA automation with {len(excel_analysis.vba_code)} modules. "
        
        return summary
    
    def _extract_excel_insights(self, excel_analysis, formula_analysis) -> List[str]:
        """Extract key insights from Excel analysis"""
        
        insights = []
        
        # Formula insights
        if formula_analysis['total_formulas'] > 50:
            insights.append(f"High formula density with {formula_analysis['total_formulas']} calculations")
        
        if formula_analysis['max_complexity'] > 20:
            insights.append("Contains highly complex formulas that may need review")
        
        # Top functions
        top_functions = list(formula_analysis['function_usage'].keys())[:3]
        if top_functions:
            insights.append(f"Most used functions: {', '.join(top_functions)}")
        
        # Business category insights
        business_cats = formula_analysis['business_categories']
        if business_cats:
            primary_cat = max(business_cats.items(), key=lambda x: x[1])[0]
            insights.append(f"Primary business focus: {primary_cat} operations")
        
        # VBA insights
        if excel_analysis.vba_code:
            insights.append(f"Automation present with {len(excel_analysis.vba_code)} VBA modules")
        
        # Worksheet insights
        if len(excel_analysis.worksheets) > 5:
            insights.append("Multi-sheet workbook indicating complex business model")
        
        return insights[:5]  # Return top 5 insights
    
    def _identify_business_logic(self, excel_analysis, formula_analysis) -> str:
        """Identify business logic patterns"""
        
        business_categories = formula_analysis['business_categories']
        if not business_categories:
            return "General spreadsheet calculations without specific business pattern"
        
        primary_category = max(business_categories.items(), key=lambda x: x[1])[0]
        
        logic_descriptions = {
            'financial': "Financial modeling and analysis workbook with budget/forecast calculations, ROI analysis, and financial metrics tracking",
            'operational': "Operational efficiency tracking with process metrics, inventory management, and performance monitoring",
            'analytical': "Data analysis workbook focused on KPI tracking, trend analysis, and business intelligence reporting",
            'strategic': "Strategic planning tool with scenario analysis, goal tracking, and strategic metrics evaluation"
        }
        
        base_logic = logic_descriptions.get(primary_category, "General business calculations")
        
        # Add VBA context
        if excel_analysis.vba_code:
            base_logic += ". Includes automated processes through VBA scripting"
        
        # Add complexity context
        avg_complexity = formula_analysis['average_complexity']
        if avg_complexity > 15:
            base_logic += " with sophisticated calculation logic"
        elif avg_complexity > 8:
            base_logic += " with moderate complexity calculations"
        
        return base_logic
    
    def _generate_excel_recommendations(self, excel_analysis, formula_analysis) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Complexity recommendations
        if formula_analysis['max_complexity'] > 20:
            recommendations.append("Review highly complex formulas for potential simplification")
        
        if formula_analysis['average_complexity'] > 15:
            recommendations.append("Consider breaking down complex formulas into intermediate steps")
        
        # Error prevention
        if 'IF' in formula_analysis.get('function_usage', {}):
            recommendations.append("Add error handling to conditional formulas using IFERROR")
        
        # Documentation
        if formula_analysis['total_formulas'] > 20:
            recommendations.append("Add comments to document business logic in complex formulas")
        
        # Automation
        if not excel_analysis.vba_code and formula_analysis['total_formulas'] > 30:
            recommendations.append("Consider VBA automation for repetitive calculations")
        
        # Data validation
        recommendations.append("Implement data validation to prevent input errors")
        
        # Backup and version control
        recommendations.append("Establish regular backup and version control procedures")
        
        return recommendations[:5]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using NLTK (free)"""
        
        if not text or not self.sentiment_analyzer:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability scores using textstat (free)"""
        
        if not text:
            return {}
        
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'gunning_fog': textstat.gunning_fog(text)
            }
        except:
            return {}
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze keywords and create word cloud data"""
        
        if not text:
            return {}
        
        try:
            # Tokenize and clean
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
            
            # Count frequencies
            word_freq = Counter(words)
            
            # Get top keywords
            top_keywords = word_freq.most_common(20)
            
            return {
                'total_words': len(words),
                'unique_words': len(set(words)),
                'top_keywords': top_keywords,
                'keyword_density': dict(top_keywords[:10])
            }
        except:
            return {}
    
    def _identify_content_themes(self, excel_analysis, text_content: str) -> List[str]:
        """Identify content themes using pattern matching"""
        
        themes = set()
        
        # Text-based theme detection
        text_lower = text_content.lower()
        
        theme_patterns = {
            'Financial Planning': ['budget', 'forecast', 'financial', 'revenue', 'profit', 'cost'],
            'Data Analysis': ['analysis', 'metrics', 'kpi', 'dashboard', 'report', 'trend'],
            'Operations Management': ['process', 'workflow', 'efficiency', 'operations', 'inventory'],
            'Strategic Planning': ['strategy', 'planning', 'goals', 'objectives', 'growth'],
            'Risk Management': ['risk', 'compliance', 'audit', 'control', 'mitigation'],
            'Performance Tracking': ['performance', 'tracking', 'monitoring', 'measurement']
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.add(theme)
        
        # Excel-specific theme detection
        if excel_analysis.vba_code:
            themes.add('Process Automation')
        
        if hasattr(excel_analysis, 'worksheets') and len(excel_analysis.worksheets) > 3:
            themes.add('Complex Modeling')
        
        return list(themes)[:5]
    
    def _assess_excel_risks(self, excel_analysis, formula_analysis) -> List[str]:
        """Assess risks in Excel file"""
        
        risks = []
        
        # Formula complexity risks
        if formula_analysis['max_complexity'] > 25:
            risks.append("Extremely complex formulas may be error-prone")
        
        # Function-based risks
        risky_functions = ['INDIRECT', 'OFFSET', 'VOLATILE']
        function_usage = formula_analysis.get('function_usage', {})
        
        for risky_func in risky_functions:
            if risky_func in function_usage:
                risks.append(f"Use of {risky_func} function may cause performance issues")
        
        # External reference risks
        if hasattr(excel_analysis, 'external_references') and excel_analysis.external_references:
            risks.append("External references create dependency risks")
        
        # VBA risks
        if excel_analysis.vba_code:
            risks.append("VBA macros require security consideration")
        
        # Circular reference risks
        if hasattr(excel_analysis, 'data_flow') and excel_analysis.data_flow.get('circular_references'):
            risks.append("Circular references detected")
        
        return risks[:5]
    
    def _identify_automation_opportunities(self, excel_analysis) -> List[str]:
        """Identify automation opportunities"""
        
        opportunities = []
        
        # High formula count suggests automation potential
        total_formulas = 0
        for worksheet in excel_analysis.worksheets:
            total_formulas += len(worksheet.formulas)
        
        if total_formulas > 50 and not excel_analysis.vba_code:
            opportunities.append("High formula count suggests VBA automation opportunity")
        
        # Repetitive calculations
        function_patterns = defaultdict(int)
        for worksheet in excel_analysis.worksheets:
            for formula_info in worksheet.formulas:
                if formula_info.get('functions'):
                    pattern = ','.join(sorted(formula_info['functions']))
                    function_patterns[pattern] += 1
        
        repetitive_patterns = [(pattern, count) for pattern, count in function_patterns.items() if count > 5]
        if repetitive_patterns:
            opportunities.append(f"Repetitive calculation patterns detected ({len(repetitive_patterns)} patterns)")
        
        # Data import/export opportunities
        if len(excel_analysis.worksheets) > 1:
            opportunities.append("Multi-sheet structure suggests data consolidation automation")
        
        # Regular reporting
        opportunities.append("Consider automating regular report generation")
        
        # Data validation
        opportunities.append("Implement automated data validation checks")
        
        return opportunities[:5]
    
    def _calculate_confidence_score(self, excel_analysis, formula_analysis) -> float:
        """Calculate confidence score for analysis"""
        
        confidence = 0.8  # Base confidence
        
        # Adjust based on data availability
        if formula_analysis['total_formulas'] > 10:
            confidence += 0.1
        
        if excel_analysis.vba_code:
            confidence += 0.05
        
        if len(excel_analysis.worksheets) > 1:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _extract_text_from_word(self, word_analysis) -> str:
        """Extract text content from Word analysis"""
        full_text = []
        
        for section in word_analysis.sections:
            for paragraph in section.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
        
        return '\n'.join(full_text)
    
    def _generate_text_summary(self, text_content: str) -> str:
        """Generate summary of text content"""
        if not text_content:
            return "Empty document - no content to analyze."
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(text_content.split())
        sentence_count = len(sentences)
        
        # Simple extractive summary - take first few sentences and key sentences
        summary_sentences = []
        
        if sentences:
            # Always include first sentence
            summary_sentences.append(sentences[0])
            
            # Find sentences with key business terms
            business_keywords = ['revenue', 'cost', 'profit', 'budget', 'analysis', 'report', 
                               'strategy', 'project', 'plan', 'recommendation', 'conclusion']
            
            for sentence in sentences[1:]:
                for keyword in business_keywords:
                    if keyword.lower() in sentence.lower() and sentence not in summary_sentences:
                        summary_sentences.append(sentence)
                        break
                
                if len(summary_sentences) >= 3:  # Limit summary length
                    break
        
        summary = '. '.join(summary_sentences) + '.'
        
        if word_count > 100:
            summary += f" This document contains {word_count} words across {sentence_count} sentences."
        
        return summary
    
    def _extract_text_insights(self, word_analysis, text_content: str) -> List[str]:
        """Extract insights from Word document"""
        insights = []
        
        # Document structure insights
        total_paragraphs = word_analysis.total_paragraphs
        total_words = word_analysis.total_words
        tables_count = word_analysis.tables_count
        headings_count = len(word_analysis.headings)
        
        if headings_count > 0:
            insights.append(f"Well-structured document with {headings_count} headings")
        
        if tables_count > 0:
            insights.append(f"Contains {tables_count} tables for data presentation")
        
        if total_words > 1000:
            insights.append("Comprehensive document requiring detailed review")
        elif total_words < 100:
            insights.append("Brief document - quick read")
        
        # Content analysis
        if 'analysis' in text_content.lower():
            insights.append("Contains analytical content")
        
        if any(term in text_content.lower() for term in ['recommend', 'suggest', 'propose']):
            insights.append("Includes recommendations or suggestions")
        
        if any(term in text_content.lower() for term in ['budget', 'cost', 'revenue', 'profit']):
            insights.append("Contains financial information")
        
        return insights
    
    def _identify_text_business_logic(self, text_content: str) -> List[str]:
        """Identify business logic patterns in text"""
        business_patterns = []
        
        # Look for decision patterns
        if re.search(r'\b(if|when|unless|provided|given)\b.*\bthen\b', text_content, re.IGNORECASE):
            business_patterns.append("Contains conditional business logic")
        
        # Look for process flows
        if re.search(r'\b(first|second|third|next|then|finally|step)\b', text_content, re.IGNORECASE):
            business_patterns.append("Describes process or workflow")
        
        # Look for criteria or requirements
        if re.search(r'\b(must|shall|should|require|criteria|standard)\b', text_content, re.IGNORECASE):
            business_patterns.append("Defines requirements or standards")
        
        # Look for metrics or KPIs
        if re.search(r'\b(metric|kpi|measure|performance|target|goal)\b', text_content, re.IGNORECASE):
            business_patterns.append("References performance metrics")
        
        return business_patterns
    
    def _generate_text_recommendations(self, word_analysis, text_content: str) -> List[str]:
        """Generate recommendations for document improvement"""
        recommendations = []
        
        # Structure recommendations
        if len(word_analysis.headings) == 0:
            recommendations.append("Add headings to improve document structure")
        
        if word_analysis.tables_count == 0 and 'data' in text_content.lower():
            recommendations.append("Consider using tables for data presentation")
        
        # Content recommendations
        if word_analysis.total_words > 2000:
            recommendations.append("Consider breaking into smaller sections for better readability")
        
        if not any(term in text_content.lower() for term in ['conclusion', 'summary']):
            recommendations.append("Add a summary or conclusion section")
        
        # Document properties recommendations
        if not word_analysis.document_properties.get('title'):
            recommendations.append("Add a document title in properties")
        
        if not word_analysis.document_properties.get('author'):
            recommendations.append("Add author information to document properties")
        
        return recommendations
    
    def _identify_text_themes(self, text_content: str) -> List[str]:
        """Identify main themes in the text"""
        themes = []
        
        # Business themes
        business_terms = {
            'strategy': ['strategy', 'strategic', 'planning', 'vision'],
            'finance': ['budget', 'cost', 'revenue', 'financial', 'profit'],
            'operations': ['process', 'procedure', 'workflow', 'operation'],
            'analysis': ['analysis', 'research', 'study', 'investigation'],
            'project': ['project', 'initiative', 'implementation'],
            'compliance': ['compliance', 'regulation', 'policy', 'standard']
        }
        
        text_lower = text_content.lower()
        
        for theme, keywords in business_terms.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme.title())
        
        return themes
    
    def _assess_text_risks(self, text_content: str) -> List[str]:
        """Assess potential risks in text content"""
        risks = []
        
        # Look for risk indicators
        risk_keywords = ['risk', 'issue', 'problem', 'concern', 'challenge', 'delay', 'over budget']
        
        for keyword in risk_keywords:
            if keyword.lower() in text_content.lower():
                risks.append(f"Mentions {keyword} - requires attention")
        
        # Check for missing critical information
        if 'tbd' in text_content.lower() or 'to be determined' in text_content.lower():
            risks.append("Contains placeholder information (TBD)")
        
        if len(text_content.split()) < 50:
            risks.append("Very brief document - may lack detail")
        
        return risks
    
    def _identify_text_automation_opportunities(self, text_content: str) -> List[str]:
        """Identify automation opportunities in text"""
        opportunities = []
        
        # Look for repetitive processes
        if re.search(r'\b(daily|weekly|monthly|quarterly|annual|regular)\b', text_content, re.IGNORECASE):
            opportunities.append("Regular processes identified - consider automation")
        
        if re.search(r'\b(manual|manually|copy|paste|type|enter)\b', text_content, re.IGNORECASE):
            opportunities.append("Manual processes mentioned - automation potential")
        
        if re.search(r'\b(report|reporting|dashboard|summary)\b', text_content, re.IGNORECASE):
            opportunities.append("Reporting processes - consider automated reporting")
        
        return opportunities
    
    def _calculate_text_confidence(self, word_analysis, text_content: str) -> float:
        """Calculate confidence score for text analysis"""
        confidence = 0.7  # Base confidence for text analysis
        
        # Adjust based on document completeness
        if word_analysis.total_words > 100:
            confidence += 0.1
        
        if len(word_analysis.headings) > 0:
            confidence += 0.1
        
        if word_analysis.document_properties.get('title'):
            confidence += 0.05
        
        if word_analysis.document_properties.get('author'):
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def export_analysis(self, analysis: IntelligentAnalysis, output_path: str):
        """Export analysis to JSON"""
        
        analysis_dict = asdict(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, default=str)

# Example usage
if __name__ == "__main__":
    analyzer = FreeIntelligentAnalyzer()
    print("Free Intelligent Analyzer initialized!")
    print("No external APIs required - 100% local processing")