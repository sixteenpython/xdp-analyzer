#!/usr/bin/env python3
"""
XDP Analyzer - 100% FREE Version - Main Entry Point
Advanced document analysis for Excel, Word, and PowerPoint files
No external APIs, no costs, complete privacy
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Import our modules
from parsers.excel_parser import AdvancedExcelParser
from analyzers.intelligent_analyzer import FreeIntelligentAnalyzer

class XDPAnalyzer:
    """Main XDP Analyzer class - 100% FREE VERSION"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.setup_logging()
        
        # Initialize parsers
        self.excel_parser = AdvancedExcelParser()
        # Note: Word and PowerPoint parsers coming soon
        
        # Initialize intelligent analyzer (FREE)
        self.intelligent_analyzer = FreeIntelligentAnalyzer()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    # Removed LLM config check - FREE VERSION uses local processing only
    
    def analyze_file(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a single file
        
        Args:
            file_path: Path to file to analyze
            output_path: Optional output path for results
            
        Returns:
            Analysis results dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        self.logger.info(f"Analyzing {file_ext} file: {file_path}")
        
        results = {
            'filename': file_path.name,
            'file_type': file_ext,
            'analysis_timestamp': None,
            'basic_analysis': None,
            'intelligent_analysis': None
        }
        
        # Parse based on file type
        if file_ext in ['.xlsx', '.xlsm', '.xlsb', '.xls']:
            results.update(self._analyze_excel(file_path))
        elif file_ext == '.docx':
            results.update(self._analyze_word(file_path))
        elif file_ext == '.pptx':
            results.update(self._analyze_powerpoint(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Export results if output path provided
        if output_path:
            self._export_results(results, output_path)
        
        return results
    
    def _analyze_excel(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Excel file with FREE intelligent analysis"""
        
        results = {}
        
        # Basic parsing
        self.logger.info("Performing Excel structure analysis...")
        basic_analysis = self.excel_parser.parse(file_path)
        results['basic_analysis'] = basic_analysis
        
        # Intelligent analysis (FREE)
        self.logger.info("Performing intelligent business analysis...")
        intelligent_analysis = self.intelligent_analyzer.analyze_excel_content(basic_analysis)
        results['intelligent_analysis'] = intelligent_analysis
        
        return results
    
    def _analyze_word(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Word document"""
        
        results = {}
        
        # Basic parsing
        self.logger.info("Performing Word document analysis...")
        basic_analysis = self.word_parser.parse(file_path)
        results['basic_analysis'] = basic_analysis
        
        # LLM insights if available
        if self.llm_analyzer:
            self.logger.info("Getting LLM insights...")
            text_content = self.word_parser.extract_text_only(file_path)
            results['llm_insights'] = asyncio.run(self._get_llm_insights_text(
                text_content, 'word'
            ))
        
        return results
    
    def _analyze_powerpoint(self, file_path: Path) -> Dict[str, Any]:
        """Analyze PowerPoint presentation"""
        
        results = {}
        
        # Basic parsing
        self.logger.info("Performing PowerPoint analysis...")
        basic_analysis = self.pptx_parser.parse(file_path)
        results['basic_analysis'] = basic_analysis
        
        # LLM insights if available
        if self.llm_analyzer:
            self.logger.info("Getting LLM insights...")
            text_content = self.pptx_parser.extract_text_only(file_path)
            results['llm_insights'] = asyncio.run(self._get_llm_insights_text(
                text_content, 'powerpoint'
            ))
        
        return results
    
    async def _get_llm_insights(self, analysis, doc_type: str):
        """Get LLM insights from document analysis"""
        
        try:
            # Create content summary for LLM
            if doc_type == 'excel':
                content = f"Excel file analysis:\n"
                content += f"- Worksheets: {len(analysis.worksheets)}\n"
                content += f"- Formulas: {len(getattr(analysis, 'formulas_summary', {}))}\n"
                content += f"- Complexity: {analysis.complexity_metrics.get('complexity_score', 0)}\n"
                
                # Add formula details if available
                if hasattr(analysis, 'business_logic'):
                    content += f"- Business Logic: {json.dumps(analysis.business_logic, default=str)[:500]}..."
            
            request = AnalysisRequest(
                content=content,
                document_type=doc_type,
                analysis_type='business_insights',
                max_tokens=800
            )
            
            response = await self.llm_analyzer.analyze_document(request)
            return response
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {str(e)}")
            return None
    
    async def _get_llm_insights_text(self, text_content: str, doc_type: str):
        """Get LLM insights from text content"""
        
        try:
            # Truncate content if too long
            if len(text_content) > 3000:
                text_content = text_content[:3000] + "..."
            
            request = AnalysisRequest(
                content=text_content,
                document_type=doc_type,
                analysis_type='summary',
                max_tokens=600
            )
            
            response = await self.llm_analyzer.analyze_document(request)
            return response
            
        except Exception as e:
            self.logger.warning(f"LLM analysis failed: {str(e)}")
            return None
    
    def _export_results(self, results: Dict[str, Any], output_path: str):
        """Export analysis results to JSON"""
        
        # Convert complex objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj

def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="XDP Analyzer - Advanced document analysis for Excel, Word, and PowerPoint"
    )
    
    parser.add_argument('file_path', help='Path to file to analyze')
    parser.add_argument('-o', '--output', help='Output path for results (JSON format)')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # LLM configuration options
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--claude-key', help='Claude API key')
    parser.add_argument('--gemini-key', help='Gemini API key')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {'log_level': args.log_level}
    
    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Add API keys if provided
    if args.openai_key:
        config['openai_api_key'] = args.openai_key
    if args.claude_key:
        config['claude_api_key'] = args.claude_key
    if args.gemini_key:
        config['gemini_api_key'] = args.gemini_key
    
    # Initialize analyzer
    analyzer = XDPAnalyzer(config)
    
    try:
        # Analyze file
        results = analyzer.analyze_file(args.file_path, args.output)
        
        # Print summary
        print(f"\n✅ Analysis completed for: {results['filename']}")
        print(f"File type: {results['file_type']}")
        
        if 'basic_analysis' in results and results['basic_analysis']:
            basic = results['basic_analysis']
            
            if results['file_type'] in ['.xlsx', '.xlsm', '.xlsb', '.xls']:
                print(f"Worksheets: {len(basic.worksheets)}")
                if hasattr(basic, 'complexity_metrics'):
                    print(f"Complexity score: {basic.complexity_metrics.get('complexity_score', 'N/A')}")
            
            elif results['file_type'] == '.docx':
                print(f"Word count: {basic.word_count}")
                print(f"Pages: {basic.page_count}")
                print(f"Reading time: {basic.reading_time:.1f} minutes")
            
            elif results['file_type'] == '.pptx':
                print(f"Slides: {basic.total_slides}")
                print(f"Words: {basic.total_words}")
                print(f"Presentation time: {basic.presentation_time:.1f} minutes")
        
        if 'business_analysis' in results and results['business_analysis']:
            business = results['business_analysis']
            print(f"Business processes: {len(business.business_processes)}")
            print(f"Risk score: {business.risk_assessment.get('overall_score', 'N/A')}")
        
        if 'llm_insights' in results and results['llm_insights']:
            print("✨ AI insights: Available")
        
        if args.output:
            print(f"📄 Detailed results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()