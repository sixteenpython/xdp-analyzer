"""
Advanced Excel Parser supporting xlsx, xlsb, xlsm formats
Extracts formulas, VBA code, charts, pivot tables, and business logic
"""

import pandas as pd
import openpyxl
from openpyxl.chart import *
from openpyxl.comments import Comment
import xlrd
import pyxlsb
from typing import Dict, List, Optional, Any, Union
import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from openpyxl.formula.translate import Translator
import ast
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CellInfo:
    """Detailed cell information"""
    address: str
    value: Any
    formula: Optional[str]
    data_type: str
    number_format: Optional[str]
    comment: Optional[str]
    hyperlink: Optional[str]
    font_info: Dict
    fill_info: Dict
    is_merged: bool
    merge_range: Optional[str]

@dataclass
class WorksheetInfo:
    """Comprehensive worksheet information"""
    name: str
    cells: List[CellInfo]
    charts: List[Dict]
    pivot_tables: List[Dict]
    data_validation: List[Dict]
    conditional_formatting: List[Dict]
    named_ranges: List[Dict]
    formulas: List[Dict]
    vba_references: List[str]
    hidden: bool
    protected: bool
    dimensions: Dict[str, int]

@dataclass
class ExcelAnalysis:
    """Complete Excel file analysis"""
    filename: str
    file_type: str
    worksheets: List[WorksheetInfo]
    vba_code: Dict[str, str]
    external_references: List[str]
    defined_names: List[Dict]
    custom_properties: Dict
    formulas_summary: Dict
    business_logic: Dict
    data_flow: Dict
    complexity_metrics: Dict

class AdvancedExcelParser:
    """
    Robust Excel parser supporting multiple formats with comprehensive extraction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.xlsx', '.xlsm', '.xlsb', '.xls']
        
    def parse(self, file_path: Union[str, Path]) -> ExcelAnalysis:
        """
        Parse Excel file with comprehensive analysis
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            ExcelAnalysis object with complete file information
        """
        file_path = Path(file_path)
        self.logger.info(f"Starting parse of {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        try:
            if file_ext in ['.xlsx', '.xlsm']:
                return self._parse_openpyxl(file_path)
            elif file_ext == '.xlsb':
                return self._parse_xlsb(file_path)
            elif file_ext == '.xls':
                return self._parse_xls(file_path)
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            raise
    
    def _parse_openpyxl(self, file_path: Path) -> ExcelAnalysis:
        """Parse using openpyxl for xlsx/xlsm files"""
        
        # Load workbook with all features
        wb = openpyxl.load_workbook(
            file_path, 
            read_only=False,
            keep_vba=True,
            data_only=False,
            keep_links=True
        )
        
        # Extract basic info
        analysis = ExcelAnalysis(
            filename=file_path.name,
            file_type=file_path.suffix,
            worksheets=[],
            vba_code={},
            external_references=[],
            defined_names=[],
            custom_properties={},
            formulas_summary={},
            business_logic={},
            data_flow={},
            complexity_metrics={}
        )
        
        # Extract VBA code if present
        if hasattr(wb, 'vba_archive') and wb.vba_archive:
            analysis.vba_code = self._extract_vba_openpyxl(wb)
        
        # Extract defined names (named ranges)
        analysis.defined_names = self._extract_defined_names(wb)
        
        # Extract custom properties
        analysis.custom_properties = self._extract_custom_properties(wb)
        
        # Process each worksheet
        for ws in wb.worksheets:
            worksheet_info = self._process_worksheet_openpyxl(ws)
            analysis.worksheets.append(worksheet_info)
        
        # Analyze formulas and business logic
        analysis.formulas_summary = self._analyze_formulas(analysis.worksheets)
        analysis.business_logic = self._extract_business_logic(analysis.worksheets)
        analysis.data_flow = self._map_data_flow(analysis.worksheets)
        analysis.complexity_metrics = self._calculate_complexity_metrics(analysis)
        
        wb.close()
        return analysis
    
    # Additional methods would continue here...
    # (Truncated for brevity - the full implementation is quite long)
    
    def export_analysis(self, analysis: ExcelAnalysis, output_path: str) -> None:
        """Export analysis results to JSON"""
        
        # Convert dataclasses to dict for JSON serialization
        analysis_dict = asdict(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, default=str)
        
        self.logger.info(f"Analysis exported to {output_path}")

# Example usage
if __name__ == "__main__":
    parser = AdvancedExcelParser()
    
    # Test with different Excel formats
    test_files = [
        "sample.xlsx",
        "complex_model.xlsm", 
        "binary_file.xlsb",
        "legacy_file.xls"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                analysis = parser.parse(file_path)
                print(f"\nAnalysis for {file_path}:")
                print(f"Worksheets: {len(analysis.worksheets)}")
                print(f"VBA modules: {len(analysis.vba_code)}")
                
                # Export detailed analysis
                parser.export_analysis(analysis, f"{file_path}_analysis.json")
                
            except Exception as e:
                print(f"Error parsing {file_path}: {str(e)}")