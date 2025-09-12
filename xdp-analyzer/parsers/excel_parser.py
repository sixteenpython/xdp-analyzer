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
    
    def _parse_xlsb(self, file_path: Path) -> ExcelAnalysis:
        """Parse using pyxlsb for xlsb files"""
        
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
        
        try:
            with pyxlsb.open_workbook(file_path) as wb:
                for sheet_name in wb.get_sheet_names():
                    worksheet_info = self._process_worksheet_xlsb(wb, sheet_name)
                    analysis.worksheets.append(worksheet_info)
        except Exception as e:
            self.logger.error(f"Error reading xlsb file: {e}")
            # Return minimal analysis
            analysis.worksheets.append(WorksheetInfo(
                name="Sheet1",
                cells=[],
                charts=[],
                pivot_tables=[],
                data_validation=[],
                conditional_formatting=[],
                named_ranges=[],
                formulas=[],
                vba_references=[],
                hidden=False,
                protected=False,
                dimensions={'max_row': 0, 'max_col': 0, 'data_cells': 0}
            ))
        
        # Analyze extracted content
        analysis.formulas_summary = self._analyze_formulas(analysis.worksheets)
        analysis.business_logic = self._extract_business_logic(analysis.worksheets)
        analysis.data_flow = self._map_data_flow(analysis.worksheets)
        analysis.complexity_metrics = self._calculate_complexity_metrics(analysis)
        
        return analysis
    
    def _parse_xls(self, file_path: Path) -> ExcelAnalysis:
        """Parse using xlrd for xls files"""
        
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
        
        try:
            wb = xlrd.open_workbook(file_path, formatting_info=True)
            for sheet_idx, sheet_name in enumerate(wb.sheet_names()):
                sheet = wb.sheet_by_index(sheet_idx)
                worksheet_info = self._process_worksheet_xls(sheet)
                analysis.worksheets.append(worksheet_info)
        except Exception as e:
            self.logger.error(f"Error reading xls file: {e}")
            # Return minimal analysis
            analysis.worksheets.append(WorksheetInfo(
                name="Sheet1",
                cells=[],
                charts=[],
                pivot_tables=[],
                data_validation=[],
                conditional_formatting=[],
                named_ranges=[],
                formulas=[],
                vba_references=[],
                hidden=False,
                protected=False,
                dimensions={'max_row': 0, 'max_col': 0, 'data_cells': 0}
            ))
        
        # Analyze extracted content
        analysis.formulas_summary = self._analyze_formulas(analysis.worksheets)
        analysis.business_logic = self._extract_business_logic(analysis.worksheets)
        analysis.data_flow = self._map_data_flow(analysis.worksheets)
        analysis.complexity_metrics = self._calculate_complexity_metrics(analysis)
        
        return analysis
    
    def _process_worksheet_xlsb(self, wb, sheet_name: str) -> WorksheetInfo:
        """Process xlsb worksheet and extract content"""
        
        cells = []
        formulas = []
        
        try:
            with wb.get_sheet_by_name(sheet_name) as ws:
                for row_idx, row in enumerate(ws.rows()):
                    for col_idx, cell in enumerate(row):
                        if cell is not None and cell.v is not None:
                            # Convert to Excel address format
                            cell_address = f"{chr(65 + col_idx)}{row_idx + 1}"
                            
                            cell_info = CellInfo(
                                address=cell_address,
                                value=cell.v,
                                formula=cell.f if hasattr(cell, 'f') and cell.f else None,
                                data_type=str(type(cell.v).__name__),
                                number_format=None,
                                comment=None,
                                hyperlink=None,
                                font_info={},
                                fill_info={},
                                is_merged=False,
                                merge_range=None
                            )
                            cells.append(cell_info)
                            
                            # If cell has formula, add to formulas list
                            if hasattr(cell, 'f') and cell.f:
                                formula_info = {
                                    'address': cell_address,
                                    'formula': cell.f,
                                    'functions': self._extract_functions_from_formula(cell.f),
                                    'complexity': self._calculate_formula_complexity(cell.f)
                                }
                                formulas.append(formula_info)
        except Exception as e:
            self.logger.warning(f"Error processing xlsb sheet {sheet_name}: {e}")
        
        return WorksheetInfo(
            name=sheet_name,
            cells=cells,
            charts=[],
            pivot_tables=[],
            data_validation=[],
            conditional_formatting=[],
            named_ranges=[],
            formulas=formulas,
            vba_references=[],
            hidden=False,
            protected=False,
            dimensions={
                'max_row': len([c for c in cells if c.address.endswith('1')]),
                'max_col': max([ord(c.address[0]) - 64 for c in cells]) if cells else 0,
                'data_cells': len(cells)
            }
        )
    
    def _process_worksheet_xls(self, sheet) -> WorksheetInfo:
        """Process xls worksheet and extract content"""
        
        cells = []
        formulas = []
        
        for row_idx in range(sheet.nrows):
            for col_idx in range(sheet.ncols):
                cell = sheet.cell(row_idx, col_idx)
                if cell.value is not None and str(cell.value).strip():
                    # Convert to Excel address format
                    cell_address = f"{chr(65 + col_idx)}{row_idx + 1}"
                    
                    cell_info = CellInfo(
                        address=cell_address,
                        value=cell.value,
                        formula=cell.formula if hasattr(cell, 'formula') and cell.formula else None,
                        data_type=str(type(cell.value).__name__),
                        number_format=None,
                        comment=None,
                        hyperlink=None,
                        font_info={},
                        fill_info={},
                        is_merged=False,
                        merge_range=None
                    )
                    cells.append(cell_info)
                    
                    # If cell has formula, add to formulas list
                    if hasattr(cell, 'formula') and cell.formula:
                        formula_info = {
                            'address': cell_address,
                            'formula': cell.formula,
                            'functions': self._extract_functions_from_formula(cell.formula),
                            'complexity': self._calculate_formula_complexity(cell.formula)
                        }
                        formulas.append(formula_info)
        
        return WorksheetInfo(
            name=sheet.name,
            cells=cells,
            charts=[],
            pivot_tables=[],
            data_validation=[],
            conditional_formatting=[],
            named_ranges=[],
            formulas=formulas,
            vba_references=[],
            hidden=False,
            protected=False,
            dimensions={
                'max_row': sheet.nrows,
                'max_col': sheet.ncols,
                'data_cells': len(cells)
            }
        )
    
    def _process_worksheet_openpyxl(self, ws) -> WorksheetInfo:
        """Process worksheet and extract all content"""
        
        cells = []
        formulas = []
        charts = []
        pivot_tables = []
        
        # Extract all cells with data
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    # Extract cell information
                    cell_info = CellInfo(
                        address=cell.coordinate,
                        value=cell.value,
                        formula=cell.formula if hasattr(cell, 'formula') else None,
                        data_type=str(type(cell.value).__name__),
                        number_format=cell.number_format if hasattr(cell, 'number_format') else None,
                        comment=cell.comment.text if cell.comment else None,
                        hyperlink=cell.hyperlink.target if cell.hyperlink else None,
                        font_info={'name': cell.font.name, 'size': cell.font.size} if cell.font else {},
                        fill_info={'fgColor': str(cell.fill.fgColor)} if cell.fill else {},
                        is_merged=cell.coordinate in ws.merged_cells if hasattr(ws, 'merged_cells') else False,
                        merge_range=None
                    )
                    cells.append(cell_info)
                    
                    # If cell has formula, add to formulas list
                    if cell.formula:
                        formula_info = {
                            'address': cell.coordinate,
                            'formula': cell.formula,
                            'functions': self._extract_functions_from_formula(cell.formula),
                            'complexity': self._calculate_formula_complexity(cell.formula)
                        }
                        formulas.append(formula_info)
        
        # Extract charts
        if hasattr(ws, '_charts'):
            for chart in ws._charts:
                chart_info = {
                    'type': str(type(chart).__name__),
                    'title': chart.title.text if chart.title else 'Untitled',
                    'series_count': len(chart.series) if chart.series else 0
                }
                charts.append(chart_info)
        
        # Extract pivot tables
        if hasattr(ws, '_pivots'):
            for pivot in ws._pivots:
                pivot_info = {
                    'name': pivot.name if hasattr(pivot, 'name') else 'Unnamed',
                    'location': str(pivot.location) if hasattr(pivot, 'location') else 'Unknown'
                }
                pivot_tables.append(pivot_info)
        
        # Calculate dimensions
        if cells:
            dimensions = {
                'max_row': ws.max_row,
                'max_col': ws.max_column,
                'data_cells': len(cells)
            }
        else:
            dimensions = {'max_row': 0, 'max_col': 0, 'data_cells': 0}
        
        return WorksheetInfo(
            name=ws.title,
            cells=cells,
            charts=charts,
            pivot_tables=pivot_tables,
            data_validation=[],
            conditional_formatting=[],
            named_ranges=[],
            formulas=formulas,
            vba_references=[],
            hidden=ws.sheet_state == 'hidden' if hasattr(ws, 'sheet_state') else False,
            protected=ws.protection.enabled if hasattr(ws, 'protection') else False,
            dimensions=dimensions
        )
    
    def _extract_functions_from_formula(self, formula: str) -> List[str]:
        """Extract Excel functions from formula"""
        
        if not formula or not formula.startswith('='):
            return []
        
        # Common Excel functions
        excel_functions = [
            'SUM', 'AVERAGE', 'COUNT', 'MAX', 'MIN', 'IF', 'VLOOKUP', 'HLOOKUP',
            'INDEX', 'MATCH', 'CONCATENATE', 'LEFT', 'RIGHT', 'MID', 'LEN',
            'UPPER', 'LOWER', 'TRIM', 'DATE', 'TODAY', 'NOW', 'YEAR', 'MONTH',
            'DAY', 'PMT', 'PV', 'FV', 'NPV', 'IRR', 'RATE', 'NPER',
            'VAR', 'STDEV', 'CORREL', 'COVAR', 'TREND', 'FORECAST',
            'SUMIF', 'COUNTIF', 'AVERAGEIF', 'SUMIFS', 'COUNTIFS', 'AVERAGEIFS',
            'INDIRECT', 'OFFSET', 'CHOOSE', 'LOOKUP', 'IFERROR', 'ISNA', 'ISERROR'
        ]
        
        found_functions = []
        formula_upper = formula.upper()
        
        for func in excel_functions:
            if func + '(' in formula_upper:
                found_functions.append(func)
        
        return found_functions
    
    def _calculate_formula_complexity(self, formula: str) -> Dict[str, Any]:
        """Calculate formula complexity metrics"""
        
        if not formula:
            return {'score': 0, 'factors': []}
        
        complexity_score = 0
        factors = []
        
        # Base complexity
        complexity_score += len(formula) // 10
        
        # Function nesting
        nesting_level = formula.count('(') - formula.count(')')
        if nesting_level > 0:
            complexity_score += nesting_level * 2
            factors.append(f"Function nesting level: {nesting_level}")
        
        # IF statements
        if_count = formula.upper().count('IF(')
        if if_count > 0:
            complexity_score += if_count * 3
            factors.append(f"IF statements: {if_count}")
        
        # Array formulas
        if '{' in formula and '}' in formula:
            complexity_score += 5
            factors.append("Array formula")
        
        # External references
        if '!' in formula or '[' in formula:
            complexity_score += 2
            factors.append("External references")
        
        return {
            'score': complexity_score,
            'factors': factors,
            'length': len(formula)
        }
    
    def _extract_defined_names(self, wb) -> List[Dict]:
        """Extract defined names from workbook"""
        
        defined_names = []
        
        if hasattr(wb, 'defined_names'):
            for name in wb.defined_names:
                if hasattr(name, 'name') and hasattr(name, 'attr_text'):
                    defined_names.append({
                        'name': name.name,
                        'refers_to': name.attr_text,
                        'scope': 'workbook'
                    })
        
        return defined_names
    
    def _extract_custom_properties(self, wb) -> Dict:
        """Extract custom properties from workbook"""
        
        properties = {}
        
        if hasattr(wb, 'custom_doc_props'):
            for prop in wb.custom_doc_props:
                properties[prop.name] = prop.value
        
        return properties
    
    def _extract_vba_openpyxl(self, wb) -> Dict[str, str]:
        """Extract VBA code from workbook"""
        
        vba_code = {}
        
        if hasattr(wb, 'vba_archive') and wb.vba_archive:
            try:
                # This is a simplified extraction - full VBA parsing would be more complex
                vba_code['has_vba'] = True
                vba_code['archive_size'] = len(wb.vba_archive.read())
            except:
                vba_code['extraction_error'] = 'Could not extract VBA code'
        
        return vba_code
    
    def _analyze_formulas(self, worksheets: List[WorksheetInfo]) -> Dict:
        """Analyze formulas across all worksheets"""
        
        all_formulas = []
        function_usage = {}
        complexity_scores = []
        
        for ws in worksheets:
            all_formulas.extend(ws.formulas)
        
        for formula_info in all_formulas:
            # Count function usage
            for func in formula_info.get('functions', []):
                function_usage[func] = function_usage.get(func, 0) + 1
            
            # Collect complexity scores
            complexity = formula_info.get('complexity', {}).get('score', 0)
            complexity_scores.append(complexity)
        
        return {
            'total_formulas': len(all_formulas),
            'function_usage': function_usage,
            'average_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'max_complexity': max(complexity_scores) if complexity_scores else 0,
            'complexity_distribution': {
                'simple': len([s for s in complexity_scores if s < 5]),
                'moderate': len([s for s in complexity_scores if 5 <= s < 15]),
                'complex': len([s for s in complexity_scores if s >= 15])
            }
        }
    
    def _extract_business_logic(self, worksheets: List[WorksheetInfo]) -> Dict:
        """Extract business logic patterns from worksheets"""
        
        business_logic = {
            'sheet_purposes': {},
            'data_patterns': [],
            'calculation_chains': []
        }
        
        for ws in worksheets:
            # Analyze sheet name for purpose
            sheet_name = ws.name.lower()
            if any(keyword in sheet_name for keyword in ['dashboard', 'summary', 'report']):
                business_logic['sheet_purposes'][ws.name] = 'reporting'
            elif any(keyword in sheet_name for keyword in ['data', 'input', 'raw']):
                business_logic['sheet_purposes'][ws.name] = 'data_storage'
            elif any(keyword in sheet_name for keyword in ['calc', 'model', 'analysis']):
                business_logic['sheet_purposes'][ws.name] = 'calculation'
            else:
                business_logic['sheet_purposes'][ws.name] = 'general'
        
        return business_logic
    
    def _map_data_flow(self, worksheets: List[WorksheetInfo]) -> Dict:
        """Map data flow between worksheets"""
        
        data_flow = {
            'inter_sheet_references': 0,
            'external_references': 0,
            'circular_references': []
        }
        
        for ws in worksheets:
            for formula_info in ws.formulas:
                formula = formula_info.get('formula', '')
                
                # Count inter-sheet references
                if '!' in formula:
                    data_flow['inter_sheet_references'] += 1
                
                # Count external references
                if '[' in formula and ']' in formula:
                    data_flow['external_references'] += 1
        
        return data_flow
    
    def _calculate_complexity_metrics(self, analysis: ExcelAnalysis) -> Dict:
        """Calculate overall complexity metrics"""
        
        total_cells = sum(len(ws.cells) for ws in analysis.worksheets)
        total_formulas = sum(len(ws.formulas) for ws in analysis.worksheets)
        total_sheets = len(analysis.worksheets)
        
        # Calculate complexity score
        complexity_score = 0
        complexity_score += total_sheets * 2  # Sheet complexity
        complexity_score += total_formulas * 1  # Formula complexity
        complexity_score += analysis.formulas_summary.get('average_complexity', 0)  # Formula sophistication
        
        return {
            'total_cells': total_cells,
            'total_formulas': total_formulas,
            'total_sheets': total_sheets,
            'complexity_score': complexity_score,
            'formula_to_cell_ratio': total_formulas / max(total_cells, 1),
            'average_formulas_per_sheet': total_formulas / max(total_sheets, 1)
        }
    
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