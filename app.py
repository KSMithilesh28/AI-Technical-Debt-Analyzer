import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ast
import re
from collections import defaultdict, Counter
from pathlib import Path
import zipfile
import io
import datetime
from fpdf import FPDF
import tempfile
from typing import Dict, List, Tuple, Set

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Technical Debt Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ... (omitted styling)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
    <div class="header-container">
        <h1 class="header-title">� AI Technical Debt Analyzer</h1>
        <p class="header-subtitle">AI-Powered Python Analysis for Students & Teams</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Header
    st.markdown('<p style="font-size: 2rem; font-weight: 900; color: #1a73e8; text-align: center; margin-bottom: 5px;">🔍 CODE ANALYZER</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.85rem; color: #666; text-align: center; margin-bottom: 20px;">Smart Detection & Suggestions</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 📚 How It Works
    with st.expander("📚 How It Works", expanded=True):
        st.markdown("""
        **1. Upload Code**
        Upload a Python file or ZIP archive
        
        **2. Instant Analysis**
        AST parsing reveals your code structure
        
        **3. Risk Assessment**
        ML model predicts technical debt
        
        **4. Smart Suggestions**
        Learn where to improve and why
        
        **5. Educational Mode**
        Detailed explanations for learning
        """)
    
    st.markdown("---")
    
    # ⚙️ Analysis Settings
    st.markdown('<p style="font-size: 1.1rem; font-weight: 700; color: #1a73e8;">⚙️ Analysis Settings</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        educational_mode = st.checkbox("📚 Learn Mode", value=False, help="Show detailed explanations")
    with col2:
        show_details = st.checkbox("🔧 Details", value=True, help="Display technical details")
    
    # Complexity Threshold
    complexity_threshold = st.slider(
        "🔄 Complexity Threshold",
        min_value=1,
        max_value=20,
        value=10,
        help="Mark functions with complexity > threshold as HIGH RISK"
    )
    
    # Function Size Threshold
    func_size_threshold = st.slider(
        "📏 Function Size Threshold",
        min_value=10,
        max_value=200,
        value=50,
        help="Mark functions with more lines as large"
    )
    
    st.markdown("---")
    
    # 💡 Code Quality Standards
    with st.expander("💡 Code Quality Standards"):
        st.markdown("""
        **Cyclomatic Complexity**
        - 🟢 1-5: Low (Good)
        - 🟡 6-10: Medium (Fixable)
        - 🔴 11+: High (Refactor needed)
        
        **Function Length**
        - 🟢 <50 lines: Good
        - 🟡 50-100 lines: Medium
        - 🔴 >100 lines: Consider breaking down
        
        **Nesting Depth**
        - 🟢 1-3 levels: Good
        - 🟡 4-5 levels: Medium
        - 🔴 6+ levels: Too deep
        """)
    
    st.markdown("---")
    
    # 🎯 Quick Tips
    with st.expander("🎯 Quick Tips"):
        st.markdown("""
        ✅ **Best Practices**
        - Keep functions small and focused
        - Reduce cyclomatic complexity
        - Add docstrings to all functions
        - Use meaningful variable names
        - Avoid deep nesting
        
        📌 **What to Improve**
        - High complexity functions
        - Long function lengths
        - Missing documentation
        - Deep nesting levels
        """)
    
    st.markdown("---")
    
    # 📊 Analysis Info
    with st.expander("📊 What Gets Analyzed", expanded=False):
        st.markdown("""
        ✓ **Complexity Metrics**
        - Cyclomatic complexity per function
        - Code nesting depth
        - Function length analysis
        
        ✓ **Code Structure**
        - Classes and methods
        - Imports and dependencies
        - Docstring coverage
        
        ✓ **Risk Assessment**
        - ML-based technical debt detection
        - Risk level predictions
        - Improvement suggestions
        """)
    
    st.markdown("---")
    
    # 🚀 Features
    with st.expander("🚀 Key Features"):
        st.markdown("""
        ✨ **Smart Detection**
        - AST-based code parsing
        - ML risk prediction
        
        💡 **Helpful Guidance**
        - Line-by-line suggestions
        - Educational explanations
        - Best practice tips
        
        📦 **Flexible Input**
        - Single Python files
        - ZIP archives
        - Multiple file support
        """)
    
    st.markdown("---")
    st.markdown('<p style="font-size: 0.9rem; color: #666; text-align: center;">v3.0 | Smart Code Analyzer</p>', unsafe_allow_html=True)

# ============================================================================
# EDUCATIONAL CONTENT
# ============================================================================

EDUCATIONAL_CONTENT = {
    "complexity": """
### 🔄 Cyclomatic Complexity
Cyclomatic complexity is a quantitative measure of the number of linearly independent paths through a program's source code.
- **1-5 (Low):** Simple code, easy to test and maintain.
- **6-10 (Medium):** Moderate complexity, consider simplifying.
- **11+ (High):** Very complex, high risk of bugs, should be refactored into smaller functions.
""",
    "nesting": """
### 🪆 Nesting Depth
Nesting depth refers to how many levels of conditional statements (if, while, for, try) are layered inside each other.
- **1-3 levels:** Readable and maintainable.
- **4-5 levels:** Starting to get difficult to follow.
- **6+ levels:** "Arrow code" pattern; very hard to read. Use guard clauses or extract methods to flatten the structure.
""",
    "functions": """
### 📏 Function size
Function size (lines of code) is a direct indicator of whether a function is trying to do too much.
- **< 50 lines:** Ideal size for most functions.
- **50-100 lines:** Consider if the function can be split.
- **100+ lines:** Likely violates the Single Responsibility Principle. Refactor into smaller, more focused functions.
""",
    "docs": """
### 📋 Documentation Importance
Docstrings and comments are essential for maintainability and collaboration.
- **Docstrings:** Explain *what* a function or class does and its parameters/return values.
- **Comments:** Explain *why* a complex piece of logic exists.
- **Standard:** Every public class and function should have a docstring (PEP 257).
"""
}

# ============================================================================
# AST-BASED CODE ANALYSIS ENGINE
# ============================================================================

class CodeAnalyzer:
    """Advanced code analysis using Python AST"""
    
    def __init__(self, code: str, filename: str = "code.py", complexity_threshold: int = 10, func_size_threshold: int = 50):
        self.code = code
        self.filename = filename
        self.lines = code.split("\n")
        self.tree = None
        self.metrics = {}
        self.complexity_threshold = complexity_threshold
        self.func_size_threshold = func_size_threshold
        self._parse_code()
    
    def _parse_code(self):
        """Parse code into AST"""
        try:
            self.tree = ast.parse(self.code)
        except SyntaxError as e:
            self.metrics['parse_error'] = str(e)
            self.tree = None
    
    def calculate_cyclomatic_complexity(self) -> Dict[str, int]:
        """Calculate cyclomatic complexity for functions"""
        if not self.tree:
            return {}
        
        complexity = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        cc += 1
                    if isinstance(child, ast.BoolOp):
                        cc += len(child.values) - 1
                
                complexity[node.name] = {
                    'value': cc,
                    'line': node.lineno,
                    'risk': 'HIGH' if cc > self.complexity_threshold else 'MEDIUM' if cc > (self.complexity_threshold // 2) else 'LOW'
                }
        
        return complexity
    
    def calculate_nesting_depth(self) -> Dict[str, int]:
        """Calculate maximum nesting depth"""
        if not self.tree:
            return {}
        
        nesting = {}
        
        class DepthVisitor(ast.NodeVisitor):
            def __init__(self):
                self.depth = 0
                self.max_depth = 0
                self.max_depth_line = 0
            
            def visit(self, node):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    self.depth += 1
                    self.max_depth = max(self.max_depth, self.depth)
                    if self.depth > 3:
                        self.max_depth_line = node.lineno
                self.generic_visit(node)
                if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    self.depth -= 1
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visitor = DepthVisitor()
                visitor.visit(node)
                if visitor.max_depth > 0:
                    nesting[node.name] = {
                        'value': visitor.max_depth,
                        'line': visitor.max_depth_line or node.lineno,
                        'risk': 'HIGH' if visitor.max_depth > 5 else 'MEDIUM' if visitor.max_depth > 3 else 'LOW'
                    }
        
        return nesting
    
    def find_long_functions(self, threshold: int = None) -> List[Dict]:
        """Find functions that exceed line count threshold"""
        if threshold is None:
            threshold = self.func_size_threshold
        if not self.tree:
            return []
        
        long_funcs = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_lines = node.end_lineno - node.lineno + 1
                if func_lines > threshold:
                    long_funcs.append({
                        'name': node.name,
                        'lines': func_lines,
                        'lineno': node.lineno,
                        'risk': 'HIGH' if func_lines > (threshold * 2) else 'MEDIUM'
                    })
        
        return sorted(long_funcs, key=lambda x: x['lines'], reverse=True)
    
    def find_large_classes(self, threshold: int = 300) -> List[Dict]:
        """Find classes that exceed line count threshold"""
        if not self.tree:
            return []
        
        large_classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_lines = node.end_lineno - node.lineno + 1
                method_count = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                if class_lines > threshold:
                    large_classes.append({
                        'name': node.name,
                        'lines': class_lines,
                        'methods': method_count,
                        'lineno': node.lineno,
                        'risk': 'HIGH' if class_lines > 500 else 'MEDIUM'
                    })
        
        return sorted(large_classes, key=lambda x: x['lines'], reverse=True)
    
    def find_missing_docstrings(self) -> List[Dict]:
        """Find functions/classes without docstrings"""
        if not self.tree:
            return []
        
        missing = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    missing.append({
                        'name': node.name,
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'lineno': node.lineno
                    })
        
        return missing
    
    def find_unused_imports(self) -> List[str]:
        """Find unused imports"""
        if not self.tree:
            return []
        
        imported = set()
        used = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.Name):
                used.add(node.id)
        
        return list(imported - used)
    
    def check_exception_handling(self) -> Dict:
        """Check exception handling patterns"""
        if not self.tree:
            return {'bare_except': 0, 'broad_except': 0, 'proper_except': 0}
        
        bare_excepts = 0
        broad_excepts = 0
        proper_excepts = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts += 1
                elif isinstance(node.type, ast.Name) and node.type.id in ('Exception', 'BaseException'):
                    broad_excepts += 1
                else:
                    proper_excepts += 1
        
        return {
            'bare_except': bare_excepts,
            'broad_except': broad_excepts,
            'proper_except': proper_excepts
        }
    
    def analyze_variable_naming(self) -> Dict:
        """Analyze variable naming quality"""
        if not self.tree:
            return {'bad_names': [], 'total_vars': 0}
        
        bad_patterns = ['x', 'y', 'z', 'temp', 'tmp', 'data', 'obj', 'val', 'var', 'item']
        bad_names = []
        total_vars = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                total_vars += 1
                if node.id in bad_patterns or bool(re.match(r'^[a-z]\d+$', node.id)):
                    bad_names.append(node.id)
        
        return {
            'bad_names': list(set(bad_names)),
            'total_vars': total_vars,
            'quality_score': max(0, 100 - (len(set(bad_names)) * 5))
        }
    
    def scan_security(self) -> List[Dict]:
        """Scan for security risks using AST and regex"""
        security_risks = []
        if not self.tree:
            return []
            
        # 1. AST-based detection
        for node in ast.walk(self.tree):
            # Check for eval() and exec()
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec'):
                        security_risks.append({
                            'type': 'Critical',
                            'message': f"Security Risk: {node.func.id}() usage detected",
                            'line': node.lineno
                        })
                    elif node.func.id == 'system' and isinstance(getattr(node.func, 'value', None), ast.Name) and node.func.value.id == 'os':
                         security_risks.append({
                            'type': 'Warning',
                            'message': "Security Risk: os.system() usage detected",
                            'line': node.lineno
                        })
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'system' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                        security_risks.append({
                            'type': 'Warning',
                            'message': "Security Risk: os.system() usage detected",
                            'line': node.lineno
                        })

            # Check for bare except blocks
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    security_risks.append({
                        'type': 'Warning',
                        'message': "Security Risk: Bare except block detected",
                        'line': node.lineno
                    })

        # 2. Regex-based detection for hardcoded passwords
        password_pattern = re.compile(r'(password|passwd|pwd|secret|token|api_key)\s*=\s*["\'](.*?)["\']', re.IGNORECASE)
        for i, line in enumerate(self.lines):
            match = password_pattern.search(line)
            if match:
                # Basic check to avoid common false positives like empty strings if needed
                if len(match.group(2)) > 0:
                    security_risks.append({
                        'type': 'Critical',
                        'message': "Security Risk: Hardcoded credential found",
                        'line': i + 1
                    })
        
        return security_risks
    
    def full_analysis(self) -> Dict:
        """Run complete analysis"""
        if not self.tree:
            return {'error': 'Failed to parse code'}
        
        return {
            'cyclomatic_complexity': self.calculate_cyclomatic_complexity(),
            'nesting_depth': self.calculate_nesting_depth(),
            'long_functions': self.find_long_functions(),
            'large_classes': self.find_large_classes(),
            'missing_docstrings': self.find_missing_docstrings(),
            'unused_imports': self.find_unused_imports(),
            'exception_handling': self.check_exception_handling(),
            'variable_naming': self.analyze_variable_naming(),
            'security_scan': self.scan_security()
        }

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_resources():
    """Load trained model and feature columns with error handling"""
    script_dir = Path(__file__).parent
    model = None
    feature_columns = ["lines", "loops", "functions", "comments"]  # Default fallback
    
    try:
        model_path = script_dir / "model.pkl"
        features_path = script_dir / "features.pkl"
        
        if model_path.exists() and features_path.exists():
            model = joblib.load(model_path)
            feature_columns = joblib.load(features_path)
            # Ensure feature_columns is a list
            if hasattr(feature_columns, 'tolist'):
                feature_columns = feature_columns.tolist()
        else:
            # Files missing - simpler warning (optional, or just rely on None check later)
            pass 
            
    except Exception as e:
        # Error loading files
        st.warning(f"⚠️ Warning: Could not load ML model ({str(e)}). Using rule-based analysis only.")
        
    return model, feature_columns

model, feature_columns = load_model_resources()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_advanced_score(metrics, ast_analysis, ml_prediction):
    """Calculate overall technical debt score"""
    score = 100
    details = {}
    
    # Complexity penalties
    cc_values = [v['value'] for v in ast_analysis['cyclomatic_complexity'].values()]
    if cc_values:
        avg_cc = sum(cc_values) / len(cc_values)
        details['avg_complexity'] = avg_cc
        if avg_cc > 10:
            score -= 20
        elif avg_cc > 5:
            score -= 10
        elif avg_cc > 3:
            score -= 5
            
    # Length penalties
    if metrics['lines'] > 500:
        score -= 15
    elif metrics['lines'] > 200:
        score -= 10
        
    # Nesting penalties
    max_depth = max((v['value'] for v in ast_analysis['nesting_depth'].values()), default=0)
    details['max_depth'] = max_depth
    if max_depth > 5:
        score -= 15
    elif max_depth > 3:
        score -= 5
        
    # Docstring penalties
    missing_docs = len(ast_analysis['missing_docstrings'])
    total_items = metrics['functions'] + metrics['classes']
    if total_items > 0:
        doc_ratio = (total_items - missing_docs) / total_items
        if doc_ratio < 0.5:
            score -= 10
        elif doc_ratio < 0.8:
            score -= 5
            
    # ML Adjustment
    if ml_prediction != -1:
        # Assuming ML prediction is 0 (low risk) or 1 (high risk)
        if ml_prediction == 1:
            score -= 15
            details['ml_risk'] = "High"
        else:
            details['ml_risk'] = "Low"
    else:
        details['ml_risk'] = "Unknown"
            
    # Cap score
    score = max(0, min(100, score))
    
    return {
        'overall_score': score,
        'details': details,
        'risk_level': 'LOW' if score >= 75 else 'MEDIUM' if score >= 50 else 'HIGH',
        'urgency': 'Low' if score >= 75 else 'Moderate' if score >= 50 else 'Critical'
    }

def generate_smart_suggestions(ast_analysis, metrics, score_data):
    """Generate actionable suggestions"""
    suggestions = []
    
    # Complexity suggestions
    for func_name, cc_info in ast_analysis['cyclomatic_complexity'].items():
        if cc_info['value'] > 10:
            suggestions.append(f"🔴 Function '{func_name}' has high complexity ({cc_info['value']}). Break into smaller functions. Line {cc_info['line']}")
        elif cc_info['value'] > 5:
            suggestions.append(f"🟡 Function '{func_name}' is moderately complex ({cc_info['value']}). Simplify logic. Line {cc_info['line']}")
            
    # Length suggestions
    for func in ast_analysis['long_functions']:
        suggestions.append(f"🔴 Function '{func['name']}' is too long ({func['lines']} lines). Refactor to < {50} lines. Line {func['lineno']}")
        
    # Nesting suggestions
    for func_name, nest_info in ast_analysis['nesting_depth'].items():
        if nest_info['value'] > 3:
            suggestions.append(f"🟡 Function '{func_name}' has deep nesting ({nest_info['value']} levels). Flatten structure. Line {nest_info['line']}")
            
    # Docstring suggestions
    if len(ast_analysis['missing_docstrings']) > 0:
        count = len(ast_analysis['missing_docstrings'])
        suggestions.append(f"📋 Add docstrings to {count} functions/classes to improve maintainability.")
        
    # General score-based suggestions
    if score_data['overall_score'] < 50:
        suggestions.append("⚠️ Overall quality is low. Consider a major refactoring sprint.")
        
    if not suggestions:
        suggestions.append("✅ Great job! Your code follows best practices.")
        
    return suggestions

def generate_pdf_report(all_results):
    """Generate a PDF report from analysis results"""
    # fpdf2 version check and enums for better control
    try:
        from fpdf.enums import XPos, YPos
    except ImportError:
        # Fallback for older fpdf2 versions or original fpdf
        XPos, YPos = None, None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for result in all_results:
        pdf.add_page()
        # Use effective page width (epw) or fallback to 190mm
        effective_width = getattr(pdf, 'epw', 190)
        
        # Header
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(26, 115, 232) # Blue
        pdf.cell(effective_width, 15, "Code Analysis Report", ln=True, align='C')
        
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(effective_width, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(5)
        
        # File Info
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(effective_width, 10, f"File: {result['file']}", ln=True)
        pdf.set_draw_color(26, 115, 232)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Metrics
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(effective_width, 10, "Code Metrics", ln=True)
        pdf.set_font("Arial", '', 11)
        metrics = result['metrics']
        col_width = effective_width / 4
        pdf.cell(col_width, 8, f"Lines: {metrics['lines']}")
        pdf.cell(col_width, 8, f"Functions: {metrics['functions']}")
        pdf.cell(col_width, 8, f"Loops: {metrics['loops']}")
        pdf.cell(col_width, 8, f"Classes: {metrics['classes']}")
        pdf.ln(10)
        
        # Quality Score
        score_data = result['score']
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(effective_width, 10, "Quality Score", ln=True)
        
        score = int(score_data['overall_score'])
        status = "Excellent" if score >= 75 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
        
        pdf.set_font("Arial", 'B', 16)
        if score >= 75: pdf.set_text_color(76, 175, 80)
        elif score >= 40: pdf.set_text_color(255, 152, 0)
        else: pdf.set_text_color(244, 67, 54)
        
        pdf.cell(effective_width, 10, f"{score}/100 ({status})", ln=True)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font("Arial", '', 11)
        pdf.cell(effective_width, 8, f"Risk Level: {score_data['risk_level']}", ln=True)
        pdf.cell(effective_width, 8, f"Urgency: {score_data['urgency']}", ln=True)
        pdf.ln(5)
        
        # Security
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(effective_width, 10, "Security Assessment", ln=True)
        pdf.set_font("Arial", '', 11)
        security_risks = result['ast_analysis'].get('security_scan', [])
        if security_risks:
            for risk in security_risks:
                pdf.set_text_color(244, 67, 54) if risk['type'] == 'Critical' else pdf.set_text_color(255, 152, 0)
                # Use explicit width and reset X position
                pdf.multi_cell(effective_width, 8, f"- [{risk['type']}] {risk['message']} (Line {risk['line']})")
                pdf.set_x(10)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_text_color(76, 175, 80)
            pdf.cell(effective_width, 8, "No major security risks detected.", ln=True)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        # Recommendations
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(effective_width, 10, "Smart Recommendations", ln=True)
        pdf.set_font("Arial", '', 11)
        for sug in result['suggestions']:
            # Clean up icons for PDF
            clean_sug = sug.replace("🔴 ", "").replace("🟡 ", "").replace("📋 ", "").replace("⚠️ ", "").replace("✅ ", "")
            pdf.multi_cell(effective_width, 8, f"- {clean_sug}")
            pdf.set_x(10)
            
    # Explicitly return bytes to avoid Streamlit marshalling issues
    output = pdf.output()
    return bytes(output) if isinstance(output, (bytearray, bytes)) else str(output).encode('latin-1')


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 📤 Upload Python File(s)")
    st.markdown("*Upload .py file or ZIP with multiple Python files*")
    uploaded_file = st.file_uploader(
        "Choose file",
        type=["py", "zip"],
        label_visibility="collapsed"
    )

st.divider()

# ============================================================================
# ANALYSIS
# ============================================================================

if uploaded_file:
    # Handle single file or ZIP
    if uploaded_file.name.endswith('.py'):
        code = uploaded_file.read().decode("utf-8")
        files_to_analyze = [(uploaded_file.name, code)]
    elif uploaded_file.name.endswith('.zip'):
        st.info("📦 Reading ZIP file...")
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            files_to_analyze = []
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.py'):
                    code = zip_ref.read(file_info.filename).decode("utf-8")
                    files_to_analyze.append((file_info.filename, code))
        
        if not files_to_analyze:
            st.error("❌ No Python files found in ZIP")
            st.stop()
    
    all_results = []
    
    for file_name, code in files_to_analyze:
        st.markdown(f"### 📊 Analysis: `{file_name}`")
        
        # Comprehensive analysis
        analyzer = CodeAnalyzer(code, file_name, complexity_threshold, func_size_threshold)
        ast_analysis = analyzer.full_analysis()
        
        # Basic metrics
        metrics = {
            'lines': len(code.split("\n")),
            'loops': code.count("for") + code.count("while"),
            'functions': code.count("def"),
            'classes': code.count("class"),
            'comments': code.count("#"),
            'docstrings': code.count('"""') // 2 + code.count("'''") // 2
        }
        
        # ML prediction
        ml_prediction = -1  # Default safe value
        if model is not None:
            try:
                # Ensure feature columns are available even if model is loaded but features.pkl failed (unlikely but safe)
                safe_features = feature_columns if feature_columns else ["lines", "loops", "functions", "comments"]
                
                input_data = pd.DataFrame([{
                    safe_features[0]: int(metrics['lines']) if len(safe_features) > 0 else 0,
                    safe_features[1]: int(metrics['loops']) if len(safe_features) > 1 else 0,
                    safe_features[2]: int(metrics['functions']) if len(safe_features) > 2 else 0,
                    safe_features[3]: int(metrics['comments']) if len(safe_features) > 3 else 0,
                }])
                prediction = model.predict(input_data)[0]
                ml_prediction = int(prediction) # Cast to float/int to avoid numpy types
            except Exception as e:
                st.warning(f"⚠️ ML prediction failed for {file_name}: {e}")
                ml_prediction = -1
        
        # Advanced score
        score_data = calculate_advanced_score(metrics, ast_analysis, ml_prediction)
        suggestions = generate_smart_suggestions(ast_analysis, metrics, score_data)
        
        all_results.append({
            'file': file_name,
            'metrics': metrics,
            'score': score_data,
            'suggestions': suggestions,
            'ast_analysis': ast_analysis
        })
        
    # ====================================================================
    # GLOBAL DOWNLOAD BUTTON
    # ====================================================================
    if all_results:
        try:
            pdf_data = generate_pdf_report(all_results)
            st.download_button(
                label="📄 Download Analysis Report",
                data=pdf_data,
                file_name=f"Code_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"⚠️ Could not generate PDF report: {e}")
        st.divider()

    for result in all_results:
        file_name = result['file']
        metrics = result['metrics']
        score_data = result['score']
        suggestions = result['suggestions']
        ast_analysis = result['ast_analysis']
        
        st.markdown(f"### 📊 Analysis: `{file_name}`")
        
        # ====================================================================
        # METRICS DISPLAY
        # ====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Lines", metrics['lines'])
        with col2:
            st.metric("Functions", metrics['functions'])
        with col3:
            st.metric("Loops", metrics['loops'])
        with col4:
            st.metric("Classes", metrics['classes'])
        with col5:
            st.metric("Comments", metrics['comments'])
        
        st.divider()
        
        # ====================================================================
        # QUALITY SCORE
        # ====================================================================
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### 🎯 Code Quality Score")
            score = score_data['overall_score']
            risk = score_data['risk_level']
            
            if score >= 75: bar_color, status = "#4caf50", "Excellent"
            elif score >= 60: bar_color, status = "#ff9800", "Good"
            elif score >= 40: bar_color, status = "#ff6b6b", "Fair"
            else: bar_color, status = "#f44336", "Poor"
            
            st.markdown(f"""
            <div style="margin: 20px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="font-size: 1.3rem; font-weight: 700;">{status}</span>
                    <span style="font-size: 1.6rem; font-weight: 900; color: {bar_color};">{score}/100</span>
                </div>
                <div style="width: 100%; height: 35px; background: #1e1e2e; border-radius: 15px; overflow: hidden; border: 2px solid {bar_color};">
                    <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {bar_color} 0%, {bar_color}dd 100%);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚡ Urgency")
            urgency = score_data['urgency']
            urgency_color = "#4caf50" if urgency == "Low" else "#ff9800" if urgency == "Moderate" else "#f44336"
            st.markdown(f"""
            <div style="background: {urgency_color}33; border: 2px solid {urgency_color}; padding: 20px; border-radius: 12px; text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 900; color: {urgency_color};">{urgency}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # RISK ASSESSMENT
        # ====================================================================
        st.markdown("### 🚨 Risk Assessment")
        
        if risk == "LOW":
            st.markdown("""
            <div class="info-box risk-low">
            <b>✅ LOW RISK</b><br>
            Excellent code quality with good structure, documentation, and low complexity.
            </div>
            """, unsafe_allow_html=True)
        elif risk == "MEDIUM":
            st.markdown("""
            <div class="info-box risk-medium">
            <b>⚠️ MEDIUM RISK</b><br>
            Several quality concerns. Review suggestions below for improvements.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box risk-high">
            <b>🔴 HIGH RISK</b><br>
            Significant technical debt. Implement refactoring recommendations.
            </div>
            """, unsafe_allow_html=True)
        
        # ====================================================================
        # SECURITY WARNINGS
        # ====================================================================
        
        st.divider()
        st.markdown("### 🛡️ Security Assessment")
        
        security_scan = ast_analysis.get('security_scan', [])
        if security_scan:
            for risk in security_scan:
                risk_style = "risk-high" if risk['type'] == 'Critical' else "risk-medium"
                icon = "🔴" if risk['type'] == 'Critical' else "⚠️"
                st.markdown(f"""
                <div class="info-box {risk_style}">
                <b>{icon} {risk['message']}</b><br>
                Found on line {risk['line']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box risk-low">
            <b>✔ No major security risks detected.</b><br>
            Basic security scan completed.
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # SMART SUGGESTIONS
        # ====================================================================
        
        st.markdown("### 💡 Smart Recommendations")
        
        suggestion_cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with suggestion_cols[idx % 2]:
                st.markdown(f"""
                <div class="suggestion-box">
                {suggestion}
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # EDUCATIONAL MODE
        # ====================================================================
        
        if educational_mode:
            st.markdown("### 📚 Learning Center")
            
            edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs(
                ["Complexity", "Nesting", "Functions", "Documentation"]
            )
            
            with edu_tab1:
                st.markdown(EDUCATIONAL_CONTENT['complexity'])
            with edu_tab2:
                st.markdown(EDUCATIONAL_CONTENT['nesting'])
            with edu_tab3:
                st.markdown(EDUCATIONAL_CONTENT['functions'])
            with edu_tab4:
                st.markdown(EDUCATIONAL_CONTENT['docs'])
            
            st.divider()
        
        # ====================================================================
        # DETAILED ANALYSIS
        # ====================================================================
        
        if show_details:
            st.markdown("### 📊 Detailed Analysis")
            
            viz_col1, viz_col2 = st.columns(2)
            
            # Complexity chart
            with viz_col1:
                if ast_analysis['cyclomatic_complexity']:
                    cc_data = ast_analysis['cyclomatic_complexity']
                    # Sort by complexity for better visualization (highest at top for barh)
                    cc_df = pd.DataFrame([
                        {'Function': k, 'Complexity': v['value'], 'Risk': v['risk']}
                        for k, v in cc_data.items()
                    ]).sort_values('Complexity', ascending=True).tail(10)
                    
                    fig = px.bar(cc_df, x='Complexity', y='Function', color='Risk', orientation='h',
                                text='Complexity',
                                color_discrete_map={'LOW': '#4caf50', 'MEDIUM': '#ff9800', 'HIGH': '#f44336'},
                                title="Function Complexity (Top 10)")
                    
                    fig.update_traces(textposition='inside', textfont_size=12)
                    fig.update_layout(
                        plot_bgcolor='#1e1e2e', 
                        paper_bgcolor='#1e1e2e',
                        font=dict(color='#e0e0e0'), 
                        height=400,
                        yaxis=dict(title=None),
                        xaxis=dict(title="Cyclomatic Complexity"),
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Metrics chart
            with viz_col2:
                metric_names = ["Lines", "Functions", "Loops", "Classes"]
                metric_vals = [metrics['lines'], metrics['functions'], metrics['loops'], metrics['classes']]
                
                fig = go.Figure(data=[go.Bar(
                    x=metric_names, y=metric_vals,
                    marker=dict(color=['#1a73e8', '#4caf50', '#ff9800', '#9c27b0'])
                )])
                fig.update_layout(title="Code Metrics", plot_bgcolor='#1e1e2e',
                                paper_bgcolor='#1e1e2e', font=dict(color='#e0e0e0'),
                                showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Detailed table
            with st.expander("📋 Detailed Metrics Table"):
                detailed_data = {
                    'Metric': [
                        'Lines', 'Functions', 'Classes', 'Loops', 'Comments',
                        'Docstrings', 'Avg Complexity', 'Max Nesting',
                        'Missing Docs', 'Unused Imports'
                    ],
                    'Value': [
                        metrics['lines'], metrics['functions'], metrics['classes'],
                        metrics['loops'], metrics['comments'], metrics['docstrings'],
                        f"{sum(v['value'] for v in ast_analysis['cyclomatic_complexity'].values()) / max(len(ast_analysis['cyclomatic_complexity']), 1):.1f}",
                        f"{max((v['value'] for v in ast_analysis['nesting_depth'].values()), default=0)}",
                        len(ast_analysis['missing_docstrings']),
                        len(ast_analysis['unused_imports'])
                    ]
                }
                
                df_details = pd.DataFrame(detailed_data)
                df_details['Value'] = df_details['Value'].astype(str)
                st.dataframe(df_details, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.95rem; margin-top: 40px;">
<p>🚀 Smart Code Quality Analyzer • Powered by AST & ML</p>
<p style="color: #666; font-size: 0.9rem;">Built for students and development teams</p>
</div>
""", unsafe_allow_html=True)
