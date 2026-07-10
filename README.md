# AI Technical Debt Analyzer

An AI-powered Python static code analysis tool that detects technical debt, code complexity, maintainability risks, and security issues using AST-based analysis and Machine Learning classification.

## Features

- AST-based static code analysis for Python codebases
- Machine Learning-based technical debt classification
- Cyclomatic complexity and nesting depth analysis
- Detection of long functions and large classes
- Missing docstring and unused import detection
- Exception handling and variable naming analysis
- Automated code quality score and risk assessment
- Security warning detection
- Smart refactoring recommendations with line references
- Interactive Streamlit dashboard with visual analytics
- Python file and ZIP project analysis
- Downloadable PDF analysis reports
- Educational learning mode for understanding code quality metrics

## Tech Stack

Python, Streamlit, Machine Learning, Scikit-learn, Python AST, Pandas, NumPy, Plotly, Joblib, ReportLab

## How It Works

1. Upload a Python file or ZIP archive containing Python files.
2. The application parses the source code using Python AST.
3. Static code metrics and software quality issues are extracted.
4. The Machine Learning model predicts the technical debt risk.
5. A code quality score and risk level are calculated.
6. Security warnings and refactoring recommendations are generated.
7. Results are displayed through an interactive Streamlit dashboard.
8. Users can download the analysis results as a PDF report.

## Installation

Clone the repository:

git clone <your-repository-url>

Navigate to the project directory:

cd AI-Technical-Debt-Analyzer

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Project Structure

AI-Technical-Debt-Analyzer/
- app.py - Main Streamlit application and code analysis engine
- create_model.py - Machine Learning model training script
- model.pkl - Trained Machine Learning model
- features.pkl - Model feature definitions
- requirements.txt - Python dependencies
- README.md - Project documentation

## Applications

This project can help students and software development teams analyze Python code quality, identify maintainability risks, prioritize refactoring efforts, and understand technical debt through automated static analysis and Machine Learning-based risk assessment.

## Future Improvements

- Support for Java, JavaScript, and other programming languages
- GitHub repository URL analysis
- Code duplication detection
- CI/CD pipeline integration
- Advanced security vulnerability scanning
- LLM-based automated code refactoring

## Author

K. S. Mithilesh
