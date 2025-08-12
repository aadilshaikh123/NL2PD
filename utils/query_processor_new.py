import pandas as pd
import numpy as np
import re
import ast
import sys
from io import StringIO
import contextlib
import traceback
from typing import Dict, Any, Optional


class QueryProcessor:
    """Process natural language queries using PandasAI with Groq"""
    
    def __init__(self):
        self.safe_functions = {
            # Pandas functions
            'df', 'pd', 'np', 'len', 'sum', 'mean', 'median', 'std', 'var', 'min', 'max',
            'head', 'tail', 'describe', 'info', 'shape', 'columns', 'dtypes', 'count',
            'groupby', 'sort_values', 'drop_duplicates', 'fillna', 'dropna', 'isnull',
            'value_counts', 'unique', 'nunique', 'corr', 'cov', 'quantile', 'rank',
            'rolling', 'expanding', 'resample', 'pivot_table', 'melt', 'merge', 'concat',
            'apply', 'map', 'filter', 'query', 'loc', 'iloc', 'at', 'iat', 'where',
            'select_dtypes', 'astype', 'copy', 'reset_index', 'set_index', 'reindex',
            
            # Python built-ins (safe ones)
            'abs', 'round', 'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple',
            'range', 'enumerate', 'zip', 'sorted', 'reversed', 'any', 'all', 'print',
            
            # Numpy functions
            'array', 'zeros', 'ones', 'arange', 'linspace', 'log', 'exp', 'sqrt', 'sin', 'cos'
        }
        
        # Built-in functions that should be available
        self.safe_builtins = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'classmethod', 'complex', 'dict', 'dir', 'divmod',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'getattr',
            'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
            'oct', 'ord', 'pow', 'print', 'property', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        
        self.dangerous_patterns = [
            r'import\s+', r'exec\s*\(', r'eval\s*\(', r'__.*__', r'open\s*\(',
            r'file\s*\(', r'input\s*\(', r'raw_input\s*\(', r'compile\s*\(',
            r'reload\s*\(', r'__import__', r'globals\s*\(', r'locals\s*\(',
            r'vars\s*\(', r'dir\s*\(', r'subprocess',
            r'os\.', r'sys\.', r'shutil\.', r'pickle\.', r'marshal\.',
            r'types\.', r'gc\.', r'inspect\.'
        ]
    
    def process_query(self, df: pd.DataFrame, query: str, groq_client) -> Dict[str, Any]:
        """
        Process natural language query and return pandas code and results
        
        Args:
            df: Input DataFrame
            query: Natural language query
            groq_client: Groq client instance
            
        Returns:
            Dict containing success status, result, code, and error if any
        """
        generated_code = ""
        try:
            # Generate pandas code using Groq
            generated_code = self._generate_pandas_code(df, query, groq_client)
            
            if not generated_code:
                return {
                    'success': False,
                    'error': 'Failed to generate pandas code',
                    'code': '',
                    'result': None
                }
            
            # Validate and execute the code
            result = self._execute_safe_code(df, generated_code)
            
            return {
                'success': True,
                'result': result,
                'code': generated_code,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'code': generated_code,
                'result': None
            }
    
    def _generate_pandas_code(self, df: pd.DataFrame, query: str, groq_client) -> str:
        """Generate pandas code using Groq LLM"""
        try:
            # Create context about the DataFrame
            df_info = self._get_dataframe_context(df)
            
            # Create the prompt
            prompt = self._create_prompt(df_info, query)
            
            # Get response from Groq
            response = groq_client.generate_code(prompt)
            
            # Extract and clean the code
            code = self._extract_code_from_response(response)
            
            return code
            
        except Exception as e:
            raise Exception(f"Error generating code: {str(e)}")
    
    def _get_dataframe_context(self, df: pd.DataFrame) -> str:
        """Create context information about the DataFrame"""
        try:
            context = f"""
DataFrame Information:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {dict(df.dtypes)}
- Sample data (first 3 rows):
{df.head(3).to_string()}

Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}
Categorical columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()}
"""
            return context
        except Exception as e:
            return f"Error getting DataFrame context: {str(e)}"
    
    def _create_prompt(self, df_info: str, query: str) -> str:
        """Create prompt for the LLM"""
        prompt = f"""
You are a pandas expert. Given a DataFrame with the following information:

{df_info}

User Query: "{query}"

Generate ONLY the pandas code to answer this query. The DataFrame is already loaded as 'df'.

Requirements:
1. Return ONLY executable pandas code
2. Use 'df' as the DataFrame variable name
3. The code should return a result (DataFrame, Series, or scalar value)
4. Do not include any imports or DataFrame loading code
5. Do not use any file operations, system calls, or dangerous functions
6. Keep the code simple and focused on the specific query
7. If creating visualizations, return the data for plotting, not the plot itself
8. For sorting operations, use df.sort_values() with ascending=False for highest values
9. For filtering by multiple criteria, use boolean indexing with & operator
10. Always use .head(n) to limit results when asked for "top N" items

Examples:
- For "show first 5 rows": df.head(5)
- For "average of column X": df['X'].mean()
- For "group by category": df.groupby('category').sum()
- For "top 5 highest prices": df.nlargest(5, 'price')
- For "lowest prices": df.nsmallest(5, 'price')
- For "top 5 with highest rating and lowest price": df.sort_values(['rating', 'price'], ascending=[False, True]).head(5)

Your code:
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean pandas code from LLM response"""
        try:
            # Remove common markdown code block indicators
            code = response.strip()
            
            # Remove markdown code blocks
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            
            if code.endswith('```'):
                code = code[:-3]
            
            # Clean up the code
            code = code.strip()
            
            # Remove any import statements or dangerous code
            lines = code.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not any(re.search(pattern, line, re.IGNORECASE) for pattern in self.dangerous_patterns):
                    clean_lines.append(line)
            
            # Join lines and ensure it's a single expression if possible
            clean_code = '\n'.join(clean_lines)
            
            # If multiple lines, wrap in parentheses for safe evaluation
            if '\n' in clean_code.strip():
                # For multi-line code, return as-is but validate each line
                return clean_code
            
            return clean_code
            
        except Exception as e:
            raise Exception(f"Error extracting code: {str(e)}")
    
    def _execute_safe_code(self, df: pd.DataFrame, code: str) -> Any:
        """Safely execute pandas code"""
        try:
            # Create safe built-ins dictionary
            safe_builtins_dict = {}
            for name in self.safe_builtins:
                if hasattr(__builtins__, name):
                    safe_builtins_dict[name] = getattr(__builtins__, name)
                elif hasattr(__builtins__, '__dict__') and name in __builtins__.__dict__:
                    safe_builtins_dict[name] = __builtins__.__dict__[name]
                elif name in globals():
                    safe_builtins_dict[name] = globals()[name]
            
            # Create a restricted environment
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                '__builtins__': safe_builtins_dict
            }
            
            # Additional safety check
            if any(re.search(pattern, code, re.IGNORECASE) for pattern in self.dangerous_patterns):
                raise Exception("Code contains potentially dangerous operations")
            
            # Capture stdout for operations that print
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Execute the code
                result = eval(code, safe_globals, {})
                
                # If result is None, check if there was printed output
                if result is None:
                    printed_output = captured_output.getvalue()
                    if printed_output.strip():
                        result = printed_output.strip()
                
                return result
                
            finally:
                sys.stdout = old_stdout
            
        except SyntaxError as e:
            # Try executing as statements instead of expression
            try:
                # Create safe built-ins dictionary
                safe_builtins_dict = {}
                for name in self.safe_builtins:
                    if hasattr(__builtins__, name):
                        safe_builtins_dict[name] = getattr(__builtins__, name)
                    elif hasattr(__builtins__, '__dict__') and name in __builtins__.__dict__:
                        safe_builtins_dict[name] = __builtins__.__dict__[name]
                    elif name in globals():
                        safe_builtins_dict[name] = globals()[name]
                
                safe_globals = {
                    'df': df,
                    'pd': pd,
                    'np': np,
                    '__builtins__': safe_builtins_dict
                }
                
                # Create a local namespace to capture results
                local_namespace = {}
                exec(code, safe_globals, local_namespace)
                
                # Return the last expression if any
                if local_namespace:
                    return list(local_namespace.values())[-1]
                else:
                    return "Code executed successfully (no return value)"
                    
            except Exception as exec_error:
                raise Exception(f"Syntax error in generated code: {str(e)}")
        
        except Exception as e:
            raise Exception(f"Error executing code: {str(e)}")
    
    def _validate_result(self, result: Any) -> Any:
        """Validate and format the result"""
        try:
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # Limit result size for display
                if hasattr(result, 'shape') and len(result) > 1000:
                    return result.head(1000)
                return result
            
            elif isinstance(result, (int, float, str, bool, type(None))):
                return result
            
            elif isinstance(result, (list, tuple, dict)):
                return str(result)
            
            else:
                return str(result)
                
        except Exception as e:
            return f"Error formatting result: {str(e)}"
