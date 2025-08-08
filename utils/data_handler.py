import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
import openpyxl

class DataHandler:
    """Handle file uploads and data loading"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def load_file(self, uploaded_file):
        """
        Load uploaded file into a pandas DataFrame
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._load_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def _load_csv(self, uploaded_file):
        """Load CSV file with automatic delimiter detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Read as string first to detect delimiter
                    string_data = StringIO(uploaded_file.getvalue().decode(encoding))
                    
                    # Try to detect delimiter
                    sample = string_data.read(1024)
                    string_data.seek(0)
                    
                    delimiter = ','
                    if sample.count(';') > sample.count(','):
                        delimiter = ';'
                    elif sample.count('\t') > sample.count(','):
                        delimiter = '\t'
                    
                    # Load the DataFrame
                    df = pd.read_csv(string_data, delimiter=delimiter)
                    
                    if not df.empty:
                        return self._clean_dataframe(df)
                        
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if encoding == encodings[-1]:  # Last encoding attempt
                        raise e
                    continue
            
            raise Exception("Could not decode file with any supported encoding")
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def _load_excel(self, uploaded_file):
        """Load Excel file"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            
            # If multiple sheets, use the first one
            sheet_names = excel_file.sheet_names
            if len(sheet_names) > 1:
                st.info(f"Multiple sheets found: {sheet_names}. Using the first sheet: '{sheet_names[0]}'")
            
            df = pd.read_excel(uploaded_file, sheet_name=0)
            return self._clean_dataframe(df)
            
        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")
    
    def _clean_dataframe(self, df):
        """Clean and prepare DataFrame"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Clean column names
            df.columns = df.columns.astype(str)
            df.columns = [col.strip() for col in df.columns]
            
            # Replace empty column names
            for i, col in enumerate(df.columns):
                if col == '' or col == 'nan' or pd.isna(col):
                    df.columns.values[i] = f'Column_{i}'
            
            # Ensure no duplicate column names
            df.columns = self._make_unique_columns(list(df.columns))
            
            # Convert object columns that look like numbers
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            
            return df
            
        except Exception as e:
            raise Exception(f"Error cleaning DataFrame: {str(e)}")
    
    def _make_unique_columns(self, columns):
        """Ensure column names are unique"""
        seen = set()
        unique_columns = []
        
        for col in columns:
            original_col = col
            counter = 1
            
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            
            seen.add(col)
            unique_columns.append(col)
        
        return unique_columns
    
    def get_data_summary(self, df):
        """Get a summary of the DataFrame"""
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            return summary
        except Exception as e:
            return {'error': str(e)}
