#!/usr/bin/env python3
"""
AI4PAIN DataFrame Viewer
------------------------
Creates a simple HTML view of the DataFrame and saves it to the results folder.
No terminal output.

Usage:
    python view_dataframe.py [--file PATH_TO_CSV]
"""

import argparse
import os
import pandas as pd
import webbrowser

def load_feature_data(file_path="results/features_complete.csv"):
    """
    Load feature data from CSV file.
    
    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file containing feature data.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the feature data.
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception:
        return None

def create_simple_html(df):
    """
    Create a simple HTML representation of the DataFrame without using styling.
    Shows ALL rows in the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert to HTML.
        
    Returns
    -------
    str
        HTML string of the DataFrame.
    """
    # Get the number of rows and columns
    num_rows, num_cols = df.shape
    
    # Convert the entire dataframe to HTML
    full_html = df.to_html(index=True)
    
    # Create column info
    column_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': [f"{df[col].count()} non-null" for col in df.columns],
        'Dtype': [str(df[col].dtype) for col in df.columns]
    })
    column_info_html = column_info.to_html(index=False)
    
    # Create the HTML content with basic styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DataFrame Viewer - All Rows</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }}
            .container {{
                max-width: 98%;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .summary {{
                position: sticky;
                top: 0;
                background: white;
                padding: 10px;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
                z-index: 100;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="summary">
                <h1>DataFrame Viewer - All Rows</h1>
                <p><strong>Dimensions:</strong> {num_rows} rows × {num_cols} columns</p>
            </div>
            
            <h2>All Rows</h2>
            {full_html}
            
            <div class="column-info">
                <h2>Column Information</h2>
                {column_info_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def view_dataframe(df):
    """
    Save the DataFrame as HTML and open it in a browser.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to view.
    """
    if df is None or len(df) == 0:
        return
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Path for the HTML file
    html_file = os.path.abspath("results/dataframe_view.html")
    
    # Create the HTML content
    html_content = create_simple_html(df)
    
    # Write the HTML file
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Open the HTML file in the default browser
    webbrowser.open('file://' + html_file, new=2)

def main():
    """Main function to load data and view dataframe."""
    # No terminal output - redirect to devnull
    import sys
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='AI4PAIN DataFrame Viewer')
        parser.add_argument('--file', type=str, default='results/features_complete.csv',
                            help='Path to the CSV file containing feature data')
        args = parser.parse_args()
        
        # Load data
        df = load_feature_data(args.file)
        if df is None:
            return
        
        # View dataframe in browser
        view_dataframe(df)
    finally:
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
