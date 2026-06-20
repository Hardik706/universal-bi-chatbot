import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

def is_date_or_time_column(series: pd.Series) -> bool:
    """
    Checks if a pandas Series is likely a date, month, or time column.
    """
    # 1. Check data type
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # 2. Check name matches common date/time fields
    col_name = str(series.name).lower()
    date_keywords = ["date", "month", "year", "day", "timestamp", "created", "updated", "time", "period", "week", "quarter"]
    if any(keyword in col_name for keyword in date_keywords):
        return True
        
    # 3. Try to convert string to datetime (only if it's string object type)
    if series.dtype == 'object':
        import warnings
        try:
            # Take a sample to test parsing (first 3 non-null values)
            sample = series.dropna().head(3)
            if not sample.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    # If we can parse all sample items as dates, consider it a date col
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().all():
                        return True
        except (ValueError, TypeError):
            pass

    return False

def is_numeric_column(series: pd.Series) -> bool:
    """
    Checks if a pandas Series is numeric.
    """
    return pd.api.types.is_numeric_dtype(series)

def generate_auto_chart(df: pd.DataFrame, title: str = "Query Results View") -> Tuple[go.Figure | None, str]:
    """
    Evaluates a DataFrame and generates a Plotly chart based on specific business rules:
    - 2 columns where col 1 is date/month and col 2 is numeric -> px.line()
    - 2 columns where row count <= 8 -> px.pie()
    - 2 columns where row count > 8 -> px.bar()
    
    Returns:
        Tuple of (Plotly Figure or None, Chart Type Description String)
    """
    if df is None or df.empty:
        return None, "No data available for charting"
        
    num_rows, num_cols = df.shape
    
    # Ensure column names are clean strings
    df = df.copy()
    df.columns = [str(col) for col in df.columns]
    cols = df.columns
    
    # Fallback if there is only 1 column (e.g. single aggregate count)
    if num_cols == 1:
        # Display as a single metric or simple value, no complex chart needed
        return None, "Single column dataset (best viewed as a value or table)"

    # Rule checks for exactly 2 columns
    if num_cols == 2:
        col1, col2 = cols[0], cols[1]
        
        col1_is_date = is_date_or_time_column(df[col1])
        col2_is_numeric = is_numeric_column(df[col2])
        col1_is_numeric = is_numeric_column(df[col1])

        # 2 columns where col 1 is date/month and col 2 is numeric -> px.line()
        if col1_is_date and col2_is_numeric:
            fig = px.line(
                df, 
                x=col1, 
                y=col2, 
                title=f"{title} (Trend Line)",
                template="plotly_white",
                markers=True
            )
            # Style layout for premium look
            fig.update_layout(
                hovermode="x unified",
                title_font=dict(size=16, family="Outfit, Inter, sans-serif"),
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            )
            return fig, "line"

        # Special check: if col2 is date and col1 is numeric, swap them and draw line
        if is_date_or_time_column(df[col2]) and col1_is_numeric:
            fig = px.line(
                df, 
                x=col2, 
                y=col1, 
                title=f"{title} (Trend Line)",
                template="plotly_white",
                markers=True
            )
            fig.update_layout(
                hovermode="x unified",
                title_font=dict(size=16, family="Outfit, Inter, sans-serif")
            )
            return fig, "line"

        # 2 columns where row count <= 8 -> px.pie()
        # Ensure at least one column has numeric values to parse as values
        if num_rows <= 8 and (col2_is_numeric or col1_is_numeric):
            names_col = col1 if col2_is_numeric else col2
            values_col = col2 if col2_is_numeric else col1
            
            fig = px.pie(
                df, 
                names=names_col, 
                values=values_col, 
                title=f"{title} (Distribution)",
                template="plotly_white",
                hole=0.4 # Modern donut chart
            )
            fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
            fig.update_layout(
                title_font=dict(size=16, family="Outfit, Inter, sans-serif")
            )
            return fig, "pie"

        # 2 columns where row count > 8 -> px.bar()
        if num_rows > 8 and (col2_is_numeric or col1_is_numeric):
            x_col = col1 if col2_is_numeric else col2
            y_col = col2 if col2_is_numeric else col1
            
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col, 
                title=f"{title} (Bar Chart)",
                template="plotly_white"
            )
            fig.update_layout(
                title_font=dict(size=16, family="Outfit, Inter, sans-serif"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            )
            return fig, "bar"

    # Fallback/General logic for more than 2 columns or mismatched 2 columns
    # Try to find a date column, or a categoric column, and at least one numeric column.
    numeric_cols = [c for c in cols if is_numeric_column(df[c])]
    date_cols = [c for c in cols if is_date_or_time_column(df[c])]
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    if not numeric_cols:
        return None, "No numeric columns found to represent on chart"

    # If there is a date column, plot it as a line chart
    if date_cols and len(numeric_cols) >= 1:
        x_col = date_cols[0]
        y_col = numeric_cols[0]
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=f"{title} (Multi-column Trend)",
            template="plotly_white",
            markers=True
        )
        fig.update_layout(
            hovermode="x unified",
            title_font=dict(size=16, family="Outfit, Inter, sans-serif")
        )
        return fig, "line"

    # If there is a category column and a numeric column, plot as bar chart
    if non_numeric_cols and numeric_cols:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template="plotly_white"
        )
        fig.update_layout(
            title_font=dict(size=16, family="Outfit, Inter, sans-serif")
        )
        return fig, "bar"

    # Last resort: fallback to simple scatter or table only
    return None, "Complex multi-column dataset (fallback to raw data table)"
