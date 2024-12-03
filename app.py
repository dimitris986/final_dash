from urllib.parse import parse_qs
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import seaborn as sns; sns.set()
import numpy as np
from sqlalchemy import create_engine
import pyodbc
from decimal import Decimal
from dotenv import load_dotenv
import os
import sys

# Ορισμός δικαιωμάτων χρηστών
user_permissions = {
    '10': {
        'business_units': ['ADR', 'Bulk'],
        'years': ['2023']
    },
    '1': {
        'business_units': ['ADR', 'PickUp'],
        'years': ['2022', '2023', '2024']
    },
    '2': {
        'business_units': ['PickUp'],
        'years': ['2023', '2024']
    },
    'admin': {
        'business_units': ['ADR','Bulk','PickUp'],
        'years': ['2022','2023', '2024']
    }
}

def get_filtered_sql_query(user_id):
    """Δημιουργεί το SQL query με βάση τα δικαιώματα του χρήστη"""
    if user_id not in user_permissions:
        return None
    
    permissions = user_permissions[user_id]
    
    # Μετατροπή των business units σε SQL format
    bu_list = ", ".join([f"'{bu}'" for bu in permissions['business_units']])
    years_list = ", ".join(permissions['years'])
    
    SQL_QUERY = f"""
    with bulk_budget as 
    (
    SELECT
        BusinessUnitCodeName,
        left(CalendarDateKey,4) as years,
        LEFT(RIGHT(CalendarDateKey,4),2) as months,
        round(SUM(Value),0) as budget_routes
    FROM
        _BudgetPerDay
    WHERE
        BusinessUnitCodeName = 'Bulk' 
        AND BudgetCharacteristicCodeName = 'Routes'
        AND left(CalendarDateKey,4) IN ({years_list})
    GROUP BY
        BusinessUnitCodeName, LEFT(CalendarDateKey,4), LEFT(RIGHT(CalendarDateKey,4),2)
    ),
    bulk_actual as 
    (
    SELECT 
        BusinessUnit,
        left(RouteDatekey,4) as years,
        LEFT(RIGHT(RouteDatekey,4),2) as months,
        COUNT(DISTINCT FINAL_ROUTE_ID) as actual_routes,
        SUM(Sales_Amount) as profit
    FROM
        BI_FactRoutes
    WHERE 
        BusinessUnit = N'ΒΥΤΙΟ'
        AND left(RouteDatekey,4) IN ({years_list})
    GROUP BY
        BusinessUnit, LEFT(RIGHT(RouteDatekey,4),2),left(RouteDatekey,4)
    ),
    adr_budget as 
    (
    SELECT
        BusinessUnitCodeName,
        left(CalendarDateKey,4) as years,
        LEFT(RIGHT(CalendarDateKey,4),2) as months,
        round(SUM(Value),0) as budget_routes
    FROM
        _BudgetPerDay
    WHERE
        BusinessUnitCodeName = 'ADR' 
        AND BudgetCharacteristicCodeName = 'Routes'
        AND left(CalendarDateKey,4) IN ({years_list})
    GROUP BY
        BusinessUnitCodeName, LEFT(CalendarDateKey,4), LEFT(RIGHT(CalendarDateKey,4),2)
    ),
    adr_actual as 
    (
    SELECT 
        BusinessUnit,
        left(RouteDatekey,4) as years,
        LEFT(RIGHT(RouteDatekey,4),2) as months,
        COUNT(DISTINCT FINAL_ROUTE_ID) as actual_routes,
        SUM(Sales_Amount) as profit
    FROM
        BI_FactRoutes
    WHERE 
        BusinessUnit = N'ADR ΒΥΤΙΟ'
        AND left(RouteDatekey,4) IN ({years_list})
    GROUP BY
        BusinessUnit, LEFT(RIGHT(RouteDatekey,4),2),left(RouteDatekey,4)
    ),
    pickup_budget as
    (
    SELECT
        BusinessUnitCodeName,
        left(CalendarDateKey,4) as years,
        LEFT(RIGHT(CalendarDateKey,4),2) as months,
        round(SUM(Value),0) as budget_routes
    FROM
        _BudgetPerDay
    WHERE
        BusinessUnitCodeName = 'PickUp' 
        AND BudgetCharacteristicCodeName = 'Routes'
        AND left(CalendarDateKey,4) IN ({years_list})
    GROUP BY
        BusinessUnitCodeName, LEFT(CalendarDateKey,4), LEFT(RIGHT(CalendarDateKey,4),2)
    ),
    pickup_actual as 
    (
    SELECT 
        BusinessUnit,
        left(RouteDatekey,4) as years,
        LEFT(RIGHT(RouteDatekey,4),2) as months,
        COUNT(DISTINCT RoutesId) as actual_routes,
        SUM(Sales_Amount) as profit
    FROM
        BI_FactRoutes
    WHERE 
        BusinessUnit = 'PICK UP'
        AND left(RouteDatekey,4) IN ({years_list})
    GROUP BY
        BusinessUnit, LEFT(RIGHT(RouteDatekey,4),2),left(RouteDatekey,4)
    )
    SELECT * FROM (
        SELECT 
            'Bulk' AS BusinessUnit,
            ba.years,
            ba.months,
            bb.budget_routes AS budget_routes,
            ba.actual_routes AS actual_routes,
        	ba.profit AS profit
        FROM 
            bulk_actual ba
        INNER JOIN bulk_budget bb
            ON ba.years = bb.years AND ba.months = bb.months
        WHERE 'Bulk' IN ({bu_list})
        
        UNION ALL
        
        SELECT 
            'ADR' AS BusinessUnit,
            aa.years,
            aa.months,
            ab.budget_routes AS budget_routes,
            aa.actual_routes AS actual_routes,
        	aa.profit AS profit
        FROM 
            adr_actual aa
        INNER JOIN adr_budget ab
            ON aa.years = ab.years AND aa.months = ab.months
        WHERE 'ADR' IN ({bu_list})
        
        UNION ALL
        
        SELECT 
            'PickUp' AS BusinessUnit,
            pa.years,
            pa.months,
            pb.budget_routes AS budget_routes,
            pa.actual_routes AS actual_routes,
        	pa.profit AS profit
        FROM 
            pickup_actual pa
        INNER JOIN pickup_budget pb
            ON pa.years = pb.years AND pa.months = pb.months
        WHERE 'PickUp' IN ({bu_list})
    ) result
    """
    return SQL_QUERY

def test_odbc_drivers():
    try:
        print("ODBC Drivers:")
        drivers = pyodbc.drivers()
        print(drivers)
        
        # Προσπάθεια σύνδεσης
        connection_string = os.getenv('DATABASE_CONNECTION_STRING')
        print("Connection String:", connection_string)
        
        if not drivers:
            print("No ODBC drivers found!", file=sys.stderr)
            return False
        
        return True
    except Exception as e:
        print(f"ODBC Driver Test Error: {e}", file=sys.stderr)
        return False

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(user_id):
    logger.info(f"Loading data for user: {user_id}")
    
    try:
        # Check ODBC drivers
        available_drivers = pyodbc.drivers()
        if not available_drivers:
            logger.error("No ODBC drivers found")
            return pd.DataFrame()
        
        # Generate query
        query = get_filtered_sql_query(user_id)
        if query is None:
            logger.warning(f"No query generated for user {user_id}")
            return pd.DataFrame()
        
        # Load environment variables
        load_dotenv()
        connection_string = os.getenv('DATABASE_CONNECTION_STRING')
        
        if not connection_string:
            logger.error("DATABASE_CONNECTION_STRING not found in environment")
            return pd.DataFrame()
        
        # Attempt connection
        with pyodbc.connect(connection_string) as conn:
            logger.info("Database connection established")
            df = pd.read_sql(query, conn)
            
            if df.empty:
                logger.warning("Query returned no data")
            
            df['years'] = df['years'].astype(int)
            df['months'] = df['months'].astype(int)
            
            return df
    
    except pyodbc.Error as e:
        logger.error(f"ODBC Connection Error: {e}")
        logger.error(f"Error details: {e.args}")
    except Exception as e:
        logger.error(f"Unexpected error in load_data: {e}")
        logger.error(traceback.format_exc())
    
    return pd.DataFrame()
# Δημιουργία του Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="user-data-store", style={'display': 'none'}),
    
    # Main container with background color
    html.Div([
        html.Div([
            # Business Unit Filter
            html.Div([
                html.Div(
                    "Business Unit",
                    style={
                        'background-color': '#213052',
                        'color': 'white',
                        'padding': '10px',
                        'text-align': 'center',
                        'font-family': 'Arial',
                        'margin-bottom': '0',
                    }
                ),
                dcc.Dropdown(
                    id="business-unit-dropdown",
                    options=[],
                    value=None,
                    style={
                        'font-family': 'Arial',
                    }
                )
            ], style={'margin-right': '10px', 'width': '200px', 'display': 'inline-block'}),

            # Year Filter
            html.Div([
                html.Div(
                    "Year",
                    style={
                        'background-color': '#213052',
                        'color': 'white',
                        'padding': '10px',
                        'text-align': 'center',
                        'font-family': 'Arial',
                        'margin-bottom': '0',
                    }
                ),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[],
                    value=None,
                    style={
                        'font-family': 'Arial',
                    }
                )
            ], style={'margin-right': '10px', 'width': '200px', 'display': 'inline-block'}),

            # Month Filter
            html.Div([
                html.Div(
                    "Month",
                    style={
                        'background-color': '#213052',
                        'color': 'white',
                        'padding': '10px',
                        'text-align': 'center',
                        'font-family': 'Arial',
                        'margin-bottom': '0',
                    }
                ),
                dcc.Dropdown(
                    id="month-dropdown",
                    options=[{"label": "All", "value": "All"}] +
                            [{"label": month, "value": month} for month in range(1, 13)],
                    value="All",
                    style={
                        'font-family': 'Arial',
                    }
                )
            ], style={'width': '200px', 'display': 'inline-block'}),

        ], style={
            'margin-bottom': '20px',
            'padding': '20px',
        }),

        # First full-width chart
        html.Div([
            html.Div(
                "Routes Per Month",  # Τίτλος του γραφήματος
                style={
                    'backgroundColor': '#213052',  # Μπλε φόντο
                    'color': 'white',             # Λευκή γραμματοσειρά
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontSize': '20px',
                    'fontFamily': 'Arial',
                    'borderRadius': '5px',
                    'marginBottom': '10px'
                }
            ),

            # Το γράφημα
            dcc.Graph(id="grouped-bar-chart")
        ], style={
            "width": "100%", 
            "display": "inline-block",
            "backgroundColor": "white",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0px 0px 10px rgba(0,0,0,0.1)",
            'marginBottom': '20px'
        }),

        # Flexbox container for lower charts
        html.Div([
            # Routes per Business Unit Chart (Left Half)
            html.Div([
                html.Div(
                    "Routes per Business Unit",
                    style={
                        'backgroundColor': '#213052',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontSize': '20px',
                        'fontFamily': 'Arial',
                        'borderRadius': '5px',
                        'marginBottom': '10px'
                    }
                ),
                dcc.Graph(id="business-unit-routes-chart")
            ], style={
                "width": "50%", 
                "padding": "20px",
                "boxSizing": "border-box"
            }),

            # Placeholder for future chart (Right Half)
            html.Div([
                html.Div(
                    "Profit Distribution",
                    style={
                        'backgroundColor': '#213052',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontSize': '20px',
                        'fontFamily': 'Arial',
                        'borderRadius': '5px',
                        'marginBottom': '10px'
                    }
                ),
                dcc.Graph(id="future-chart")
            ], style={
                "width": "50%", 
                "padding": "20px",
                "boxSizing": "border-box"
            })
        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "backgroundColor": "#F3F5F7"
        })
    ], style={
        'backgroundColor': '#F3F5F7',
        'padding': '20px',
        'minHeight': '100vh'
    })
])
@app.callback(
    Output("business-unit-routes-chart", "figure"),
    [Input("user-data-store", "children"),
     Input("year-dropdown", "value"),
     Input("month-dropdown", "value")]
)
def update_business_unit_routes_chart(data_json, selected_year, selected_month):
    if not data_json or not selected_year:
        return go.Figure()
    
    # Λίστα με τα ονόματα των μηνών
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    df = pd.read_json(data_json, orient='split')
    
    # Αποκτούμε τα διαθέσιμα έτη
    available_years = sorted(df['years'].unique())
    
    # Φιλτράρισμα για το επιλεγμένο έτος
    filtered_df = df[df["years"] == selected_year]
    
    # Φιλτράρισμα για συγκεκριμένο μήνα
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df["months"] == selected_month]
    
    # Βρίσκουμε το προηγούμενο διαθέσιμο έτος
    try:
        previous_year_index = available_years.index(selected_year) - 1
        previous_year = available_years[previous_year_index] if previous_year_index >= 0 else None
    except ValueError:
        previous_year = None
    
    # Δεδομένα του προηγούμενου έτους
    last_year_df = df[df["years"] == previous_year] if previous_year else pd.DataFrame()
    
    # Φιλτράρισμα για συγκεκριμένο μήνα στο προηγούμενο έτος
    if not last_year_df.empty and selected_month != "All":
        last_year_df = last_year_df[last_year_df["months"] == selected_month]
    
    # Υπολογισμός συγκεντρωτικών δεδομένων
    bu_summary = filtered_df.groupby("BusinessUnit").agg({
        "budget_routes": "sum", 
        "actual_routes": "sum"
    }).reset_index()
    
    # Υπολογισμός συγκεντρωτικών δεδομένων του προηγούμενου έτους
    bu_summary_last_year = (
        last_year_df.groupby("BusinessUnit")
        .agg({"actual_routes": "sum"})
        .reset_index() if not last_year_df.empty else 
        pd.DataFrame(columns=["BusinessUnit", "actual_routes"])
    )
    
    # Συγχώνευση των δεδομένων
    bu_summary = bu_summary.merge(
        bu_summary_last_year, 
        on="BusinessUnit", 
        how="left", 
        suffixes=('', '_last_year')
    )
    
    # Συμπλήρωση με 0 όπου δεν υπάρχουν δεδομένα
    bu_summary['actual_routes_last_year'] = bu_summary['actual_routes_last_year'].fillna(0)
    
    # Υπολογισμός ποσοστών
    bu_summary['% Routes vs Budget'] = (bu_summary["actual_routes"] / bu_summary["budget_routes"] * 100).round(2)
    bu_summary['Routes %YoY'] = (
        (bu_summary["actual_routes"] / bu_summary["actual_routes_last_year"] * 100).round(2)
        if previous_year else 
        pd.Series([0] * len(bu_summary))
    )
    
    # Προετοιμασία για τίτλο και labels
    title_suffix = f" - {selected_year}"
    if selected_month != "All":
        title_suffix += f" - {month_names[selected_month-1]}"
    
    # Δημιουργία του γράφου
    fig = go.Figure(data=[
        go.Bar(
            x=bu_summary["BusinessUnit"], 
            y=bu_summary["budget_routes"], 
            name="Budget Routes",
            marker_color="#213052",
            text=bu_summary["budget_routes"].round(0),
            textposition="outside"
        ),
        go.Bar(
            x=bu_summary["BusinessUnit"], 
            y=bu_summary["actual_routes"], 
            name="Actual Routes",
            marker_color="#7D1C30",
            text=bu_summary["actual_routes"].round(0),
            textposition="outside"
        ),
        go.Bar(
            x=bu_summary["BusinessUnit"], 
            y=bu_summary["actual_routes_last_year"], 
            name="Routes LY",
            marker_color="#EABA48",
            text=bu_summary["actual_routes_last_year"].round(0),
            textposition="outside"
        ),
        go.Scatter(
            x=bu_summary["BusinessUnit"],
            y=bu_summary['% Routes vs Budget'],
            name="% Routes vs Budget",
            mode="markers",
            yaxis="y2",
            marker=dict(color="#4CAF50", size=10),
            hovertemplate="Business Unit: %{x}<br>% Routes vs Budget: %{y}%<extra></extra>"
        ),
        go.Scatter(
            x=bu_summary["BusinessUnit"],
            y=bu_summary['Routes %YoY'],
            name="Routes %YoY",
            mode="markers",
            yaxis="y2",
            marker=dict(color="#FF9800", size=10),
            hovertemplate="Business Unit: %{x}<br>Routes %YoY: %{y}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        xaxis_title="Business Unit",
        yaxis_title="Number of Routes",
        yaxis2=dict(
            title="Percentage (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(  # Προσαρμόζει το εύρος του άξονα
                max(bu_summary['% Routes vs Budget'].max(), bu_summary['Routes %YoY'].max()) * 1.1,  # 10% παραπάνω
                100  # Τουλάχιστον ως το 100%
            )]
        ),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0,
            bgcolor='rgba(255, 255, 255, 0.5)',
            itemclick="toggleothers"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=80, t=60, b=80)
    )
    
    return fig


@app.callback(
    [Output("user-data-store", "children"),
     Output("business-unit-dropdown", "options"),
     Output("business-unit-dropdown", "value"),
     Output("year-dropdown", "options"),
     Output("year-dropdown", "value"),
     Output("month-dropdown", "value")],
    [Input("url", "search")]
)
def initialize_data(search):
    print("Search parameter received:", search)  # Debug print
    
    if not search:
        print("No search parameter found")  # Debug print
        return "", [], None, [], None, "All"
    
    try:
        # Remove the leading '?' if present
        search = search.lstrip('?')
        
        # Use parse_qs to handle the query parameters
        query_params = parse_qs(search)
        print("Parsed query parameters:", query_params)  # Debug print
        
        # Extract user parameter differently
        user_id = query_params.get('user', [None])[0]
        selected_year = query_params.get('year', [None])[0]
        selected_month = query_params.get('month', ["All"])[0]
        
        print(f"Extracted parameters: user={user_id}, year={selected_year}, month={selected_month}")  # Debug print
        
        if user_id not in user_permissions:
            print(f"User {user_id} not in permissions")  # Debug print
            return "", [], None, [], None, "All"
        
        # Φόρτωση δεδομένων
        df = load_data(user_id)
        if df.empty:
            print("Loaded dataframe is empty")  # Debug print
            return "", [], None, [], None, "All"
        
        # Δημιουργία επιλογών για τα dropdowns
        permissions = user_permissions[user_id]
        business_units = permissions['business_units']
        years = [int(year) for year in permissions['years']]
        
        # Προσθήκη επιλογής του έτους από το URL
        default_year = int(selected_year) if selected_year and int(selected_year) in years else max(years)
        
        # Επεξεργασία του μήνα
        try:
            default_month = int(selected_month) if selected_month != "All" else "All"
            if default_month != "All" and (default_month < 1 or default_month > 12):
                default_month = "All"
        except ValueError:
            default_month = "All"
        
        return (
            df.to_json(date_format='iso', orient='split'),
            [{"label": bu, "value": bu} for bu in business_units],
            business_units[0],
            [{"label": str(year), "value": year} for year in sorted(years)],
            default_year,
            default_month
        )
    except Exception as e:
        print(f"Error in initialize_data: {e}")  # Debug print
        import traceback
        traceback.print_exc()
        return "", [], None, [], None, "All"

@app.callback(
    Output("grouped-bar-chart", "figure"),
    [Input("user-data-store", "children"),
     Input("business-unit-dropdown", "value"),
     Input("year-dropdown", "value"),
     Input("month-dropdown", "value")]
)
def update_chart(data_json, selected_business_unit, selected_year, selected_month):
    if not data_json or not selected_business_unit or not selected_year:
        return go.Figure()
    
    # Λίστα με τα ονόματα των μηνών
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    df = pd.read_json(data_json, orient='split')
    
    # Φιλτράρισμα για το επιλεγμένο Business Unit και έτος
    filtered_df = df[
        (df["BusinessUnit"] == selected_business_unit) &
        (df["years"] == selected_year)
    ]

    # Δεδομένα προηγούμενου έτους
    last_year_df = df[
        (df["BusinessUnit"] == selected_business_unit) &
        (df["years"] == selected_year - 1)
    ]

    # Συγχώνευση των δύο dataframes βάσει των μηνών
    merged_df = pd.merge(filtered_df, last_year_df, 
                         on='months', 
                         suffixes=('_current', '_last'),
                         how='left')

    if selected_month != "All":
        # Φιλτράρισμα για συγκεκριμένο μήνα
        merged_df = merged_df[merged_df["months"] == selected_month]
        x_labels = [month_names[selected_month - 1]]
    else:
        # Μετατροπή των αριθμών των μηνών σε ονόματα
        merged_df['month_names'] = merged_df['months'].apply(lambda x: month_names[x - 1])
        x_labels = merged_df['month_names']

    # Υπολογισμός line chart δεικτών
    merged_df['% Routes vs Budget'] = (merged_df["actual_routes_current"] / merged_df["budget_routes_current"] * 100).round(2)
    merged_df['Routes %YoY'] = (merged_df["actual_routes_current"] / merged_df["actual_routes_last"] * 100).round(2)

    # Δημιουργία του γράφου
    fig = go.Figure()

    # Budget Routes
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=merged_df["budget_routes_current"],
            name="Budget Routes",
            marker_color="#213052",
            text=merged_df["budget_routes_current"],
            textposition="outside",
            hovertemplate="Month: %{x}<br>Budget Routes: %{y}<extra></extra>"
        )
    )

    # Actual Routes
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=merged_df["actual_routes_current"],
            name="Actual Routes",
            marker_color="#7D1C30",
            text=merged_df["actual_routes_current"],
            textposition="outside",
            hovertemplate="Month: %{x}<br>Actual Routes: %{y}<extra></extra>"
        )
    )

    # Last Year Actual Routes
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=merged_df["actual_routes_last"],
            name="Routes LY",
            marker_color="#EABA48",
            text=merged_df["actual_routes_last"],
            textposition="outside",
            hovertemplate="Month: %{x}<br>Routes LY: %{y}<extra></extra>"
        )
    )

    # Line Chart: % Routes vs Budget
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=merged_df['% Routes vs Budget'],
            name="% Routes vs Budget",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#4CAF50", width=3),
            marker=dict(size=10),
            hovertemplate="Month: %{x}<br>% Routes vs Budget: %{y}%<extra></extra>"
        )
    )

    # Line Chart: Routes %YoY
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=merged_df['Routes %YoY'],
            name="Routes %YoY",
            mode="lines+markers",
            yaxis="y2",  # Αυτό άλλαξε για να χρησιμοποιεί τον ίδιο δευτερεύοντα άξονα
            line=dict(color="#FF9800", width=3),
            marker=dict(size=10),
            hovertemplate="Month: %{x}<br>Routes %YoY: %{y}%<extra></extra>"
        )
    )

    # Ενημέρωση του layout
    fig.update_layout(
        barmode="group",
        xaxis_title="Months" if selected_month == "All" else "Month",
        yaxis_title="Number of Routes",
        yaxis2=dict(
            title="Percentage (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(  # Προσαρμόζει το εύρος του άξονα
                max(merged_df['% Routes vs Budget'].max(), merged_df['Routes %YoY'].max()) * 1.1,  # 10% παραπάνω
                100  # Τουλάχιστον ως το 100%
            )]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0,
            bgcolor='rgba(255, 255, 255, 0.5)',
            itemclick="toggleothers"
        ),
        legend_title_text="",
        uniformtext_minsize=10,
        uniformtext_mode="show",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=80, t=40, b=80)
    )

    # Προσαρμογή σχήματος γραφήματος
    fig.update_traces(marker=dict(line=dict(width=0)))

    # Προσθήκη grid lines
    #fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
    #fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')

    return fig

@app.callback(
    Output("future-chart", "figure"),
    [Input("user-data-store", "children"),
     Input("year-dropdown", "value"),
     Input("month-dropdown", "value")]
)
def update_profit_pie_chart(data_json, selected_year, selected_month):
    if not data_json or not selected_year:
        return go.Figure()
    
    # Λίστα με τα ονόματα των μηνών
    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    df = pd.read_json(data_json, orient='split')
    
    # Φιλτράρισμα για το επιλεγμένο έτος
    filtered_df = df[df["years"] == selected_year]
    
    # Φιλτράρισμα για συγκεκριμένο μήνα
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df["months"] == selected_month]
        title_suffix = f" - {selected_year} - {month_names[selected_month-1]}"
    else:
        title_suffix = f" - {selected_year}"
    
    # Υπολογισμός συνολικού profit ανά Business Unit
    profit_summary = filtered_df.groupby("BusinessUnit")["profit"].sum().reset_index()
    
    # Δημιουργία pie chart
    fig = go.Figure(data=[go.Pie(
        labels=profit_summary["BusinessUnit"], 
        values=profit_summary["profit"],
        textinfo='percent+value',
        hole=0.3,  # Δημιουργία donut chart
        marker_colors=['#213052', '#7D1C30', '#EABA48']  # Χρήση των ίδιων χρωμάτων με τα προηγούμενα γραφήματα
    )])
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="left",
            x=0,
            bgcolor='rgba(255, 255, 255, 0.5)',
            itemclick="toggleothers"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=80)
    )
    
    return fig

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    server.run(host='0.0.0.0', port=port)
