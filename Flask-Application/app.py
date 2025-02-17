from flask import Flask, jsonify
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# Initialize Flask and Dash apps
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
df = pd.read_csv('/path/to/your/fraud_data.csv')


@server.route('/summary')
def summary():
    total_transactions = df.shape[0]
    total_fraud_cases = df[df['Class'] == 1].shape[0]
    fraud_percentage = (total_fraud_cases / total_transactions) * 100
    return jsonify({
        "total_transactions": total_transactions,
        "total_fraud_cases": total_fraud_cases,
        "fraud_percentage": fraud_percentage
    })


@server.route('/fraud_trends')
def fraud_trends():
    df['Date'] = pd.to_datetime(df['Date'])
    trends = df[df['Class'] == 1].groupby(df['Date'].dt.to_period('M')).size()
    return jsonify(trends.to_dict())


app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),

    # Summary Boxes
    dbc.Row([
        dbc.Col(html.Div(id="total-transactions"), width=3),
        dbc.Col(html.Div(id="total-fraud-cases"), width=3),
        dbc.Col(html.Div(id="fraud-percentage"), width=3),
    ]),

    # Line Chart for Fraud Cases Over Time
    html.Div([
        dcc.Graph(id="fraud-trends-line-chart")
    ]),

    # Fraud Cases by Geography and Device/Browser
    dbc.Row([
        dbc.Col(dcc.Graph(id="geo-fraud-map"), width=6),
        dbc.Col(dcc.Graph(id="device-fraud-bar-chart"), width=6),
    ]),
])




from dash.dependencies import Input, Output

@app.callback(
    [Output("total-transactions", "children"),
     Output("total-fraud-cases", "children"),
     Output("fraud-percentage", "children")],
    Input("total-transactions", "id")
)
def update_summary_boxes(_):
    summary_data = server.test_client().get('/summary').get_json()
    return [
        f"Total Transactions: {summary_data['total_transactions']}",
        f"Total Fraud Cases: {summary_data['total_fraud_cases']}",
        f"Fraud Percentage: {summary_data['fraud_percentage']:.2f}%"
    ]

@app.callback(Output("fraud-trends-line-chart", "figure"), Input("fraud-trends-line-chart", "id"))
def update_fraud_trends(_):
    trends_data = server.test_client().get('/fraud_trends').get_json()
    df_trends = pd.DataFrame.from_dict(trends_data, orient='index', columns=['Fraud Cases'])
    df_trends.index = pd.to_datetime(df_trends.index, format='%Y-%m')
    fig = px.line(df_trends, x=df_trends.index, y="Fraud Cases", title="Fraud Cases Over Time")
    return fig

# Additional callbacks for geographic map and device/browser bar chart would go here
@app.callback(Output("geo-fraud-map", "figure"), Input("geo-fraud-map", "id"))
def update_geo_map(_):
    fraud_geo_df = df[df['Class'] == 1]
    fig = px.scatter_mapbox(
        fraud_geo_df, lat="latitude", lon="longitude", color="Class", 
        title="Fraud Locations", mapbox_style="carto-positron"
    )
    return fig



@app.callback(Output("device-fraud-bar-chart", "figure"), Input("device-fraud-bar-chart", "id"))
def update_device_browser_chart(_):
    fraud_device_df = df[df['Class'] == 1].groupby(['device', 'browser']).size().reset_index(name='Count')
    fig = px.bar(
        fraud_device_df, x="device", y="Count", color="browser",
        title="Fraud Cases by Device and Browser"
    )
    return fig
