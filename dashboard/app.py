"""
Streamlit Dashboard for AQI Prediction System
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import OpenMeteoFetcher, AQICalculator
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.config import LocationConfig, ModelConfig
from src.alerts import AlertManager

# Page configuration
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models"""
    trainer = ModelTrainer()
    config = ModelConfig()
    
    if config.model_path.exists():
        trainer.load_models(config.model_path)
    
    return trainer


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_current_aqi():
    """Fetch current AQI data"""
    fetcher = OpenMeteoFetcher()
    return fetcher.fetch_current_air_quality()


@st.cache_data(ttl=3600)
def fetch_historical_data(days=30):
    """Fetch historical AQI data"""
    fetcher = OpenMeteoFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return fetcher.fetch_air_quality_history(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )


def get_aqi_color(aqi):
    """Get color based on AQI value"""
    if aqi <= 50:
        return "#00e400"  # Green - Good
    elif aqi <= 100:
        return "#ffff00"  # Yellow - Moderate
    elif aqi <= 150:
        return "#ff7e00"  # Orange - Unhealthy for Sensitive Groups
    elif aqi <= 200:
        return "#ff0000"  # Red - Unhealthy
    elif aqi <= 300:
        return "#8f3f97"  # Purple - Very Unhealthy
    else:
        return "#7e0023"  # Maroon - Hazardous


def plot_aqi_gauge(aqi_value):
    """Create AQI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current AQI"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': get_aqi_color(aqi_value)},
            'steps': [
                {'range': [0, 50], 'color': "#00e400"},
                {'range': [50, 100], 'color': "#ffff00"},
                {'range': [100, 150], 'color': "#ff7e00"},
                {'range': [150, 200], 'color': "#ff0000"},
                {'range': [200, 300], 'color': "#8f3f97"},
                {'range': [300, 500], 'color': "#7e0023"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def plot_historical_trend(df):
    """Plot historical AQI trend"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['aqi'],
        mode='lines',
        name='AQI',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)'
    ))
    
    # Add category reference lines
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
    
    fig.update_layout(
        title="Historical AQI Trend",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_pollutants(df):
    """Plot pollutant concentrations"""
    pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    fig = go.Figure()
    
    for pollutant in available_pollutants:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df[pollutant],
            mode='lines',
            name=pollutant.upper().replace('_', '.')
        ))
    
    fig.update_layout(
        title="Pollutant Concentrations Over Time",
        xaxis_title="Date",
        yaxis_title="Concentration (µg/m³)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_predictions(predictions_df):
    """Plot AQI predictions"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=predictions_df['date'],
        y=predictions_df['predicted_aqi'],
        marker_color=[get_aqi_color(aqi) for aqi in predictions_df['predicted_aqi']],
        name='Predicted AQI',
        text=predictions_df['predicted_aqi'].round(0),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="3-Day AQI Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted AQI",
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Title
    st.title("🌍 Air Quality Index Prediction Dashboard")
    
    # Location info
    location = LocationConfig()
    st.markdown(f"**Location:** {location.city_name} ({location.latitude}°N, {location.longitude}°E)")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Historical data days selection
    historical_days = st.sidebar.slider(
        "Historical Data (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    # Model selection
    trainer = load_models()
    available_models = list(trainer.models.keys()) if trainer.models else ["ridge"]
    selected_model = st.sidebar.selectbox(
        "Prediction Model",
        available_models,
        index=0
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Current Status", "📈 Historical Data", "🔮 Predictions", "⚠️ Alerts", "ℹ️ Model Info"])
    
    # Tab 1: Current Status
    with tab1:
        st.header("Current Air Quality")
        
        try:
            current_data = fetch_current_aqi()
            
            if current_data:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # AQI Gauge
                    fig_gauge = plot_aqi_gauge(current_data['aqi'])
                    st.plotly_chart(fig_gauge, width='stretch')
                
                with col2:
                    st.metric("AQI Value", current_data['aqi'])
                    st.markdown(f"**Category:** {current_data['aqi_category']}")
                    st.markdown(f"**Dominant Pollutant:** {current_data['dominant_pollutant']}")
                
                with col3:
                    st.metric("PM2.5", f"{current_data.get('pm2_5', 0):.1f} µg/m³")
                    st.metric("PM10", f"{current_data.get('pm10', 0):.1f} µg/m³")
                    st.metric("O3", f"{current_data.get('o3', 0):.1f} µg/m³")
                
                # Pollutant details
                st.subheader("Pollutant Concentrations")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("PM2.5", f"{current_data.get('pm2_5', 0):.1f}")
                with col2:
                    st.metric("PM10", f"{current_data.get('pm10', 0):.1f}")
                with col3:
                    st.metric("CO", f"{current_data.get('co', 0):.1f}")
                with col4:
                    st.metric("NO2", f"{current_data.get('no2', 0):.1f}")
                with col5:
                    st.metric("SO2", f"{current_data.get('so2', 0):.1f}")
                with col6:
                    st.metric("O3", f"{current_data.get('o3', 0):.1f}")
                
                st.caption("All pollutant concentrations in µg/m³")
                
            else:
                st.error("Failed to fetch current air quality data")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Tab 2: Historical Data
    with tab2:
        st.header("Historical Air Quality Trends")
        
        try:
            df = fetch_historical_data(historical_days)
            
            if not df.empty:
                # AQI Trend
                fig_trend = plot_historical_trend(df)
                st.plotly_chart(fig_trend, width='stretch')
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average AQI", f"{df['aqi'].mean():.1f}")
                with col2:
                    st.metric("Max AQI", f"{df['aqi'].max():.0f}")
                with col3:
                    st.metric("Min AQI", f"{df['aqi'].min():.0f}")
                with col4:
                    st.metric("Std Dev", f"{df['aqi'].std():.1f}")
                
                # Category distribution
                st.subheader("AQI Category Distribution")
                if 'aqi_category' in df.columns:
                    category_counts = df['aqi_category'].value_counts()
                    fig_pie = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Distribution of AQI Categories"
                    )
                    st.plotly_chart(fig_pie, width='stretch')
                
                # Pollutant trends
                st.subheader("Pollutant Trends")
                fig_pollutants = plot_pollutants(df)
                st.plotly_chart(fig_pollutants, width='stretch')
                
            else:
                st.warning("No historical data available")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Tab 3: Predictions
    with tab3:
        st.header("AQI Predictions (3-Day Forecast)")
        
        try:
            if not trainer.models:
                st.warning("Models not loaded. Please train models first.")
            else:
                # Fetch recent data for predictions
                df = fetch_historical_data(7)
                
                if not df.empty:
                    # Engineer features
                    engineer = FeatureEngineer()
                    df_engineered = engineer.engineer_features(df)
                    
                    X, _, _ = engineer.prepare_training_data(df_engineered)
                    
                    if not X.empty:
                        # Make predictions for next 3 days
                        X_latest = X.iloc[-1:].copy()
                        
                        predictions = []
                        aqi_calc = AQICalculator()
                        
                        for day in range(1, 4):
                            pred_aqi = trainer.predict(X_latest, model_name=selected_model)[0]
                            pred_aqi = max(0, min(500, pred_aqi))
                            
                            predictions.append({
                                'date': (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                                'predicted_aqi': pred_aqi,
                                'category': aqi_calc.get_aqi_category(int(pred_aqi))
                            })
                        
                        predictions_df = pd.DataFrame(predictions)
                        
                        # Plot predictions
                        fig_pred = plot_predictions(predictions_df)
                        st.plotly_chart(fig_pred, width='stretch')
                        
                        # Display predictions table
                        st.subheader("Detailed Forecast")
                        for idx, pred in enumerate(predictions):
                            col1, col2, col3 = st.columns([2, 1, 2])
                            with col1:
                                st.markdown(f"**{pred['date']}**")
                            with col2:
                                st.markdown(f"**AQI: {pred['predicted_aqi']:.0f}**")
                            with col3:
                                st.markdown(f"**{pred['category']}**")
                    else:
                        st.error("Failed to prepare features for prediction")
                else:
                    st.warning("No recent data available for predictions")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Tab 4: Alerts
    with tab4:
        st.header("⚠️ AQI Hazard Alerts")

        alert_mgr = AlertManager()

        try:
            current_data = fetch_current_aqi()
            if current_data:
                aqi_val = current_data["aqi"]

                # Evaluate & possibly trigger alert
                alert_mgr.evaluate(aqi_val, location=location.city_name)
                status = alert_mgr.get_current_status(aqi_val)

                # Status banner
                level = status["level"]
                color_map = {"info": "green", "warning": "orange", "alert": "red", "critical": "darkred"}
                banner_color = color_map.get(level, "gray")
                st.markdown(
                    f'<div style="padding:16px;border-radius:8px;background:{banner_color};color:white;'
                    f'font-size:18px;margin-bottom:16px;">'
                    f'<b>{status["category"]}</b> — AQI {aqi_val} — {status["message"]}</div>',
                    unsafe_allow_html=True,
                )

                # Recommendations
                st.subheader("Health Recommendations")
                for rec in status["recommendations"]:
                    st.markdown(f"- {rec}")
            else:
                st.warning("Could not fetch current AQI for alert evaluation.")
        except Exception as e:
            st.error(f"Error evaluating alerts: {e}")

        # Alert history
        st.subheader("Alert History")
        history = alert_mgr.get_history(limit=30)
        if history:
            hist_df = pd.DataFrame(history)
            hist_df = hist_df[["timestamp", "aqi", "category", "level", "message"]]
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("No alerts recorded yet.")

    # Tab 5: Model Info
    with tab5:
        st.header("Model Performance Metrics")
        
        if trainer.metrics:
            metrics_df = pd.DataFrame([
                {
                    'Model': metrics['model_name'],
                    'MAE': f"{metrics['mae']:.2f}",
                    'RMSE': f"{metrics['rmse']:.2f}",
                    'R²': f"{metrics['r2']:.4f}",
                    'Test Samples': metrics['n_test_samples']
                }
                for metrics in trainer.metrics.values()
            ])
            
            st.dataframe(metrics_df, width='stretch')
            
            # Success criteria
            st.subheader("Success Criteria")
            st.markdown("""
            - ✅ **MAE < 15**: Mean Absolute Error should be less than 15 AQI points
            - ✅ **R² > 0.6**: Model should explain more than 60% of variance
            """)
            
            # Feature importance (if Random Forest available)
            if 'random_forest' in trainer.models:
                st.subheader("Feature Importance (Random Forest)")
                importance_df = trainer.get_feature_importance('random_forest')
                
                if not importance_df.empty:
                    top_features = importance_df.head(15)
                    fig_importance = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Features"
                    )
                    st.plotly_chart(fig_importance, width='stretch')
        else:
            st.warning("No model metrics available. Please train models first.")
        
        # About
        st.subheader("About")
        st.markdown("""
        This dashboard uses **Open-Meteo API** (no API key required) to fetch real-time air quality data
        and machine learning models to predict future AQI values.
        
        **Models:**
        - Ridge Regression (Primary)
        - Random Forest
        - XGBoost
        
        **Data Source:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
        """)


if __name__ == "__main__":
    main()
