import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Configure Streamlit page
st.set_page_config(
    page_title="AI Health Intervention System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Access Claude API key from Streamlit Secrets
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    import anthropic
    claude_client = anthropic.Anthropic(api_key=api_key)
    api_available = True
    st.sidebar.success("✅ Claude API Connected")
except Exception as e:
    api_key = None
    claude_client = None
    api_available = False
    st.sidebar.info("💡 Claude API not configured - using template responses")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .persona-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .intervention-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .alert-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-normal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class UserPersona:
    """Represents a user persona with health characteristics"""
    name: str
    age: int
    condition: str
    personality_type: str
    health_goals: List[str]
    baseline_glucose: float
    glucose_variability: float
    weight: float
    activity_level: str
    dietary_preferences: str
    medication: str
    stress_level: str
    sleep_quality: str

@dataclass
class HealthData:
    """Represents health data for a specific time point"""
    timestamp: datetime
    glucose_level: float
    weight: float
    steps: int
    heart_rate: int
    sleep_hours: float
    stress_level: int
    meal_logged: bool
    meal_type: str
    meal_carbs: float

class PersonaGenerator:
    """Generates realistic user personas for the health intervention system"""
    
    @staticmethod
    def generate_personas() -> List[UserPersona]:
        """Generate a diverse set of user personas"""
        personas = [
            UserPersona(
                name="Sarah Chen",
                age=34,
                condition="Type 2 Diabetes",
                personality_type="Analytical & Goal-oriented",
                health_goals=["Maintain HbA1c under 7%", "Lose 15 lbs", "Exercise 4x/week"],
                baseline_glucose=140,
                glucose_variability=25,
                weight=165,
                activity_level="Moderate",
                dietary_preferences="Low-carb focused",
                medication="Metformin 500mg 2x daily",
                stress_level="Moderate",
                sleep_quality="Good"
            ),
            UserPersona(
                name="Michael Rodriguez",
                age=45,
                condition="Prediabetes",
                personality_type="Busy professional, needs convenience",
                health_goals=["Prevent diabetes progression", "Reduce stress", "Improve sleep"],
                baseline_glucose=115,
                glucose_variability=15,
                weight=185,
                activity_level="Low",
                dietary_preferences="Quick meals, eating out frequently",
                medication="None",
                stress_level="High",
                sleep_quality="Poor"
            ),
            UserPersona(
                name="Emma Thompson",
                age=28,
                condition="Type 1 Diabetes",
                personality_type="Tech-savvy, social media active",
                health_goals=["Optimize CGM readings", "Build muscle", "Travel confidently"],
                baseline_glucose=120,
                glucose_variability=35,
                weight=135,
                activity_level="High",
                dietary_preferences="Flexible, tracks macros",
                medication="Insulin pump + CGM",
                stress_level="Low",
                sleep_quality="Excellent"
            ),
            UserPersona(
                name="Robert Johnson",
                age=62,
                condition="Type 2 Diabetes + Hypertension",
                personality_type="Traditional, prefers simple approaches",
                health_goals=["Simplify medication routine", "Gentle exercise", "Family time"],
                baseline_glucose=160,
                glucose_variability=30,
                weight=210,
                activity_level="Low",
                dietary_preferences="Traditional meals, portion control",
                medication="Metformin + Insulin + BP meds",
                stress_level="Moderate",
                sleep_quality="Fair"
            ),
            UserPersona(
                name="Priya Patel",
                age=31,
                condition="Gestational Diabetes",
                personality_type="Cautious, family-focused",
                health_goals=["Healthy pregnancy", "Prevent future T2D", "Learn meal planning"],
                baseline_glucose=125,
                glucose_variability=20,
                weight=155,
                activity_level="Light",
                dietary_preferences="Vegetarian, cultural foods",
                medication="Diet-controlled, monitoring",
                stress_level="Moderate",
                sleep_quality="Poor (pregnancy-related)"
            )
        ]
        return personas

class HealthDataGenerator:
    """Generates realistic health data for personas"""
    
    @staticmethod
    def generate_health_timeline(persona: UserPersona, days: int = 30) -> List[HealthData]:
        """Generate a timeline of health data for a persona"""
        data_points = []
        base_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            # Multiple readings per day
            for reading in range(4):  # 4 readings per day
                timestamp = base_date + timedelta(days=day, hours=6 + reading * 4)
                
                # Generate glucose with realistic patterns
                glucose = HealthDataGenerator._generate_glucose_reading(persona, reading, day)
                
                # Generate other metrics
                weight = persona.weight + random.gauss(0, 0.5)  # Small daily variations
                steps = HealthDataGenerator._generate_steps(persona, reading)
                heart_rate = HealthDataGenerator._generate_heart_rate(persona)
                sleep_hours = HealthDataGenerator._generate_sleep(persona) if reading == 0 else 0
                stress_level = random.randint(1, 10)
                
                # Meal logging based on time of day
                meal_logged, meal_type, meal_carbs = HealthDataGenerator._generate_meal_data(reading)
                
                data_points.append(HealthData(
                    timestamp=timestamp,
                    glucose_level=glucose,
                    weight=weight,
                    steps=steps,
                    heart_rate=heart_rate,
                    sleep_hours=sleep_hours,
                    stress_level=stress_level,
                    meal_logged=meal_logged,
                    meal_type=meal_type,
                    meal_carbs=meal_carbs
                ))
        
        return data_points
    
    @staticmethod
    def _generate_glucose_reading(persona: UserPersona, reading_time: int, day: int) -> float:
        """Generate realistic glucose reading based on persona and time"""
        base = persona.baseline_glucose
        variability = persona.glucose_variability
        
        # Time-of-day patterns
        time_multipliers = [0.95, 1.1, 1.05, 0.98]  # Morning, lunch, dinner, evening
        time_factor = time_multipliers[reading_time]
        
        # Add some weekly patterns
        weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 7)
        
        # Random variation
        noise = random.gauss(0, variability * 0.3)
        
        glucose = base * time_factor * weekly_factor + noise
        return max(70, min(400, glucose))  # Realistic bounds
    
    @staticmethod
    def _generate_steps(persona: UserPersona, reading_time: int) -> int:
        """Generate step count based on persona activity level"""
        activity_multipliers = {"Low": 0.3, "Light": 0.5, "Moderate": 0.7, "High": 1.0}
        base_steps = 2500 * activity_multipliers.get(persona.activity_level, 0.5)
        
        # More steps during middle readings (active hours)
        time_multipliers = [0.1, 0.4, 0.4, 0.1]
        daily_steps = int(base_steps * time_multipliers[reading_time] + random.gauss(0, 500))
        return max(0, daily_steps)
    
    @staticmethod
    def _generate_heart_rate(persona: UserPersona) -> int:
        """Generate heart rate based on persona characteristics"""
        base_hr = 70 - (persona.age - 30) * 0.5  # Age-adjusted baseline
        variation = random.gauss(0, 10)
        return int(max(50, min(120, base_hr + variation)))
    
    @staticmethod
    def _generate_sleep(persona: UserPersona) -> float:
        """Generate sleep hours based on persona sleep quality"""
        quality_multipliers = {"Poor": 5.5, "Fair": 6.5, "Good": 7.5, "Excellent": 8.0}
        base_sleep = quality_multipliers.get(persona.sleep_quality, 7.0)
        return max(3, min(12, base_sleep + random.gauss(0, 0.8)))
    
    @staticmethod
    def _generate_meal_data(reading_time: int) -> Tuple[bool, str, float]:
        """Generate meal data based on time of reading"""
        meal_times = ["Breakfast", "Lunch", "Dinner", "Snack"]
        meal_probs = [0.8, 0.9, 0.85, 0.3]  # Probability of logging meal
        
        meal_logged = random.random() < meal_probs[reading_time]
        meal_type = meal_times[reading_time] if meal_logged else ""
        
        # Carb content varies by meal type
        carb_ranges = {"Breakfast": (20, 45), "Lunch": (30, 60), "Dinner": (35, 70), "Snack": (10, 25)}
        meal_carbs = random.uniform(*carb_ranges.get(meal_type, (0, 0))) if meal_logged else 0
        
        return meal_logged, meal_type, meal_carbs

class InterventionEngine:
    """Generates AI-powered health interventions using Claude"""
    
    def __init__(self, client=None):
        """Initialize the intervention engine with Claude client"""
        self.client = client
    
    def generate_intervention(self, persona: UserPersona, recent_data: List[HealthData], 
                            trigger_event: Dict) -> str:
        """Generate a personalized intervention using Claude"""
        if not self.client:
            return self._generate_fallback_intervention(persona, trigger_event)
        
        try:
            # Prepare context for Claude
            context = self._prepare_intervention_context(persona, recent_data, trigger_event)
            
            prompt = f"""You are an empathetic AI health coach working with diabetes and metabolic health patients. 

Based on the following patient context, generate a personalized, supportive intervention message:

{context}

Please provide:
1. A warm, non-judgmental acknowledgment of the current situation
2. A brief explanation of what might have caused this reading
3. 2-3 specific, actionable steps the patient can take right now
4. Encouragement that's personalized to their personality and goals

Keep the tone conversational, supportive, and tailored to their personality type. The message should be 150-200 words and feel like it's coming from a knowledgeable friend who cares about their health journey."""

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            st.error(f"Error generating intervention: {str(e)}")
            return self._generate_fallback_intervention(persona, trigger_event)
    
    def _prepare_intervention_context(self, persona: UserPersona, recent_data: List[HealthData], 
                                    trigger_event: Dict) -> str:
        """Prepare context information for Claude"""
        latest_data = recent_data[-1] if recent_data else None
        
        context = f"""
PATIENT PROFILE:
- Name: {persona.name}
- Age: {persona.age}
- Condition: {persona.condition}
- Personality: {persona.personality_type}
- Goals: {', '.join(persona.health_goals)}
- Activity Level: {persona.activity_level}
- Stress Level: {persona.stress_level}
- Sleep Quality: {persona.sleep_quality}

TRIGGER EVENT:
- Event Type: {trigger_event['type']}
- Current Glucose: {trigger_event['glucose_level']} mg/dL
- Severity: {trigger_event['severity']}
- Description: {trigger_event['description']}

RECENT CONTEXT:
"""
        
        if latest_data:
            context += f"""
- Recent meal: {latest_data.meal_type} ({latest_data.meal_carbs}g carbs) 
- Steps today: {latest_data.steps}
- Stress level: {latest_data.stress_level}/10
- Sleep last night: {latest_data.sleep_hours} hours
"""

        return context
    
    def _generate_fallback_intervention(self, persona: UserPersona, trigger_event: Dict) -> str:
        """Generate a fallback intervention when Claude API is not available"""
        glucose = trigger_event['glucose_level']
        severity = trigger_event['severity']
        
        # Template-based interventions based on persona and glucose level
        interventions = {
            "high_glucose": {
                "Analytical & Goal-oriented": f"Hi {persona.name}! I see your glucose is at {glucose} mg/dL. Let's break this down: this could be from your recent meal or stress. Try taking a 10-minute walk, drinking water, and checking again in 1 hour. Remember, one high reading doesn't derail your progress toward your HbA1c goal!",
                "Busy professional, needs convenience": f"{persona.name}, quick heads up - your glucose is elevated at {glucose} mg/dL. Here's a 2-minute action plan: drink a large glass of water, do 20 desk push-ups if possible, and set a phone reminder to check again in an hour. Small actions, big impact!",
                "Tech-savvy, social media active": f"Hey {persona.name}! 📊 Glucose spike alert: {glucose} mg/dL. Time for damage control! Try a quick HIIT session (even 5 minutes helps), hydrate well, and maybe share your comeback story later. Your travel goals are still totally achievable! 💪",
                "Traditional, prefers simple approaches": f"Hello {persona.name}. Your glucose reading is {glucose} mg/dL, which is higher than ideal. Try these simple steps: take a gentle 15-minute walk, have a glass of water, and avoid any snacks for now. These small steps make a big difference for your health.",
                "Cautious, family-focused": f"Hi {persona.name}, your glucose is at {glucose} mg/dL. For you and baby's health, let's address this gently: try some light movement like walking around the house, sip water slowly, and rest. Contact your healthcare provider if it stays high. You're doing great taking care of both of you! 💕"
            }
        }
        
        personality_key = persona.personality_type
        if severity == "High" and personality_key in interventions["high_glucose"]:
            return interventions["high_glucose"][personality_key]
        else:
            return f"Hi {persona.name}! I noticed your glucose is at {glucose} mg/dL. Consider taking a short walk, staying hydrated, and monitoring how you feel. You've got this! 💪"

def analyze_glucose_patterns(data: List[HealthData]) -> Dict:
    """Analyze glucose patterns and identify triggers"""
    if not data:
        return {}
    
    df = pd.DataFrame([{
        'timestamp': d.timestamp,
        'glucose': d.glucose_level,
        'meal_logged': d.meal_logged,
        'meal_carbs': d.meal_carbs,
        'steps': d.steps,
        'stress': d.stress_level
    } for d in data])
    
    # Identify potential triggers for high glucose
    high_glucose_threshold = 180
    high_readings = df[df['glucose'] > high_glucose_threshold]
    
    analysis = {
        'avg_glucose': df['glucose'].mean(),
        'glucose_std': df['glucose'].std(),
        'high_readings_count': len(high_readings),
        'time_in_range': len(df[(df['glucose'] >= 70) & (df['glucose'] <= 180)]) / len(df) * 100,
        'correlation_meal_carbs': df['glucose'].corr(df['meal_carbs']) if df['meal_carbs'].sum() > 0 else 0,
        'correlation_stress': df['glucose'].corr(df['stress']),
        'correlation_activity': df['glucose'].corr(df['steps'])
    }
    
    return analysis

def detect_intervention_triggers(recent_data: List[HealthData]) -> List[Dict]:
    """Detect events that should trigger an intervention"""
    triggers = []
    
    if not recent_data:
        return triggers
    
    latest = recent_data[-1]
    
    # High glucose trigger
    if latest.glucose_level > 180:
        severity = "Critical" if latest.glucose_level > 250 else "High"
        triggers.append({
            'type': 'High Glucose',
            'glucose_level': latest.glucose_level,
            'severity': severity,
            'description': f'Glucose level of {latest.glucose_level:.0f} mg/dL detected',
            'timestamp': latest.timestamp
        })
    
    # Post-meal spike trigger
    if len(recent_data) >= 2:
        prev = recent_data[-2]
        if (latest.meal_logged and 
            latest.glucose_level > prev.glucose_level + 50 and 
            latest.glucose_level > 160):
            triggers.append({
                'type': 'Post-Meal Spike',
                'glucose_level': latest.glucose_level,
                'severity': 'Moderate',
                'description': f'Post-meal glucose spike: {latest.glucose_level:.0f} mg/dL after {latest.meal_type}',
                'timestamp': latest.timestamp
            })
    
    # Low activity + high stress trigger
    if latest.steps < 1000 and latest.stress_level > 7 and latest.glucose_level > 150:
        triggers.append({
            'type': 'Stress & Inactivity',
            'glucose_level': latest.glucose_level,
            'severity': 'Moderate',
            'description': f'High stress ({latest.stress_level}/10) and low activity may be affecting glucose',
            'timestamp': latest.timestamp
        })
    
    return triggers

def create_glucose_chart(data: List[HealthData]) -> go.Figure:
    """Create an interactive glucose chart"""
    df = pd.DataFrame([{
        'timestamp': d.timestamp,
        'glucose': d.glucose_level,
        'meal_logged': d.meal_logged,
        'meal_type': d.meal_type,
        'meal_carbs': d.meal_carbs
    } for d in data])
    
    fig = go.Figure()
    
    # Add glucose line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['glucose'],
        mode='lines+markers',
        name='Glucose Level',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Add target range
    fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target Min (70)")
    fig.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="Target Max (180)")
    fig.add_hline(y=250, line_dash="dash", line_color="red", annotation_text="Critical (250)")
    
    # Highlight meal times
    meal_data = df[df['meal_logged'] == True]
    if not meal_data.empty:
        fig.add_trace(go.Scatter(
            x=meal_data['timestamp'],
            y=meal_data['glucose'],
            mode='markers',
            name='Meal Times',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond'
            ),
            text=meal_data['meal_type'] + '<br>' + meal_data['meal_carbs'].astype(str) + 'g carbs',
            hovertemplate='%{text}<br>Glucose: %{y} mg/dL<extra></extra>'
        ))
    
    fig.update_layout(
        title='Continuous Glucose Monitoring',
        xaxis_title='Time',
        yaxis_title='Glucose Level (mg/dL)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI-Powered Health Intervention System</h1>', unsafe_allow_html=True)
    st.markdown("*Demonstrating personalized health interventions using Claude AI and realistic health data*")
    
    # Initialize the intervention engine
    intervention_engine = InterventionEngine(claude_client)
    
    # Generate personas
    personas = PersonaGenerator.generate_personas()
    
    # Persona selection
    st.sidebar.header("👤 Select Patient")
    selected_persona_name = st.sidebar.selectbox(
        "Choose a patient persona:",
        [persona.name for persona in personas]
    )
    
    selected_persona = next(p for p in personas if p.name == selected_persona_name)
    
    # Time range selection
    days_back = st.sidebar.slider("Days of data to generate:", 7, 60, 30)
    
    # Generate health data
    if f'health_data_{selected_persona.name}' not in st.session_state:
        with st.spinner(f"Generating health data for {selected_persona.name}..."):
            st.session_state[f'health_data_{selected_persona.name}'] = HealthDataGenerator.generate_health_timeline(
                selected_persona, days_back
            )
    
    health_data = st.session_state[f'health_data_{selected_persona.name}']
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Patient profile
        st.subheader(f"📋 Patient Profile: {selected_persona.name}")
        
        profile_html = f"""
        <div class="persona-card">
            <h4>{selected_persona.name}, {selected_persona.age} years old</h4>
            <p><strong>Condition:</strong> {selected_persona.condition}</p>
            <p><strong>Personality:</strong> {selected_persona.personality_type}</p>
            <p><strong>Goals:</strong> {', '.join(selected_persona.health_goals)}</p>
            <p><strong>Medication:</strong> {selected_persona.medication}</p>
            <p><strong>Activity Level:</strong> {selected_persona.activity_level}</p>
        </div>
        """
        st.markdown(profile_html, unsafe_allow_html=True)
        
        # Glucose chart
        st.subheader("📊 Glucose Monitoring Dashboard")
        glucose_chart = create_glucose_chart(health_data)
        st.plotly_chart(glucose_chart, use_container_width=True)
        
        # Health metrics summary
        analysis = analyze_glucose_patterns(health_data)
        
        if analysis:
            st.subheader("📈 Health Metrics Summary")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric(
                    "Average Glucose",
                    f"{analysis['avg_glucose']:.1f} mg/dL",
                    delta=None
                )
            
            with metrics_col2:
                st.metric(
                    "Time in Range",
                    f"{analysis['time_in_range']:.1f}%",
                    delta=None
                )
            
            with metrics_col3:
                st.metric(
                    "High Readings",
                    f"{analysis['high_readings_count']}",
                    delta=None
                )
            
            with metrics_col4:
                st.metric(
                    "Glucose Variability",
                    f"{analysis['glucose_std']:.1f}",
                    delta=None
                )
    
    with col2:
        # Real-time intervention system
        st.subheader("🤖 AI Intervention System")
        
        # Show API status
        if api_available:
            st.success("🤖 Claude AI Active")
        else:
            st.info("📝 Template Mode")
        
        # Detect triggers
        triggers = detect_intervention_triggers(health_data[-10:])  # Check last 10 readings
        
        if triggers:
            st.warning(f"⚠️ {len(triggers)} intervention trigger(s) detected!")
            
            for trigger in triggers[-3:]:  # Show last 3 triggers
                severity_color = {
                    'Critical': 'alert-high',
                    'High': 'alert-high', 
                    'Moderate': 'alert-normal'
                }.get(trigger['severity'], 'alert-normal')
                
                trigger_html = f"""
                <div class="{severity_color}">
                    <strong>{trigger['type']}</strong><br>
                    {trigger['description']}<br>
                    <small>Severity: {trigger['severity']}</small>
                </div>
                """
                st.markdown(trigger_html, unsafe_allow_html=True)
                
                # Generate intervention
                if st.button(f"Generate Intervention for {trigger['type']}", key=f"btn_{trigger['type']}"):
                    with st.spinner("Generating personalized intervention..."):
                        intervention = intervention_engine.generate_intervention(
                            selected_persona, 
                            health_data[-5:], 
                            trigger
                        )
                        
                        intervention_html = f"""
                        <div class="intervention-box">
                            <h5>🤖 AI Health Coach Response:</h5>
                            <p>{intervention}</p>
                            <small><em>Generated at {datetime.now().strftime('%H:%M:%S')}</em></small>
                        </div>
                        """
                        st.markdown(intervention_html, unsafe_allow_html=True)
        else:
            st.success("✅ No intervention triggers detected")
            st.info("All glucose readings are within acceptable ranges.")
        
        # Manual intervention trigger
        st.subheader("🎯 Manual Intervention Test")
        
        test_glucose = st.number_input(
            "Test Glucose Level (mg/dL):",
            min_value=50,
            max_value=400,
            value=200,
            step=5
        )
        
        if st.button("Generate Test Intervention"):
            test_trigger = {
                'type': 'Manual Test',
                'glucose_level': test_glucose,
                'severity': 'High' if test_glucose > 180 else 'Moderate',
                'description': f'Manual test with glucose level {test_glucose} mg/dL',
                'timestamp': datetime.now()
            }
            
            with st.spinner("Generating intervention..."):
                intervention = intervention_engine.generate_intervention(
                    selected_persona,
                    health_data[-5:],
                    test_trigger
                )
                
                st.markdown(f"""
                <div class="intervention-box">
                    <h5>🤖 Test Intervention Response:</h5>
                    <p>{intervention}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Recent activity log
    st.subheader("📝 Recent Activity Log")
    
    # Show last 10 data points in a table
    recent_df = pd.DataFrame([{
        'Time': d.timestamp.strftime('%m/%d %H:%M'),
        'Glucose (mg/dL)': f"{d.glucose_level:.0f}",
        'Meal': f"{d.meal_type} ({d.meal_carbs:.0f}g)" if d.meal_logged else "None",
        'Steps': f"{d.steps:,}",
        'Stress (1-10)': d.stress_level,
        'Sleep (hrs)': f"{d.sleep_hours:.1f}" if d.sleep_hours > 0 else "-"
    } for d in health_data[-10:]])
    
    st.dataframe(recent_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Demo:**
    This application demonstrates how AI-powered health interventions can be personalized based on:
    - Individual patient personas and health conditions
    - Real-time health data (glucose, activity, sleep, stress)
    - Contextual triggers (meal spikes, stress patterns)
    - Empathetic, actionable messaging tailored to personality types
    
    *Built with Streamlit, Plotly, and Anthropic's Claude API*
    """)

if __name__ == "__main__":
    main()