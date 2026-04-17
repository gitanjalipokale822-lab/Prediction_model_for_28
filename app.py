import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Experience Predictor", page_icon="📈")

# Custom CSS for animation and styling
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        animation: fadeIn 1.5s ease-out;
        text-align: center;
        color: #2E7D32;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        transition: all 0.3s ease;
        background-color: #2E7D32;
        color: white;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #1B5E20;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    try:
        # Loading the provided sklearn model 
        with open('model (6).pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header Section
st.markdown("<h1 class='main-header'>📈 Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

if model:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        # Based on feature_names_in_ ['YearsExperience'] 
        experience = st.number_input("Years of Experience:", 
                                     min_value=0.0, 
                                     max_value=50.0, 
                                     value=1.0, 
                                     step=0.5)
        
        predict_btn = st.button("Calculate Prediction")

    with col2:
        st.subheader("Result")
        if predict_btn:
            # Prepare data and predict
            input_data = np.array([[experience]])
            prediction = model.predict(input_data)
            
            # Display result with a "pop" effect
            st.balloons()
            st.success(f"### Predicted Value: {prediction[0]:,.2f}")
            
            # Contextual info from model metadata 
            st.info(f"Model Rank: {model.rank_} | Intercept: {model.intercept_:,.2f}")
else:
    st.warning("Please ensure 'model (6).pkl' is in the same directory.")

# Decorative sidebar
with st.sidebar:
    st.title("About Model")
    st.markdown("""
    This **Linear Regression** model was built using:
    - **Library:** Scikit-Learn 
    - **Feature:** Years of Experience 
    - **Version:** 1.6.1 
    """)
