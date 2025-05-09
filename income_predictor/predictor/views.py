import json
import pickle
import numpy as np
import os
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
from .models import UserProfile, Prediction, Advice
from .forms import UserRegistrationForm, PredictionForm

def home(request):
    return render(request, 'predictor/home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'predictor/register.html', {'form': form})

@login_required
def dashboard(request):
    predictions = Prediction.objects.filter(user=request.user)
    
    # Prepare chart data
    chart_data = {
        'labels': [],
        'confidence': [],
        'predictions': []
    }
    
    # Get the last 10 predictions in chronological order
    recent_predictions = predictions.order_by('date')[:10]
    
    for prediction in recent_predictions:
        chart_data['labels'].append(prediction.date.strftime('%Y-%m-%d'))
        chart_data['confidence'].append(prediction.confidence)
        chart_data['predictions'].append(1 if prediction.prediction else 0)
    
    context = {
        'predictions': predictions,
        'chart_data': json.dumps(chart_data)
    }
    
    return render(request, 'predictor/dashboard.html', context)

@login_required
def predict(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Save user profile
            user_profile = form.save(commit=False)
            user_profile.user = request.user
            user_profile.save()
            
            # Use the trained model to make a prediction
            prediction_result = make_prediction(user_profile)
            
            # Save prediction
            prediction = Prediction.objects.create(
                user=request.user,
                user_profile=user_profile,
                prediction=prediction_result['prediction'],
                confidence=prediction_result['confidence']
            )
            
            # Generate and save advice
            generate_advice(prediction, prediction_result['prediction'])
            
            return redirect('results', prediction_id=prediction.id)
    else:
        form = PredictionForm()
    
    return render(request, 'predictor/predict.html', {'form': form})

@login_required
def results(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    
    # Group advice by category
    advice_by_category = {}
    for advice in prediction.advice.all():
        if advice.category not in advice_by_category:
            advice_by_category[advice.category] = []
        advice_by_category[advice.category].append(advice)
    
    context = {
        'prediction': prediction,
        'advice_by_category': advice_by_category
    }
    
    return render(request, 'predictor/results.html', context)

def make_prediction(user_profile):
    """
    Use the trained model to make a prediction.
    """
    try:
        # Load the trained model
        model_path = os.path.join(settings.BASE_DIR, 'income_prediction_model.pkl')
        features_path = os.path.join(settings.BASE_DIR, 'selected_features.pkl')
        preprocessor_path = os.path.join(settings.BASE_DIR, 'preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(features_path) and os.path.exists(preprocessor_path):
            # Load the model and related files
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            with open(features_path, 'rb') as file:
                selected_features = pickle.load(file)
            
            with open(preprocessor_path, 'rb') as file:
                preprocessor = pickle.load(file)
            
            # Prepare the input data
            input_data = {}
            for feature in selected_features:
                if hasattr(user_profile, feature):
                    input_data[feature] = getattr(user_profile, feature)
            
            # Convert to DataFrame (single row)
            import pandas as pd
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction_proba = model.predict_proba(input_df)
            prediction = prediction_proba[0][1] >= 0.5  # Threshold at 0.5
            confidence = prediction_proba[0][1] * 100 if prediction else (1 - prediction_proba[0][1]) * 100
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
        else:
            # Fall back to rule-based prediction if model files don't exist
            return rule_based_prediction(user_profile)
    except Exception as e:
        print(f"Error using trained model: {e}")
        # Fall back to rule-based prediction
        return rule_based_prediction(user_profile)

def rule_based_prediction(user_profile):
    """
    Fallback rule-based prediction if the model can't be loaded.
    """
    # Simple rules for demo purposes
    high_income_probability = 0.0
    
    # Education factor
    if user_profile.education in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']:
        high_income_probability += 0.3
    elif user_profile.education in ['Assoc-voc', 'Assoc-acdm', 'Some-college']:
        high_income_probability += 0.15
    
    # Occupation factor
    if user_profile.occupation in ['Exec-managerial', 'Prof-specialty']:
        high_income_probability += 0.25
    elif user_profile.occupation in ['Tech-support', 'Sales']:
        high_income_probability += 0.15
    
    # Hours worked factor
    if user_profile.hours_per_week > 40:
        high_income_probability += 0.15
    
    # Capital gain factor
    if user_profile.capital_gain > 0:
        high_income_probability += 0.15
    
    # Age factor
    if 30 <= user_profile.age <= 50:
        high_income_probability += 0.1
    
    # Marital status factor
    if user_profile.marital_status in ['Married-civ-spouse', 'Married-AF-spouse']:
        high_income_probability += 0.1
    
    # Ensure probability is between 0 and 1
    high_income_probability = max(0.1, min(0.9, high_income_probability))
    
    # Convert to percentage
    confidence = high_income_probability * 100
    
    # Determine prediction
    prediction = high_income_probability >= 0.5
    
    return {
        'prediction': prediction,
        'confidence': confidence
    }

def generate_advice(prediction, is_high_income):
    """
    Generate personalized financial advice based on the prediction.
    """
    # Career advice
    if is_high_income:
        Advice.objects.create(
            prediction=prediction,
            category='career',
            advice="Consider negotiating for leadership roles or mentoring opportunities to leverage your high-income potential."
        )
        Advice.objects.create(
            prediction=prediction,
            category='career',
            advice="Explore specialized certifications in your field to further increase your market value."
        )
    else:
        Advice.objects.create(
            prediction=prediction,
            category='career',
            advice="Consider upskilling in high-demand areas like data analysis or project management to increase your income potential."
        )
        Advice.objects.create(
            prediction=prediction,
            category='career',
            advice="Networking can open doors to better opportunities. Join professional groups in your industry."
        )
    
    # Education advice
    if is_high_income:
        Advice.objects.create(
            prediction=prediction,
            category='education',
            advice="Consider advanced degrees or executive education programs to maintain your competitive edge."
        )
    else:
        Advice.objects.create(
            prediction=prediction,
            category='education',
            advice="Investing in education, such as completing a degree or certification, could significantly increase your income potential."
        )
    
    # Investment advice
    if is_high_income:
        Advice.objects.create(
            prediction=prediction,
            category='investment',
            advice="With your income level, consider diversifying investments across stocks, bonds, and real estate."
        )
        Advice.objects.create(
            prediction=prediction,
            category='investment',
            advice="Maximize tax-advantaged retirement accounts like 401(k) and IRA to optimize your high income."
        )
    else:
        Advice.objects.create(
            prediction=prediction,
            category='investment',
            advice="Start with low-cost index funds to build wealth gradually while minimizing risk."
        )
        Advice.objects.create(
            prediction=prediction,
            category='investment',
            advice="Consider micro-investing apps to begin building a portfolio with small amounts."
        )
    
    # Savings advice
    if is_high_income:
        Advice.objects.create(
            prediction=prediction,
            category='savings',
            advice="Aim to save at least 20% of your income and consider automating transfers to savings accounts."
        )
    else:
        Advice.objects.create(
            prediction=prediction,
            category='savings',
            advice="Build an emergency fund covering 3-6 months of expenses before focusing on other financial goals."
        )
    
    # General advice
    Advice.objects.create(
        prediction=prediction,
        category='general',
        advice="Regularly review and adjust your financial plan as your life circumstances change."
    )
