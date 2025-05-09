from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class PredictionForm(forms.ModelForm):
    # Personal Information
    age = forms.IntegerField(min_value=17, max_value=90, required=True)
    sex = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')], required=True)
    race = forms.ChoiceField(choices=[
        ('White', 'White'),
        ('Black', 'Black'),
        ('Asian-Pac-Islander', 'Asian/Pacific Islander'),
        ('Amer-Indian-Eskimo', 'American Indian/Eskimo'),
        ('Other', 'Other')
    ], required=True)
    native_country = forms.ChoiceField(choices=[
        ('United-States', 'United States'),
        ('Canada', 'Canada'),
        ('Mexico', 'Mexico'),
        ('Philippines', 'Philippines'),
        ('Germany', 'Germany'),
        ('India', 'India'),
        ('Japan', 'Japan'),
        ('China', 'China'),
        ('England', 'England'),
        ('Other', 'Other')
    ], required=True)
    
    # Education
    education = forms.ChoiceField(choices=[
        ('Bachelors', 'Bachelors'),
        ('HS-grad', 'High School Graduate'),
        ('Some-college', 'Some College'),
        ('Masters', 'Masters'),
        ('Assoc-voc', 'Associate Degree (Vocational)'),
        ('Assoc-acdm', 'Associate Degree (Academic)'),
        ('Doctorate', 'Doctorate'),
        ('Prof-school', 'Professional School'),
        ('9th', '9th Grade'),
        ('10th', '10th Grade'),
        ('11th', '11th Grade'),
        ('12th', '12th Grade (No Diploma)'),
        ('1st-4th', '1st-4th Grade'),
        ('5th-6th', '5th-6th Grade'),
        ('7th-8th', '7th-8th Grade'),
        ('Preschool', 'Preschool')
    ], required=True)
    education_num = forms.IntegerField(min_value=1, max_value=16, required=True)
    
    # Employment
    workclass = forms.ChoiceField(choices=[
        ('Private', 'Private'),
        ('Self-emp-not-inc', 'Self Employed (Not Incorporated)'),
        ('Self-emp-inc', 'Self Employed (Incorporated)'),
        ('Federal-gov', 'Federal Government'),
        ('Local-gov', 'Local Government'),
        ('State-gov', 'State Government'),
        ('Without-pay', 'Without Pay'),
        ('Never-worked', 'Never Worked')
    ], required=True)
    occupation = forms.ChoiceField(choices=[
        ('Exec-managerial', 'Executive/Managerial'),
        ('Prof-specialty', 'Professional Specialty'),
        ('Tech-support', 'Tech Support'),
        ('Sales', 'Sales'),
        ('Admin-clerical', 'Administrative/Clerical'),
        ('Craft-repair', 'Craft/Repair'),
        ('Machine-op-inspct', 'Machine Operator/Inspector'),
        ('Transport-moving', 'Transportation/Moving'),
        ('Handlers-cleaners', 'Handlers/Cleaners'),
        ('Farming-fishing', 'Farming/Fishing'),
        ('Protective-serv', 'Protective Service'),
        ('Priv-house-serv', 'Private House Service'),
        ('Armed-Forces', 'Armed Forces'),
        ('Other-service', 'Other Service')
    ], required=True)
    hours_per_week = forms.IntegerField(min_value=1, max_value=99, required=True)
    relationship = forms.ChoiceField(choices=[
        ('Husband', 'Husband'),
        ('Wife', 'Wife'),
        ('Own-child', 'Own Child'),
        ('Not-in-family', 'Not in Family'),
        ('Unmarried', 'Unmarried'),
        ('Other-relative', 'Other Relative')
    ], required=True, help_text="Your relationship status in your household")
    marital_status = forms.ChoiceField(choices=[
        ('Married-civ-spouse', 'Married (civilian spouse)'),
        ('Divorced', 'Divorced'),
        ('Never-married', 'Never married'),
        ('Separated', 'Separated'),
        ('Widowed', 'Widowed'),
        ('Married-spouse-absent', 'Married (spouse absent)'),
        ('Married-AF-spouse', 'Married (Armed Forces spouse)')
    ], required=True)
    
    # Financial
    capital_gain = forms.IntegerField(min_value=0, required=True)
    capital_loss = forms.IntegerField(min_value=0, required=True)
    
    class Meta:
        model = UserProfile
        fields = [
            'age', 'sex', 'race', 'native_country',
            'education', 'education_num',
            'workclass', 'occupation', 'hours_per_week', 'relationship', 'marital_status',
            'capital_gain', 'capital_loss'
        ]
