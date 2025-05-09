from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.conf import settings

class UserProfile(models.Model):
    # Personal Information
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='profiles')
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    race = models.CharField(max_length=20)
    native_country = models.CharField(max_length=30)
    
    # Education
    education = models.CharField(max_length=20)
    education_num = models.IntegerField()
    
    # Employment
    workclass = models.CharField(max_length=30)
    occupation = models.CharField(max_length=30)
    hours_per_week = models.IntegerField()
    relationship = models.CharField(max_length=20)
    marital_status = models.CharField(max_length=30)
    
    # Financial
    capital_gain = models.IntegerField()
    capital_loss = models.IntegerField()
    
    def __str__(self):
        return f"{self.user.username}'s Profile - {self.id}"

def get_default_user():
    # This function returns the ID of the first user or creates one if none exists
    User = settings.AUTH_USER_MODEL
    try:
        # Try to get the first user
        return User.objects.first().id
    except:
        # If no user exists, return None (this will cause an error, but it's better than a silent failure)
        return None

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions', default=get_default_user)
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='predictions')
    prediction = models.BooleanField()  # True for >50K, False for <=50K
    confidence = models.FloatField()
    date = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        result = ">50K" if self.prediction else "<=50K"
        return f"{self.user.username}'s Prediction: {result} ({self.confidence:.2f}%)"
    
    class Meta:
        ordering = ['-date']

class Advice(models.Model):
    CATEGORY_CHOICES = (
        ('career', 'Career'),
        ('education', 'Education'),
        ('investment', 'Investment'),
        ('savings', 'Savings'),
        ('general', 'General'),
    )
    
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='advice')
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    advice = models.TextField()
    
    def __str__(self):
        return f"{self.category} advice for {self.prediction.user.username}"
