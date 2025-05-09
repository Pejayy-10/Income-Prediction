from django.contrib import admin
from .models import UserProfile, Prediction, Advice

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'age', 'education', 'occupation', 'hours_per_week')
    list_filter = ('education', 'workclass', 'marital_status')
    search_fields = ('user__username', 'occupation')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'confidence', 'date')
    list_filter = ('prediction', 'date')
    search_fields = ('user__username',)
    date_hierarchy = 'date'

@admin.register(Advice)
class AdviceAdmin(admin.ModelAdmin):
    list_display = ('prediction', 'category', 'advice')
    list_filter = ('category',)
    search_fields = ('advice', 'prediction__user__username')
