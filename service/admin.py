from django.contrib import admin
from service.models import UserProfile, Prediction, Result

# UserProfile Admin
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user',)  # Wrap 'user' in a tuple

admin.site.register(UserProfile, UserProfileAdmin)

# Prediction Admin
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'dpf', 'age')

admin.site.register(Prediction, PredictionAdmin)

# Result Admin
class ResultAdmin(admin.ModelAdmin):
    list_display = ('prediction', 'result', 'log_reg_prediction', 'dt_prediction', 'rf_prediction', 'log_reg_accuracy', 'dt_accuracy', 'rf_accuracy')  # Correct the field name

admin.site.register(Result, ResultAdmin)
