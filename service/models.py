from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Add any additional fields you want to store about the user

    def __str__(self):
        return self.username

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pregnancies = models.FloatField()
    glucose = models.FloatField()
    blood_pressure = models.FloatField()
    skin_thickness = models.FloatField()
    insulin = models.FloatField()
    bmi = models.FloatField()
    dpf = models.FloatField()
    age = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class Result(models.Model):
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE)
    result = models.CharField(max_length=10)  # "Positive" or "Negative"
    log_reg_prediction = models.BooleanField()
    dt_prediction = models.BooleanField()
    rf_prediction = models.BooleanField()
    log_reg_accuracy = models.FloatField()
    dt_accuracy = models.FloatField()
    rf_accuracy = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)