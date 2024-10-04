import matplotlib
matplotlib.use('Agg')

from django.shortcuts import render,redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login 
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from service.models import UserProfile
from service.models import Prediction, Result

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import os


def home(request):
    return render(request,"home.html")

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Use Django's authenticate method to check the user's credentials
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Log the user in using Django's login method
            auth_login(request, user)  # Pass the user object to login
            return redirect('detail')  # Redirect to the homepage after login
        else:
            # If the username does not exist, prompt the user to register
            if not User.objects.filter(username=username).exists():
                messages.error(request, f"Username '{username}' not found. Please register.")
                return redirect('register')  # Redirect to the registration page
            else:
                # Invalid login credentials, show error message
                messages.error(request, "Invalid password. Please try again.")
                return render(request, 'login.html')

    # If the request method is GET, simply render the login page
    return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')

        # Initialize an empty dictionary for errors
        errors = {}

        # Check for username validity
        if not username:
            errors['username'] = "Username is required."
        elif User.objects.filter(username=username).exists():
            errors['username'] = "Username already exists."

        # Check for email validity
        if not email:
            errors['email'] = "Email is required."
        elif User.objects.filter(email=email).exists():
            errors['email'] = "Email already exists."

        # Check for password validity
        if not password:
            errors['password'] = "Password is required."
        elif len(password) < 6:
            errors['password'] = "Password must be at least 6 characters long."

        # Check if passwords match
        if password != password2:
            errors['password2'] = "Passwords do not match."

        # If there are no errors, save the user
        if not errors:
            user = User(username=username, email=email)
            user.set_password(password)  # Use set_password for hashing
            user.save()
            
            UserProfile.objects.create(user=user)
            
            messages.success(request, "Registration successful! You can now log in.")
            return redirect('login')  # Redirect to the login page

        # If there are errors, send them to the template
        for field, error in errors.items():
            messages.error(request, f"{field.capitalize()}: {error}")

    # Render the form page
    return render(request, "register.html")

@login_required
def detail(request):
    user = request.user

    # Fetch all predictions for the current user
    predictions = Prediction.objects.filter(user=user).order_by('-created_at')

    # Prepare a list to hold the combined data (Prediction and Result)
    combined_data = []

    for prediction in predictions:
        # Get the result associated with this prediction, if it exists
        result = Result.objects.filter(prediction=prediction).first()

        # Append both prediction and result to the list as a dictionary
        combined_data.append({
            'prediction': prediction,
            'result': result
        })

    return render(request, "detail.html", {'combined_data': combined_data, 'user': user})

def predict(request):
    if request.method == 'POST':
        try:
            # Create a new Prediction object
            prediction = Prediction(
                user=request.user,
                pregnancies=float(request.POST.get('pregnancies')),
                glucose=float(request.POST.get('glucose')),
                blood_pressure=float(request.POST.get('bloodPressure')),
                skin_thickness=float(request.POST.get('skinThickness')),
                insulin=float(request.POST.get('insulin')),
                bmi=float(request.POST.get('BMI')),
                dpf=float(request.POST.get('DPF')),
                age=float(request.POST.get('age'))
            )
            prediction.save()
            
            # Redirect to the result view with the prediction id
            return redirect('result', prediction_id=prediction.id)
        except ValueError as e:
            messages.error(request, f"Invalid input: {str(e)}")
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
    
    return render(request, "predict.html")

def result(request,prediction_id):
    # Retrieve the Prediction instance
    prediction_instance = get_object_or_404(Prediction, id=prediction_id)

    # Load dataset
    data = pd.read_csv(r"D:\BCA 8th sem\project_practical\diabetes_detection_machine_learning\diabetes_detection\diabetes.csv")

    # Prepare data
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the directory for saving confusion matrices if it doesn't exist
    confusion_matrix_dir = r"D:\BCA 8th sem\project_practical\diabetes_detection_machine_learning\diabetes_detection\static\confusion_matrices"
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)

    # Train models
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, Y_train)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, Y_train)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, Y_train)

    # Predictions
    log_reg_pred = log_reg.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Precision, Recall, and Accuracy
    log_reg_precision = precision_score(Y_test, log_reg_pred)
    log_reg_recall = recall_score(Y_test, log_reg_pred)
    log_reg_accuracy = accuracy_score(Y_test, log_reg_pred)

    dt_precision = precision_score(Y_test, dt_pred)
    dt_recall = recall_score(Y_test, dt_pred)
    dt_accuracy = accuracy_score(Y_test, dt_pred)

    rf_precision = precision_score(Y_test, rf_pred)
    rf_recall = recall_score(Y_test, rf_pred)
    rf_accuracy = accuracy_score(Y_test, rf_pred)

    print('Precision, Recall, Accuracy')
    print(f'Logistic Regression: Precision={log_reg_precision}, Recall={log_reg_recall}, Accuracy={log_reg_accuracy}')
    print(f'Decision Tree: Precision={dt_precision}, Recall={dt_recall}, Accuracy={dt_accuracy}')
    print(f'Random Forest: Precision={rf_precision}, Recall={rf_recall}, Accuracy={rf_accuracy}')


    # Confusion Matrices
    log_reg_cm = confusion_matrix(Y_test, log_reg_pred)
    dt_cm = confusion_matrix(Y_test, dt_pred)
    rf_cm = confusion_matrix(Y_test, rf_pred)

    # Plotting confusion matrices
    def plot_confusion_matrix(cm, model_name):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        file_path = os.path.join(confusion_matrix_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(file_path)
        plt.close()
        return file_path

    log_reg_cm_path = plot_confusion_matrix(log_reg_cm, "Logistic Regression")
    dt_cm_path = plot_confusion_matrix(dt_cm, "Decision Tree")
    rf_cm_path = plot_confusion_matrix(rf_cm, "Random Forest")

    # Retrieve form values from POST request and ensure they are safe
    val1 = prediction_instance.pregnancies
    val2 = prediction_instance.glucose
    val3 = prediction_instance.blood_pressure
    val4 = prediction_instance.skin_thickness
    val5 = prediction_instance.insulin
    val6 = prediction_instance.bmi
    val7 = prediction_instance.dpf
    val8 = prediction_instance.age

    input_data = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    print(f"Input Data: Pregnancies={val1}, Glucose={val2}, Blood Pressure={val3}, Skin Thickness={val4}, Insulin={val5}, BMI={val6}, DPF={val7}, Age={val8}")

    input_data = scaler.transform(input_data)

    # Predict for input data
    log_reg_input_pred = log_reg.predict(input_data)
    dt_input_pred = dt_model.predict(input_data)
    rf_input_pred = rf_model.predict(input_data)
    
    # Majority Voting
    final_prediction = max([log_reg_input_pred[0], dt_input_pred[0], rf_input_pred[0]], 
                           key=[log_reg_input_pred[0], dt_input_pred[0], rf_input_pred[0]].count)
    
    
    result = "Positive" if final_prediction == 1 else "Negative"

    print("Logistic Regression Confusion Matrix:", log_reg_cm)
    print("Decision Tree Confusion Matrix:", dt_cm)
    print("Random Forest Confusion Matrix:", rf_cm)

    # Create and save Result instance
    try:
        result_instance = Result.objects.create(
            prediction=prediction_instance,
            result=result,
            log_reg_prediction=bool(log_reg_pred[0]),
            dt_prediction=bool(dt_pred[0]),
            rf_prediction=bool(rf_pred[0]),
            log_reg_accuracy=log_reg_accuracy,
            dt_accuracy=dt_accuracy,
            rf_accuracy=rf_accuracy
        )
    except Exception as e:
        print("Error saving result:", str(e))
        messages.error(request, "Failed to save results.")

    # Prepare context data for rendering
    context = {
        "result2": result,
        "log_reg_precision": log_reg_precision,
        "log_reg_recall": log_reg_recall,
        "log_reg_accuracy": log_reg_accuracy,
        "dt_precision": dt_precision,
        "dt_recall": dt_recall,
        "dt_accuracy": dt_accuracy,
        "rf_precision": rf_precision,
        "rf_recall": rf_recall,
        "rf_accuracy": rf_accuracy,
        "log_reg_cm_path": log_reg_cm_path,
        "dt_cm_path": dt_cm_path,
        "rf_cm_path": rf_cm_path,
    }

    return render(request, 'result.html', context)