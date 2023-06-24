import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
# Load the data
data = pd.read_csv("Smote_data.csv")  # Replace "your_data_file.csv" with the actual file name

# Separate the features (X) and labels (y)
X = data.iloc[:, :-1]
y = data["label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
joblib.dump(classifier,"Trained_data.pkl")
# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred)
print(report)
def predict_cough():
    # Get the feature values from the GUI input fields
    feature_values = []
    for entry in entry_fields:
        feature_values.append(float(entry.get()))

    # Create a numpy array from the input values
    input_data = np.array(feature_values).reshape(1, -1)

    # Make a prediction using the trained model
    prediction = classifier.predict(input_data)

    # Update the result label with the predicted label
    result_label.configure(text="Prediction: " + str(prediction[0]))

# Create the main window
window = tk.Tk()
window.title("Cough Classifier")

# Create a list to hold the entry fields
entry_fields = []

# Add labels and entry fields for each feature
features = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
            'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6',
            'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14',
            'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']

for i, feature in enumerate(features):
    label = ttk.Label(window, text=feature + ":")
    label.grid(row=i, column=0)

    entry = ttk.Entry(window)
    entry.grid(row=i, column=1)

    entry_fields.append(entry)

# Add button to trigger prediction
predict_button = ttk.Button(window, text="Predict", command=predict_cough)
predict_button.grid(row=len(features), column=0, columnspan=2)

# Add label to display the prediction result
result_label = ttk.Label(window, text="Prediction: ")
result_label.grid(row=len(features) + 1, column=0, columnspan=2)

# Run the GUI application
window.mainloop()

