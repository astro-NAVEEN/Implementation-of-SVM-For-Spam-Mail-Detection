# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the email dataset containing email text and corresponding labels (Spam or Ham).
2. Convert the email text into numerical features using TF-IDF Vectorizer and split the dataset into training and testing sets.
3. Train the Support Vector Machine (SVM) classifier using the training data and predict the class labels for the test data.
4. Evaluate the model using accuracy and predict whether a new email is Spam or Ham.

## Program:

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
emails = [
    "Congratulations! You have won a free lottery ticket.",
    "Meeting is scheduled at 10 AM tomorrow.",
    "Claim your free prize now.",
    "Project report submission is due today.",
    "Win cash rewards by clicking this link.",
    "Please attend the seminar tomorrow.",
    "Exclusive offer! Buy one get one free.",
    "Lunch meeting at 1 PM.",
    "Limited time offer! Click here to win.",
    "Your assignment has been approved."
]

# Labels: 1 = Spam, 0 = Ham
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(emails)

y = labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM Classifier
model = SVC(kernel='linear')

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Labels:")

print(y_test)

print("\nPredicted Labels:")

print(y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# Predict a new email
new_email = ["Congratulations! You have won a free mobile phone."]

new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

if prediction[0] == 1:

    print("\nPrediction: Spam Mail")
else:

    print("\nPrediction: Ham Mail")

## Output:

<img width="802" height="670" alt="image" src="https://github.com/user-attachments/assets/87d94109-6a45-4b78-9417-81e343c7f66f" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
