from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------- Dataset ----------------
df = pd.read_csv("Social_Network_Ads.csv")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Age', 'EstimatedSalary', 'Gender']].values
y = df['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(random_state=0, max_iter=300)
classifier.fit(X_train, y_train)

# ---------------- Flask App ----------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form.get("age"))
        salary = int(request.form.get("salary"))
        gender_input = request.form.get("gender")
        gender = 1 if gender_input.lower() == "male" else 0

        # Prepare data for prediction
        data = np.array([[age, salary, gender]])
        scaled = sc.transform(data)

        prob = classifier.predict_proba(scaled)[0][1]
        prediction = "WILL Purchase ✔" if prob >= 0.45 else "Will NOT Purchase ❌"

        return jsonify({
            "prediction": prediction,
            "probability": f"{prob:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
