import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('datasets/student-por.csv', sep=';')
df = pd.get_dummies(df, drop_first=True)
features_to_scale = [
    'age', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2',
    'freetime'
]

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

x = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R² Score: {r2:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def questions():
    age = float(input("Âge de l'élève : "))
    traveltime = float(input("Temps de trajet (1: <15 min, 2: 15-30 min, 3: 30-60 min, 4: >60 min) : "))
    studytime = float(input("Temps d'étude (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h) : "))
    failures = float(input("Nombre d'échecs scolaires : "))
    absences = float(input("Nombre d'absences : "))
    G1 = float(input("Première note (G1) : "))
    G2 = float(input("Deuxième note (G2) : "))
    freetime = float(input("Temps libre (1 à 5) : "))

    new_data = pd.DataFrame({
        'age': [age], 'traveltime': [traveltime], 'studytime': [studytime], 
        'failures': [failures], 'absences': [absences], 'G1': [G1], 'G2': [G2], 
        'freetime': [freetime]
    })

    new_data[features_to_scale] = scaler.transform(new_data[features_to_scale])

    for col in x.columns:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[x.columns]

    prediction = model.predict(new_data)[0]
    print(f"La note finale prédite (G3) est : {prediction:.2f}")

questions()