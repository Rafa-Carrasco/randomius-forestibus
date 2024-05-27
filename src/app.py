from utils import db_connect
engine = db_connect()

# your code here
# 1. descargar data

url = "https://raw.githubusercontent.com/Rafa-Carrasco/arbol-de-regresion/main/data/processed/clean_diabetes_test.csv"
url2 = "https://raw.githubusercontent.com/Rafa-Carrasco/arbol-de-regresion/main/data/processed/clean_diabetes_train.csv"

respuesta = requests.get(url)
nombre_archivo = "clean_diabetes_test.csv"
with open(nombre_archivo, 'wb') as archivo:
    archivo.write(respuesta.content)

respuesta = requests.get(url2)
nombre_archivo = "clean_diabetes_train.csv"
with open(nombre_archivo, 'wb') as archivo:
    archivo.write(respuesta.content)

# 2. convertir csv en dataframe

X_tra = pd.read_csv("../data/processed/clean_diabetes_train.csv")
X_tes = pd.read_csv("../data/processed/clean_diabetes_test.csv")


# Separar las características y la variable objetivo para el conjunto de entrenamiento
X_train = X_tra.drop(columns='Outcome')
y_train = X_tra['Outcome']

# Separar las características y la variable objetivo para el conjunto de prueba
X_test = X_tes.drop(columns='Outcome')
y_test = X_tes['Outcome']

# 3. iniciar y entrenar el modelo CLASIFICACION

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state = 42)
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
from sklearn import tree

fig, axis = plt.subplots(2, 2, figsize = (15, 15))

tree.plot_tree(model.estimators_[0], ax = axis[0, 0], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[1], ax = axis[0, 1], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[2], ax = axis[1, 0], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
tree.plot_tree(model.estimators_[3], ax = axis[1, 1], feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)

plt.show()

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

# optimizacion RF classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rfc = RandomForestClassifier(
        n_estimators=28, 
        max_depth=10, 
        min_samples_split=2,
        min_samples_leaf=20,
        max_features=10,
        random_state=42
    )

rfc.fit(X_train, y_train)
    
# Hacer predicciones en el conjunto de prueba
y_pred = rfc.predict(X_test)
    
# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

from pickle import dump
dump(model, open("RandomForestClassifier_tweaked_42.sav", "wb"))

# 5.regression random forest

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state = 42)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}")
print(f"MAE: {mae}")

# optimizar random forest regression

rfr = RandomForestRegressor(
        n_estimators=100, 
        max_depth=50, 
        min_samples_split=18, 
        min_samples_leaf=3, 
        max_features='log2', 
        random_state=62
    )

rfr.fit(X_train, y_train)
    
# Hacer predicciones en el conjunto de prueba
y_pred = rfr.predict(X_test)
    
# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}")
print(f"MAE: {mae}")

from pickle import dump
dump(model, open("RandomForestRegressor_tweaked_62.sav", "wb"))