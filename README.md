# arbol de desicion 
datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)


modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 20, max_features = 'sqrt', criterion = "gini", max_depth=4))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred
labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC


def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}
indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))
labels = ["Si", "No"]
ks = list(range(2, 20))
errores = []

for k in ks:
  modelo_k = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = k))
  ])
  no_print = modelo_k.fit(X_train, y_train)
  pred = modelo_k.predict(X_test)
  MC = confusion_matrix(y_test, pred, labels=labels)
  indices = indices_general(MC, labels)
  errores.append(indices["Error Global"])

fig, ax = plt.subplots()
no_print = ax.plot(ks, errores)
no_print = ax.set_xlabel("Cantidad de Árboles")
no_print = ax.set_ylabel("Error Global")
plt.show()

#potenciacion

datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()
datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()
X = datos.loc[:, datos.columns != 'BuenPagador']
X
y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

arbol = DecisionTreeClassifier(max_depth=4)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', AdaBoostClassifier(n_estimators = 20, learning_rate = 1.0, estimator = arbol))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred


labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC



def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))

labels = ["Si", "No"]
ks = list(range(2, 20))
errores = []

for k in ks:
  modelo_k = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', AdaBoostClassifier(n_estimators = k))
  ])
  no_print = modelo_k.fit(X_train, y_train)
  pred = modelo_k.predict(X_test)
  MC = confusion_matrix(y_test, pred, labels=labels)
  indices = indices_general(MC, labels)
  errores.append(indices["Error Global"])

fig, ax = plt.subplots()
no_print = ax.plot(ks, errores)
no_print = ax.set_xlabel("Cantidad de Árboles")
no_print = ax.set_ylabel("Error Global")
plt.show()

#potenciacion con gradientes

datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X
y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', GradientBoostingClassifier(n_estimators = 20, learning_rate = 1.0, max_features = 'sqrt', max_depth = 4))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred
labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC


def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))

labels = ["Si", "No"]
ks = list(range(2, 20))
errores = []

for k in ks:
  modelo_k = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', GradientBoostingClassifier(n_estimators = k))
  ])
  no_print = modelo_k.fit(X_train, y_train)
  pred = modelo_k.predict(X_test)
  MC = confusion_matrix(y_test, pred, labels=labels)
  indices = indices_general(MC, labels)
  errores.append(indices["Error Global"])

fig, ax = plt.subplots()
no_print = ax.plot(ks, errores)
no_print = ax.set_xlabel("Cantidad de Árboles")
no_print = ax.set_ylabel("Error Global")
plt.show()

#xgboost
datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X

from sklearn.preprocessing import LabelEncoder

codificacion = LabelEncoder()
y = datos.loc[:, 'BuenPagador'].to_numpy()
y = codificacion.fit_transform(y)
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', xgb.XGBClassifier(n_estimators = 20, learning_rate = 1.0, max_depth = 4))
])

modelo.fit(X_train, y_train)
pred = modelo.predict(X_test)
pred
MC = confusion_matrix(y_test, pred)
MC



def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

indices = indices_general(MC, codificacion.classes_.tolist())
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))
labels = codificacion.classes_.tolist()
ks = list(range(2, 20))
errores = []

for k in ks:
  modelo_k = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', xgb.XGBClassifier(n_estimators = k))
  ])
  no_print = modelo_k.fit(X_train, y_train)
  pred = modelo_k.predict(X_test)
  MC = confusion_matrix(y_test, pred)
  indices = indices_general(MC, labels)
  errores.append(indices["Error Global"])

fig, ax = plt.subplots()
no_print = ax.plot(ks, errores)
no_print = ax.set_xlabel("Cantidad de Árboles")
no_print = ax.set_ylabel("Error Global")
plt.show()


datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".") datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category') datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category') datos["MontoCuota"] = datos["MontoCuota"].astype('category') datos["GradoAcademico"] = datos["GradoAcademico"].astype('category') datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador'] X

y = datos.loc[:, 'BuenPagador'].to_numpy() y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer( transformers=[ ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']), ('num', StandardScaler(), ['MontoCredito']) ] )

modelo = Pipeline(steps=[ ('preprocesador', preprocesamiento), ('clasificador', SVC(kernel="rbf", gamma = 5, C = 10)) ])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test) pred

labels = ["Si", "No"] MC = confusion_matrix(y_test, pred, labels=labels) MC

indices = indices_general(MC, labels) for k in indices: print("\n%s:\n%s"%(k,str(indices[k])))

print("\nComparación Final")

resultados = { "Random Forest": metricas_rf, "AdaBoost": metricas_ada, "Gradient Boosting": metricas_gb, "XGBoost": metricas_xgb, "Árbol de Decisión": metricas_dt }

comparacion = pd.DataFrame(columns=[ "Precisión Global", "Error Global", "Precisión Positiva (PP)", "Precisión Negativa (PN)" ])

for nombre, met in resultados.items(): comparacion.loc[nombre] = [ met["Precisión Global"], met["Error Global"], met["Precisión Positiva (PP)"], met["Precisión Negativa (PN)"] ]

comparacion = comparacion.sort_values(by="Precisión Global", ascending=False) print("\nTabla comparativa de modelos:") print(comparacion)

vvmachinr
d = { 'X': [1, 1, 1, 3, 1, 3, 1, 3, 1], 'Y': [0, 0, 1, 1, 1, 2, 2, 2, 1], 'Z': [1, 2, 2, 4, 3, 3, 1, 1, 0], 'Clase': ['Rojo', 'Rojo', 'Rojo', 'Rojo', 'Rojo', 'Azul', 'Azul', 'Azul', 'Azul'] } df = pd.DataFrame(data = d) df

df_rojo = df[df['Clase'] == 'Rojo'] df_azul = df[df['Clase'] == 'Azul']

fig = go.Figure()

no_plot = fig.add_trace( go.Scatter3d( x = df_rojo['X'], y = df_rojo['Y'], z = df_rojo['Z'], mode = 'markers', marker = dict(size = 5, color = 'red'), name = 'Rojo' ) )

no_plot = fig.add_trace( go.Scatter3d( x = df_azul['X'], y = df_azul['Y'], z = df_azul['Z'], mode = 'markers', marker = dict(size = 5, color = 'blue'), name = 'Azul' ) )

fig

P = np.array([1, 1, 2]) Q = np.array([1, 0, 1]) R = np.array([3, 1, 4])

PQ = P - Q PR = P - R

PQ PR

ecuacion = np.cross(PQ, PR) ecuacion

P = np.array([1, 2, 1]) Q = np.array([1, 1, 0]) R = np.array([3, 2, 3])

PQ = P - Q PR = P - R

PQ PR

ecuacion = np.cross(PQ, PR) ecuacion

x, y = np.meshgrid(range(5), range(5)) z_optimo = x + y - 1

no_plot = fig.add_trace( go.Surface( x = x, y = y, z = z_optimo, opacity = .7, showscale = False, colorscale = np.repeat('green', x.size, axis = 0) ) )

fig

no_plot = fig.add_trace( go.Scatter3d( x = [1], y = [1], z = [4], mode = 'markers', marker = dict(size = 5, color = 'blue'), name = 'Azul' ) )

fig








