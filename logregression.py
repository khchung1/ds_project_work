import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config


def logistic_regression(sales, profit, state, subcat):
    a = pd.read_csv('final_sample.csv')
    y = a.iloc[:, 0]
    X = a.iloc[:, 1:]

    oversample = SMOTENC(categorical_features=[2, 3])
    X, y = oversample.fit_resample(X, y)

    #Feature Scaling on Numeric Features
    numeric_features = ["Sales", "Profit"]
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    #One-Hot Encode on Categorical Features
    categorical_features = ["State", "Sub-Category"]
    categorical_transformer = OneHotEncoder(drop='first')


    #Column Transformer with Mixed Types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    #Define Pipeline with Preprocessing Transformers and LogRegression Model
    lr_pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(C=100, max_iter=50,
                                                                                 solver='lbfgs', random_state=1))])
    set_config(display="diagram")

    #Split dataset to Train Set and Test Set, Ratio 9:1 with Stratified Sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

    #Run the Pipeline Model
    lr_pipe.fit(X_train, y_train)

    #Assess the Model's Performance on Unseen Data (Test Set)
    y_pred = lr_pipe.predict(X_test)

    #Print Confusion Matrix Report
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(classification_report(y_test, y_pred))

    #Predict using Single Entry
    input_array = np.array([[sales, profit, state, subcat]])
    input_df = pd.DataFrame(input_array, columns=['Sales', 'Profit', 'State', 'Sub-Category'])
    try:
        y_pred = lr_pipe.predict(input_df)
        y_prob = lr_pipe.predict_proba(input_df)

        y_results = {
            'prediction': y_pred[0],
            'probability': round(y_prob[:,1][0], 2)
        }
    except ValueError:
        y_results = 'Error Occurs'

    return y_results