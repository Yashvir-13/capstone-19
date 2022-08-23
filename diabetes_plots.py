import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image  
def app(df):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualise the Diabetes Prediction Web app ")

    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize = (10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)
        bottom, top = ax.get_ylim() 
        ax.set_ylim(bottom + 0.5, top - 0.5) 
        st.pyplot()

    st.subheader("Predictor Selection")
    plot_select = st.selectbox("Select the Classifier to Visualise the Diabetes Prediction:", ('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier')) 

    if plot_select == 'Decision Tree Classifier':
        feature_columns = list(df.columns)
        feature_columns.remove('Pregnancies')
        feature_columns.remove('Skin_Thickness')
        feature_columns.remove('Outcome')

        X = df[feature_columns]
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

        dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        dtree_clf.fit(X_train, y_train) 
        y_train_pred = dtree_clf.predict(X_train)
        y_test_pred = dtree_clf.predict(X_test)

         
        if st.checkbox("Plot confusion matrix"):
            plt.figure(figsize = (10, 6))
            plot_confusion_matrix(dtree_clf, X_train, y_train, values_format = 'd')
            st.pyplot()

        if st.checkbox("Plot Decision Tree"):   
            dot_data = tree.export_graphviz(decision_tree = dtree_clf, max_depth = 3, out_file = None, filled = True, rounded = True,
                feature_names = feature_columns, class_names = ['0', '1'])
            st.graphviz_chart(dot_data)


    if plot_select == 'GridSearchCV Best Tree Classifier':
        feature_columns = list(df.columns)

        feature_columns.remove('Pregnancies')
        feature_columns.remove('Skin_Thickness')
        feature_columns.remove('Outcome')

        X = df[feature_columns]
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

        param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

        grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

        grid_tree.fit(X_train, y_train)
        best_tree = grid_tree.best_estimator_

        grid_tree.fit(X_train, y_train) 
        y_train_pred = grid_tree.predict(X_train)
        y_test_pred = grid_tree.predict(X_test)

         
        if st.checkbox("Plot confusion matrix"):
            plt.figure(figsize = (5, 3))
            plot_confusion_matrix(grid_tree, X_train, y_train, values_format = 'd')
            st.pyplot()

        if st.checkbox("Plot Decision Tree"):   
            dot_data = tree.export_graphviz(decision_tree = best_tree, max_depth = 3, out_file = None, filled = True, rounded = True,
                feature_names = feature_columns, class_names = ['0', '1'])
            st.graphviz_chart(dot_data)