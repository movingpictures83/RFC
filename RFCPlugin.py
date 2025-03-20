#!/usr/bin/env python
# coding: utf-8

# In[1]:
# NEW PREDICTORS: Intron length, gene length, # of introns
# OLD PREDICTORS: Amino acid (sequence) length, GC content, coding sequence length, transcript length

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def plotImportance(x, y, based, outputfile, data=None, orient=None):
 plt.figure(figsize=(10,6))
 if (orient == ""):
  sns.barplot(x=x, y=y)
 else:
  sns.barplot(x=x, y=y, data=data, orient=orient)
 plt.xlabel("Feature Importance")
 plt.ylabel("Features")
 plt.title("Feature Importances from "+based)
 plt.savefig(outputfile)
 plt.show()
  
def runRF(nem_df, X, column, rs, testpct, outputfile):
 X.drop(X.columns[X.columns.str.contains("unnamed", case=False)],axis=1, inplace=True)
 y = nem_df.iloc[:,column] # Classification
 rf = RandomForestClassifier(random_state = rs)
 X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rs, test_size = testpct)
 rf.fit(X_train, y_train)
 y_pred = rf.predict(X_test)
 train_accuracy = rf.score(X_train, y_train)
 print("Train accuracy: " + str(round(train_accuracy,4)))
 test_accuracy = rf.score(X_test, y_test)
 print("Test accuracy: " + str(round(test_accuracy,4)))
 print(classification_report(y_test, y_pred))
 features_imp = pd.DataFrame(rf.feature_importances_, index = X.columns)
 print(features_imp)
 sorted_features = features_imp.sort_values(by=0, ascending=False)
 plotImportance(sorted_features[0], sorted_features.index, "Random Forest Classifier",outputfile)

import PyPluMA
import PyIO
class RFCPlugin:
 def input(self, inputfile):
    self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
   nem_df = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["original"], encoding = "latin-1")
   X = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["filtered"], encoding = "latin-1")
   runRF(nem_df, X, int(self.parameters["catcol"]), int(self.parameters["numrds"]), float(self.parameters["testpct"]), outputfile)

