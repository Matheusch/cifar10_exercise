import os
 
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import svm

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.model_selection import cross_val_score

def load_data(train_batches):
    data = []
    labels = []
    for data_batch_i in train_batches:
        featuresFile = np.load(data_batch_i)
        data.append(featuresFile['representations'])
        labels.append(featuresFile['labels'])
        
    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels


def apply_svm(features, labels):  

    trainedModelFilename = str(features.shape[0]) + '_C005_trained_model.pkl'
    clf = []
    from sklearn.externals import joblib

    if os.path.exists(trainedModelFilename):
        clf = joblib.load(trainedModelFilename)
        testFeatures, testLabels = load_data(["test_batch_tensorflow.npz"])
        print ('To be predicted...')
        score = clf.score(testFeatures, testLabels)
        print ('final score: ', score)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)        

        C=0.005
        clf = svm.LinearSVC(C=C, dual=False, verbose=1)
        print('C =', C)
        scores = cross_val_score(clf, features, labels, cv=2, n_jobs=-1, verbose=1)
                                 #scoring='f1_macro')
        
        print ('C-V scores: ', scores)
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        clf.fit(features, labels)
        joblib.dump(clf, trainedModelFilename)
       
 
##        y_predict=clf.predict(X_test)
##        score = clf.score(X_test, y_test)
##        print ('\nscore on validation data', score)
## 
##        labels=sorted(list(set(labels)))
##        print("\nConfusion matrix:")
##        print("Labels: {0}\n".format(",".join(str(labels))))
##        print(confusion_matrix(y_test, y_predict, labels=labels))
## 
##        print("\nClassification report:")
##        print(classification_report(y_test, y_predict))


if __name__ == "__main__":

    trainData, trainLabels = load_data(["data_batch_{}_tensorflow.npz".format(i) for i in range(1, 6)])
    apply_svm(trainData, trainLabels)

