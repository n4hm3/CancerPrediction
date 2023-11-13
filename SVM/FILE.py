import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


cancer = datasets.load_breast_cancer()

for i in range(5):
    x = cancer.data
    y = cancer.target

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    print('patient ',i+1,':')

    classes = ['malignant','benign']

    clf = svm.SVC(kernel='linear',C=1)
    clf.fit(x_train, y_train)

    y_prediction = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_prediction)

    if y_prediction[i] == 0:
        print(classes[0])
    else:
        print(classes[1])
    print(acc)
