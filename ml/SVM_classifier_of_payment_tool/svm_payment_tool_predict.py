# SVM classifier to detect different types of events
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import model_selection  # Cross validation

input_file = 'D:\\Python\github\\basic\ml\\SVM_classifier_of_payment_tool\\payment_multiclass.txt'

# Reading the data
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        #X.append([data[0]] + data[2:])
        X.append(data[2:])
X = np.array(X)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    # �ж��ַ����Ƿ��Ǵ����֣�����С���㲻���ԣ�
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Build SVM
params = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'}
classifier = SVC(**params)
classifier.fit(X, y)

# Cross validation
accuracy = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print("Cross validation: Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + "%")

# Testing encoding on single data instance
input_data = ['F', '20', '9999']
input_data_encoded = [-1] * len(input_data)  # [-1, -1, -1]

count = 0
# ��һ���ɱ��������ݶ������Ϊһ����������
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1
input_data_encoded = np.array(input_data_encoded)
input_data_encoded = input_data_encoded.reshape(1, len(input_data))  # ������תΪһά���飬����predict�����������С��Ҫ��

# Predict and print output for a particular datapoint
output_class = classifier.predict(input_data_encoded)
print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])
