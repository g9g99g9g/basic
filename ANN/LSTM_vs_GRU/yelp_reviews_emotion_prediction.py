import string
import numpy as np
import tensorflow as tf
import time
from sklearn import metrics

data = ""
#with open('D:\\Python\\Yelp\\predict-yelp-review\\trainSet.txt', 'r') as file:
#with open('D:\\Python\\Yelp\\yelp-dataset\\yelp_academic_dataset_review_1K_short_YES.txt', 'r') as file:
#with open('D:\\Python\\Yelp\\yelp-dataset\\yelp_academic_dataset_review_1K_long_Max326_YES.txt', 'r') as file:
with open('D:\\Python\\Yelp\\yelp-dataset\\yelp_academic_dataset_review_500_short_YES.txt', 'r') as file:
#with open('D:\\Python\\Yelp\\yelp-dataset\\yelp_academic_dataset_review_500_long_Max326_YES.txt', 'r') as file:
    data = file.read().split('\n')  # split the data line by line

num_reviews = len(data)  # total number of reviews

# fill up old_train with a tuple (review, rating)
x_train = []
y_train = []
for sentence in data:
    arr = sentence.split('\t')
    if len(arr) == 2:
        word = arr[0]
        rating = arr[1][1]  # ÿ��ĩβ�����ǿո�Tab�����ǿո�
        x_train.append(word)
        y_train.append(rating)

# strip all punctuation and add the review and rating to their own separate arrays
train = []
for sentence in x_train:
    train.append(sentence.translate(str.maketrans('', '', string.punctuation)).lower())

# take off the trailing space at the end of each sentence
sentences = []
for sentence in train:
    sentences.append(sentence[0:-1])

# get an array of arrays of just the words... ex: [['this', 'food', 'was', 'terrible'], ['awesome', 'food']]
justWords = []
for sentence in sentences:
    justWords.append(sentence.split(' '))

# remove any empty strings in the arrays because when trying to vectorize you will get an error
for arr in justWords:
    for word in arr:
        if word == '':
            arr.remove('')

tokenizer = tf.keras.preprocessing.text.Tokenizer()  # Tokenizer�����ı�ת��������������
tokenizer.fit_on_texts(sentences)  # ���ı�����Tokenizer����ѵ��
text_sequences = np.array(tokenizer.texts_to_sequences(sentences))  # ���ı�תΪ���У��ٽ�����ת��Ϊnp����
sequence_dict = tokenizer.word_index
dictionary = dict((num, val) for (val, num) in sequence_dict.items())

# We get a map of encoding-to-word in sequence_dict
reviews_encoded = []
for i, review in enumerate(justWords):
    reviews_encoded.append([sequence_dict[x] for x in review])  # Generate encoded reviews
    # ע�⣺�����������ո񣬱�����ǰ�����пո񣬲�����˫���ţ������з�Ӣ���ַ�

# we will now add padding to our array to make them all the same size
# kerasֻ�ܽ��ܳ�����ͬ���������룬������Ŀǰ���г��Ȳβ�룬��ʱ��Ҫʹ��pad_sequences()
max = 326
X = tf.keras.preprocessing.sequence.pad_sequences(reviews_encoded, maxlen=max, truncating='post')  # post����β�ض�

# make one hot array for each review
Y = np.array([[0, 1] if '0' in label else [1, 0] for label in y_train])

X_train, Y_train = X[0:799], Y[0:799]
X_val, Y_val = X[800:995], Y[800:995]

t1 = time.time()

# acc��0.82, 28.9s

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(len(dictionary) + 1, max, input_length=max))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences=True, recurrent_dropout=0.5)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(40, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(40, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
print(model.summary())


# acc��0.78, 25.4s
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(len(dictionary) + 1, max, input_length=max))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(40, return_sequences=True, recurrent_dropout=0.5)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(40, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(40, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
print(model.summary())
'''

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.025, decay=0.1), metrics=['acc'])
hist = model.fit(X_train, Y_train, batch_size=120, epochs=5, validation_data=(X_val, Y_val))
acc = model.evaluate(X_val, Y_val)
Y_pred = model.predict_classes(X_val)
#print("Y_true: ", Y_val[:, 1], " Y_pred: ", Y_pred)

t2 = time.time() - t1
print("simulation time is", t2, "s")
#print("accuracy: " + str(acc[1]))
print("accuracy: ", metrics.accuracy_score(y_true=Y_val[:, 1], y_pred=Y_pred))
print("Precision: ", metrics.precision_score(y_true=Y_val[:, 1], y_pred=Y_pred, average='macro'))
print("Recall: ", metrics.recall_score(y_true=Y_val[:, 1], y_pred=Y_pred, average='micro'))
print("F1: ", metrics.f1_score(y_true=Y_val[:, 1], y_pred=Y_pred, average='weighted'))
print("ROC_AUC: ", metrics.roc_auc_score(y_true=Y_val[:, 1], y_score=Y_pred))
