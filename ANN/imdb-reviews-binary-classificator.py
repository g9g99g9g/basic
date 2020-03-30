import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import Sequential, layers

# based on <https://geektutu.com/post/tf2doc-rnn-lstm-text.html>
# <https://tensorflow.google.cn/datasets/catalog/imdb_reviews>
ds, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_ds, test_ds = ds['train'], ds['test']

BUFFER_SIZE, BATCH_SIZE = 10000, 64  # batch=64时每次迭代耗时约25秒；batch=1K时卡死
# batch=100时每次迭代耗时约35秒；batch=10时每次迭代1秒

train_ds = train_ds.shuffle(BUFFER_SIZE)  # buffer_size:表示新数据集将从原数据集中采样的元素数
train_ds = train_ds.padded_batch(BATCH_SIZE, train_ds.output_shapes)
test_ds = test_ds.padded_batch(BATCH_SIZE, test_ds.output_shapes)

tokenizer = info.features['text'].encoder  # 通过 tfds 获取到的数据已经经过了文本预处理，即 Tokenizer，向量化文本(将文本转为数字序列)
print('词汇个数:', tokenizer.vocab_size)

sample_str = 'I am YSD'
tokenized_str = tokenizer.encode(sample_str)
print('I am YSD的向量化文本:', tokenized_str)

for ts in tokenized_str:
    print(ts, '-->', tokenizer.decode([ts]))

# ----------------- Bi-LSTM model -----------------
model = Sequential([layers.Embedding(tokenizer.vocab_size, 64), \
                    layers.Bidirectional(layers.LSTM(64, return_sequences=True)), \
                    layers.Bidirectional(layers.LSTM(32, dropout=0.1, recurrent_dropout=0.2)), \
                    layers.Dense(64, activation='relu'), \
                    layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 设置该模型的学习流程：二元交叉熵、ADAM
history = model.fit(train_ds, epochs=1, validation_data=test_ds)
loss, acc = model.evaluate(test_ds)
print('准确率:', acc)

# ----------------- 数据可视化 -----------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20

def plot_graphs(history, name):
    plt.plot(history.history[name])
    plt.plot(history.history['验证集 - '+ name])
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.legend([name, '验证集 - ' + name])
    plt.show()

plot_graphs(history, 'accuracy')
