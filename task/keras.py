"""
  keras:二分类
  机器学习问题中，二分类和多分类问题是最为常见，下面使用keras在imdb和newswires数据上进行相应的实验
"""

# 1.文本获取
# 2.文本预处理
# 3.定义模型/神经网络
# 4.定义评价指标和优化方法
# 5.划分数据集，进行训练
# 6.测试集对模型进行测试

from keras import layers
from keras import models
from keras.datasets import imdb
import numpy as np
from keras import metrics, losses

# 1. 加载数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 2. 定义模型
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

######################
# input_tensor = layers.Input(shape=(784,))
# x = layers.Dense(32, activation='relu')(input_tensor)
# output_tensor = layers.Dense(10, activation='softmax')(x)
# model = models.Model(input=input_tensor, output=output_tensor)

# 3.1  文本数据预处理
word_index = imdb.get_word_index()  # 获取词索引  id to word
reverse_word_index = dict((value, key) for key, value in word_index.items())  # word to id
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 3.2 特征向量化
def vectorize_sequences(sequences, dim=10000):
    """
    词袋模型，获取词向量
    :param sequences: [[sentence1], [sentence2], [...]]
    :param dim: 词袋大小，词特征维度
    :return:
    """
    results = np.zeros((len(sequences), dim))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model.compile(optimizer='rmsprop',
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])  # metrics 传入list，可以使用多种评价方式

# 划分验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 4. 训练模型
history = model.fit(partial_x_train, partial_y_train,
                      epochs=20,
                      batch_size=512,
                      validation_data=(x_val, y_val))  # 验证集

# history 获取训练过程的acc loss val_acc val_loss
history_dict = history.history
print(history_dict.keys())

# 5. Plotting the training and validation loss

import matplotlib.pyplot as plt

# 画出训练集和验证集的损失和精度变化，分析模型状态
acc = history.history['binary_accuracy']  # 训练集acc
val_acc = history.history['val_binary_accuracy']  # 验证集 acc
loss = history.history['loss']  # 训练损失
val_loss = history.history['val_loss']  # 验证损失

epochs = range(1, len(acc)+1)  # 迭代次数

plt.plot(epochs, loss, 'bo', label='Training loss')  # bo for blue dot 蓝色点
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clar figure


plt.plot(epochs, acc, 'bo', label='Training acc')  # bo for blue dot 蓝色点
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()