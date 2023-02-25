# Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. 
# Each example is a 28x28 grayscale image, associated with a label from 10 classes.


'''Labels

Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot '''



from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train1 = x_train/255
x_test1 = x_test/255

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

m_classifier = Sequential()
m_classifier.add(Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
m_classifier.add(MaxPooling2D(pool_size=(2,2)))
m_classifier.add(Conv2D(64,(3,3),activation='relu',))
m_classifier.add(MaxPooling2D(pool_size=(2,2)))
m_classifier.add(Flatten())
m_classifier.add(Dense(250,activation='relu'))
m_classifier.add(Dense(10, activation='softmax'))

m_classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

m_classifier.fit(x_train1,y_train,batch_size=500,epochs=3,validation_data=(x_test1,y_test))

m_classifier.evaluate(x_test1,y_test)
