import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Input
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28,28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def LeNet():
    inputs = Input((img_cols, img_rows, 1))

    #layer 1
    x = Conv2D(filters=6, kernel_size=(5, 5), padding='valid')(inputs)  #input_shape = (28,28,1)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x) 
    x = Activation('sigmoid')(x) 

    #layer 2
    x = Conv2D(filters=16, kernel_size=(5, 5), padding='valid')(x)  #input_shape = (12, 12, 6)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)  
    x = Activation('sigmoid')(x) 
    x = Flatten()(x)

    # layer 3
    x = Dense(units=120, activation='relu')(x)

    # layer 4
    x = Dense(units=64, activation='relu')(x)

    # layer 5
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":
    model = LeNet()
    
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    plot_model(model,to_file='model.png',show_shapes=True)

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    train_acc = history.history['acc']
    test_acc = history.history['val_acc']
    x = np.arange(len(train_acc))

    plt.plot(x, train_acc, marker="o",label = 'train accuracy')
    plt.plot(x, test_acc, marker="o", label = 'test accuracy')
    plt.legend()
 
    plt.title('LeNet')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('lenet_result.png')