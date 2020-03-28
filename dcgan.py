from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist

import os
import time

class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.batch_size = 128
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.z_dim = 100
        
        self.iterations = 100000
        self.batch_size = 128
        self.sample_interval = 1000

        self.losses = []
        self.accuracies = []
        self.iteration_checkpoints = []

        self.img_grid_rows = 9
        self.img_grid_cols = 9

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        self.gan = Sequential()
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)

        self.gan.compile(loss='binary_crossentropy', optimizer=Adam())

    def build_generator(self):
        inputs = Input((self.z_dim,))
        x = Dense(256 * 7 * 7, input_dim=self.z_dim)(inputs)
        x = Reshape((7, 7, 256))(x)
        x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same')(x)
        x = Activation('tanh')(x)

        model = Model(inputs=inputs,outputs=x)
        return model

    def build_discriminator(self):
        inputs = Input((self.img_shape))
        x = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same')(inputs)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(128, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model

    def train(self):

        (X_train, _), (_, _) = mnist.load_data()

        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)
        
        #label for real images
        real = np.ones((self.batch_size, 1))

        #label for fake images
        fake = np.zeros((self.batch_size, 1))
        
        time_init = time.time()
        for iteration in range(self.iterations):

            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
            
            gen_imgs = self.generator.predict(z)

            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)  # get mean
            
            # train generator
            z = np.random.normal(0, 1, (self.batch_size, 100))
            gen_imgs = self.generator.predict(z)

            g_loss = self.gan.train_on_batch(z, real)
            
            if (iteration + 1) % self.sample_interval == 0:
                time_passed = time.time()
                elapsed_time = time_passed - time_init

                self.losses.append((d_loss, g_loss))
                self.accuracies.append(100.0 * accuracy)
                self.iteration_checkpoints.append(iteration + 1)

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [elapsed_time : %f]" %
                    (iteration + 1, d_loss, 100.0 * accuracy, g_loss, elapsed_time))
                
                self.save_images(iteration + 1)
                time_init = time.time()


    def save_images(self, iteration):
        
        z = np.random.normal(0, 1, (self.img_grid_rows * self.img_grid_cols, self.z_dim))

        gen_imgs = self.generator.predict(z)


        #rescale
        gen_imgs = 0.5 * gen_imgs + 0.5  
        
        fig, axs = plt.subplots(self.img_grid_rows,
                            self.img_grid_cols,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

        cnt = 0
        for i in range(self.img_grid_rows):
            for j in range(self.img_grid_cols):
                # Output a grid of images
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.suptitle('iteration %d' % iteration)
        fig.savefig("gen_imgs/%d.png" % iteration)
        plt.close()
    
    def visualise_loss(self):
        losses = np.array(self.losses)

        plt.figure(figsize=(20, 5))
        plt.plot(self.iteration_checkpoints, losses.T[0], label="d loss")
        plt.plot(self.iteration_checkpoints, losses.T[1], label="g loss")

        plt.xticks(self.iteration_checkpoints, rotation=90)

        plt.title("Loss")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig('loss.png')


    def visualise_acc(self):
        accuracies = np.array(self.accuracies)

        plt.figure(figsize=(20, 5))
        plt.plot(self.iteration_checkpoints, accuracies, label="d acc")

        plt.xticks(self.iteration_checkpoints, rotation=90)
        plt.yticks(range(0, 100, 5))

        plt.title("Acc")
        plt.xlabel("iter")
        plt.ylabel("acc (%)")
        plt.legend()
        plt.savefig('acc.png')

if __name__ == "__main__":
    
    dcgan = DCGAN()
    if not os.path.exists('gen_imgs'):
        os.mkdir('gen_imgs')
    print('--start--')
    dcgan.train()
    dcgan.visualise_loss()
    dcgan.visualise_acc()

    
