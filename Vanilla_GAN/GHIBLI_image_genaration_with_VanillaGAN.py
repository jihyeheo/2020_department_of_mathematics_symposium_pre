#!/usr/bin/env python
# coding: utf-8

# # GHIBLI Character generation with GANs
# - Vanilla GAN with Keras

# ## 1) Import Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import imageio

import tensorflow as tf
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


# ## 2) Make Dataset

# In[48]:


def load_and_preprocessing(dir):
    data = []
    img_list = os.listdir(dir)
    for name in img_list :
        
        img = Image.open(dir + name)
        img_resize = img.resize((120,120))   
        
        images = tf.keras.preprocessing.image.img_to_array(img_resize)
        images /= 255.  # preprocessing
        
        data.append(images)
        
    return np.stack(data)


# In[49]:


dir = "images/"
img_list = os.listdir(dir)
img_len = len(os.listdir(dir))

print("The number of images :",img_len)
print(img_list[0:10])


# In[50]:


# Make dataset
dataset = load_and_preprocessing(dir)
print("Shape of dataset :", dataset.shape)


# In[51]:


# Random shuffle

s = np.arange(dataset.shape[0])
np.random.shuffle(s)

x_shuffle = dataset[s,:]

print(x_shuffle.shape)


# ## 3) Explore Dataset

# In[52]:


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     image = np.reshape(x_shuffle[i,:] , [120,120,3])
#     plt.imshow(image)
# plt.show()


# ## 4) Generator

# In[53]:


# params
latent_dim = 100
height = 120
width = 120
channels = 3


# In[54]:


generator_input = layers.Input(shape=(latent_dim,))
g = layers.Dense(128 * 15 * 15)(generator_input)
g = layers.Reshape((15, 15, 128))(g)

g = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(g)
g = layers.BatchNormalization(momentum=0.8)(g)
g = layers.ReLU()(g)

g = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(g)
g = layers.BatchNormalization(momentum=0.8)(g)
g = layers.ReLU()(g)

g = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(g)
g = layers.BatchNormalization(momentum=0.8)(g)
g = layers.ReLU()(g)

g = layers.Conv2D(channels, 3, activation='tanh', padding='same')(g)

generator = Model(generator_input, g)
generator.summary()


# ## 5) Discriminator

# In[55]:


discriminator_input = layers.Input(shape=(height, width, channels))

d = layers.Conv2D(128, 3, strides=2, padding='same')(discriminator_input)
d = layers.LeakyReLU(alpha=0.2)(d)

d = layers.Conv2D(128, 3, strides=2, padding='same')(d)
d = layers.BatchNormalization(momentum=0.8)(d)
d = layers.LeakyReLU(alpha=0.2)(d)

d = layers.Conv2D(64, 3, strides=2, padding='same')(d)
d = layers.BatchNormalization(momentum=0.8)(d)
d = layers.LeakyReLU()(d)

d = layers.Conv2D(64, 3, strides=2, padding='same')(d)
d = layers.BatchNormalization(momentum=0.8)(d)
d = layers.LeakyReLU()(d)

d = layers.Flatten()(d)
d = layers.Dense(1, activation='sigmoid')(d)

discriminator = Model(discriminator_input, d)
discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5, clipvalue=1.0)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
discriminator.summary()


# ## 6) GANs - Training

# In[56]:


gan_input = layers.Input(shape=(latent_dim,))
discriminator.trainable = False

gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan_optimizer = Adam(lr=0.0002, beta_1=0.5, clipvalue=1.0)
gan.compile(optimizer = gan_optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])


# In[57]:


batch_size = 128

def train(epochs, print_step):
    hist = []
    
    for epoch in range(epochs):
        
        real_images = x_shuffle[np.random.randint(0, x_shuffle.shape[0], batch_size)]
        real_label = np.ones((batch_size, 1))
        
        noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
        generated_images = generator.predict(noise)
        fake_label = np.zeros((batch_size, 1))
        
        real_label += 0.05 * np.random.normal(0, 1, size=real_label.shape)
        fake_label += 0.05 * np.random.normal(0, 1, size=fake_label.shape)
        
        # discriminator
        dis_loss_real = discriminator.train_on_batch(real_images, real_label)
        dis_loss_fake = discriminator.train_on_batch(generated_images, fake_label)
        dis_loss = 0.5 * np.add(dis_loss_real, dis_loss_fake)
        
        # Generator
        gene_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        record = (epoch, dis_loss[0], gene_loss[0])
        hist.append(record)

        if epoch % print_step == 0:
            print("%5d iteration - discriminator loss: %.3f, generator loss: %.3f" % record)
            
    return hist


# In[58]:


import warnings
warnings.simplefilter("ignore")


# In[61]:


get_ipython().run_cell_magic('time', '', 'hist_1000 = train(1000, 10)')


# ## 7) Save Models

# In[66]:


generator.save('gan_g_1000')
discriminator.save('gan_d_1000')

