#!/usr/bin/env python
# coding: utf-8

# ### Install Required Libraries

# In[1]:


#!pip install numpy


# In[2]:


#!pip install pandas opencv-python keras tensorflow


# In[3]:


#!pip install pandas opencv-python keras tensorflow


# ### Task 1: Import libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2


# ### Task 2: Import Dataset:

# In[19]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[20]:


x_train.shape


# In[21]:


print(x_test.shape, y_test.shape)


# Let's test any data sample to check if it is corect data or not

# In[22]:


print("label=",y_train[0])
plt.imshow(x_train[0])


# ### Task 2: Data Preprocessing

# In[23]:


print(x_train.max(), x_train.min(), x_test.max(), x_test.min())


# - The training data ranges between 0-255, now we will rescale the feature values to be in the range [0, 1]

# In[24]:


x_train_processed, x_test_processed = x_train / 255.0, x_test / 255.0


# In[25]:


print(x_train_processed.max(), x_train_processed.min(), x_test_processed.max(), x_test_processed.min())


# ### Task 3: Build a Classifier using MLP (Multi Layer perceptron)

# In[13]:


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
from keras.optimizers import Adam


# In[26]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


# In[28]:


model.summary()


# ### Task 4: Compile the Model

# In[32]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Task 5: Train and Test the model.

# In[34]:


history = model.fit(x_train_processed, y_train, epochs=5)
history


# # Evaluation

# In[37]:


test_loss, test_acc = model.evaluate(x_test_processed, y_test)


# In[38]:


print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)


# - Access Loss and Accuracy details from the training history

# In[41]:


training_loss = history.history['loss']
training_accuracy = history.history['accuracy']


# In[76]:


# Create subplots for loss and accuracy
plt.figure(figsize=(8, 3))
# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# - As we can see in the graph, loss has been decreased with each epoch where accuracy has been increased

# ### Let's make predictions on the test test and check whether those predictions are correct or not

# In[53]:


model.predict(x_test_processed)[0].argmax()


# In[54]:


y_test[0]


# In[71]:


predictions = model.predict(x_test_processed)
plt.figure(figsize=(12, 5))
for i in range(9):
    plt.subplot(1, 9, i+1)
    prediction = predictions[i].argmax()
    image =plt.imshow(x_test_processed[i])
    plt.xlabel('prediction: '+str(prediction))
    plt.xticks([])  # Hide the x-axis scale and ticks
    plt.yticks([])  # Hide the y-axis scale and ticks
    

