{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "db5a3c6d-c665-4387-bcc5-d181f0768192",
   "metadata": {},
   "source": [
    "### Install Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe3fa7ed-42fe-4e4f-b2d2-c8497cfa2f83",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "dab8e81d-6e20-49f4-96b3-8f42c6dac7f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install pandas opencv-python keras tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3bcc2bd4-832d-47b3-a49c-5ddfb8aaa662",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install pandas opencv-python keras tensorflow"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53a67c42-dd34-4928-a876-b1f3e507fd98",
   "metadata": {},
   "source": [
    "### Task 1: Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ec155fe2-245d-433c-85c5-70f2a2eeb936",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from keras.models import load_model\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38dfd552-fef4-4729-9ea7-ecd9585dfe83",
   "metadata": {},
   "source": [
    "### Task 2: Import Dataset:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0ef37fb0-30d1-4469-96d9-6472993f5709",
   "metadata": {},
   "outputs": [],
   "source": [
    "(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e24a5a28-eaab-4bc1-8959-07097d9e46d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "30796991-87e3-495f-89c2-534574a07549",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(10000, 28, 28) (10000,)\n"
     ]
    }
   ],
   "source": [
    "print(x_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd4202f8-8375-4a7a-af73-fce95e3c5398",
   "metadata": {},
   "source": [
    "Let's test any data sample to check if it is corect data or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "0bb20727-9dbd-4603-9b57-747bd38eaa3c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "label= 5\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x26c25b723b0>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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wAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJrgZKfAVgWsme8/85V/8b9z5qxkvec/MuuC498y51OW6vWfqPy/wP1Bvi/8MBiSugAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAE9yMFAPeiILx3jP778yL61hPLH7Ne2bRRZ/FdayB7JG2a7xnan9xnffMt16q857B0MEVEADABAECAJjwClBlZaWuvfZapaWlKSsrSwsWLFBDQ0PMPseOHVN5ebkuvvhiXXTRRVq0aJHa2toSumgAwODnFaDa2lqVl5ervr5e7777rrq7uzV37lx1dnZG97n//vv11ltvaePGjaqtrdXBgwd18803J3zhAIDBzetDCFu3bo35uqqqSllZWdq1a5dmzZqlcDisX//619qwYYO+973vSZLWr1+vK6+8UvX19bruOv83KQEAQ9M3eg8oHA5LkjIyMiRJu3btUnd3t0pKSqL7TJo0SePGjVNdXd+fdunq6lIkEonZAABDX9wB6u3t1cqVKzVjxgxNnjxZktTa2qrU1FSNHj06Zt/s7Gy1trb2+fdUVlYqFApFt/z8/HiXBAAYROIOUHl5ufbu3avXXvP/uYmvqqioUDgcjm7Nzc3f6O8DAAwOcf0g6ooVK/T2229r+/btGjt2bPTxnJwcHT9+XO3t7TFXQW1tbcrJyenz7woGgwoGg/EsAwAwiHldATnntGLFCm3atEnbtm1TQUFBzPPTpk1TSkqKqquro481NDTowIEDKi4uTsyKAQBDgtcVUHl5uTZs2KAtW7YoLS0t+r5OKBTSyJEjFQqFdNddd2nVqlXKyMhQenq67rvvPhUXF/MJOABADK8APf/885Kk2bNnxzy+fv16LV26VJL085//XMOGDdOiRYvU1dWl0tJS/fKXv0zIYgEAQ0fAOeesF/FVkUhEoVBIszVfIwIp1svBGYy4ZJz3THharvfM4h9vPftOp7hn9F+9Zwa6B1r8v4tQ90v/m4pKUkbV7/2HenviOhaGnhOuWzXaonA4rPT09H73415wAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMBHXb0TFwDUit+/fPHsmn794YVzHWl5Q6z1z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6yUlVTgcliRlZGRIknbt2qXu7u6Y18OkSZM0bty4If16OPU8fOmVV15RZmamJk+erIqKCh09etRief0acDcjPdVnn32mnp4eZWdnxzyenZ2tP//5z0arslFUVKSqqipdccUVamlp0Zo1a3T99ddr7969SktLs16eidbWVknq8/Xx5XPni3nz5unmm29WQUGB9u/fr0ceeURlZWWqq6vT8OHDrZeXcL29vVq5cqVmzJihyZMnSzr5ekhNTdXo0aNj9h3Kr4e+zoMk3X777Ro/frzy8vK0Z88ePfzww2poaNCbb75puNpYAz5A+IeysrLon6dOnaqioiKNHz9eb7zxhu666y7DlWEguPXWW6N/njJliqZOnaqJEyeqpqZGc+bMMVxZcpSXl2vv3r3nxfugZ9Lfebj77rujf54yZYpyc3M1Z84c7d+/XxMnTjzXy+zTgP8WXGZmpoYPH37ap1ja2tqUk5NjtKqBYfTo0br88svV2NhovRQzX74GeH2cbsKECcrMzBySr48VK1bo7bff1vvvvx/z61tycnJ0/Phxtbe3x+w/VF8P/Z2HvhQVFUnSgHo9DPgApaamatq0aaquro4+1tvbq+rqahUXFxuuzN6RI0e0f/9+5ebmWi/FTEFBgXJycmJeH5FIRDt27DjvXx+ffvqpDh8+PKReH845rVixQps2bdK2bdtUUFAQ8/y0adOUkpIS83poaGjQgQMHhtTr4WznoS+7d++WpIH1erD+FMTX8dprr7lgMOiqqqrcH//4R3f33Xe70aNHu9bWVuulnVMPPPCAq6mpcU1NTe53v/udKykpcZmZme7QoUPWS0uqjo4O9/HHH7uPP/7YSXJPP/20+/jjj93f/vY355xzP/3pT93o0aPdli1b3J49e9z8+fNdQUGB++KLL4xXnlhnOg8dHR3uwQcfdHV1da6pqcm999577rvf/a677LLL3LFjx6yXnjDLly93oVDI1dTUuJaWluh29OjR6D733HOPGzdunNu2bZvbuXOnKy4udsXFxYarTryznYfGxkb34x//2O3cudM1NTW5LVu2uAkTJrhZs2YZrzzWoAiQc84999xzbty4cS41NdVNnz7d1dfXWy/pnFu8eLHLzc11qamp7tvf/rZbvHixa2xstF5W0r3//vtO0mnbkiVLnHMnP4r92GOPuezsbBcMBt2cOXNcQ0OD7aKT4Ezn4ejRo27u3LluzJgxLiUlxY0fP94tW7ZsyP2ftL7++SW59evXR/f54osv3L333uu+9a1vuVGjRrmFCxe6lpYWu0UnwdnOw4EDB9ysWbNcRkaGCwaD7tJLL3U/+tGPXDgctl34Kfh1DAAAEwP+PSAAwNBEgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJj4f4W4/AnknuSPAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"label=\",y_train[0])\n",
    "plt.imshow(x_train[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a457d5e7-1267-438f-958e-5376a6361ba4",
   "metadata": {},
   "source": [
    "### Task 2: Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4f11aa1e-f87f-48b9-84b0-52d2564cb9a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "255 0 255 0\n"
     ]
    }
   ],
   "source": [
    "print(x_train.max(), x_train.min(), x_test.max(), x_test.min())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84f57f77-419c-459c-8b4f-7a4eddfaa0ff",
   "metadata": {},
   "source": [
    "- The training data ranges between 0-255, now we will rescale the feature values to be in the range [0, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0cf9a6a3-12af-46f2-98a4-6d8cad0ca016",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_processed, x_test_processed = x_train / 255.0, x_test / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "7cc37190-33b0-439b-bb76-3d875ea7837a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0 0.0 1.0 0.0\n"
     ]
    }
   ],
   "source": [
    "print(x_train_processed.max(), x_train_processed.min(), x_test_processed.max(), x_test_processed.min())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f232a8de-64c7-484d-890e-d4a50a34d994",
   "metadata": {},
   "source": [
    "### Task 3: Build a Classifier using MLP (Multi Layer perceptron)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f2fef9ea-1927-4d6b-ab84-aa617028da10",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Convolution2D, MaxPooling2D\n",
    "from keras.layers import Flatten, Dense,Dropout\n",
    "from keras.optimizers import Adam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "a9c828a9-c99f-4b51-9f8f-6fd19710d5c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.models.Sequential([\n",
    "    keras.layers.Flatten(input_shape=(28, 28)),\n",
    "    keras.layers.Dense(128, activation='relu'),\n",
    "    keras.layers.Dropout(0.2),\n",
    "    keras.layers.Dense(10, activation='softmax')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "85d3dd84-2dac-417f-9883-2b9fc78186ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " flatten_1 (Flatten)         (None, 784)               0         \n",
      "                                                                 \n",
      " dense_2 (Dense)             (None, 128)               100480    \n",
      "                                                                 \n",
      " dropout_1 (Dropout)         (None, 128)               0         \n",
      "                                                                 \n",
      " dense_3 (Dense)             (None, 10)                1290      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 101770 (397.54 KB)\n",
      "Trainable params: 101770 (397.54 KB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29788d2f-f656-4c79-9ac9-7b02c49347a2",
   "metadata": {},
   "source": [
    "### Task 4: Compile the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "751a1d1f-c857-42bc-81d5-1ecaf7a69ed5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',\n",
    "              loss='sparse_categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20cff72c-7f8f-4c8c-810b-95fd69a99cc8",
   "metadata": {},
   "source": [
    "### Task 5: Train and Test the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "08f3a757-4a6e-4fc6-a863-4c79eff7fbd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "1875/1875 [==============================] - 13s 7ms/step - loss: 0.0660 - accuracy: 0.9792\n",
      "Epoch 2/5\n",
      "1875/1875 [==============================] - 14s 7ms/step - loss: 0.0576 - accuracy: 0.9817\n",
      "Epoch 3/5\n",
      "1875/1875 [==============================] - 12s 6ms/step - loss: 0.0539 - accuracy: 0.9820\n",
      "Epoch 4/5\n",
      "1875/1875 [==============================] - 11s 6ms/step - loss: 0.0475 - accuracy: 0.9847\n",
      "Epoch 5/5\n",
      "1875/1875 [==============================] - 13s 7ms/step - loss: 0.0443 - accuracy: 0.9852\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x26c21565d20>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history = model.fit(x_train_processed, y_train, epochs=5)\n",
    "history"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02613952-90f8-4a25-aaac-b399dc5b72e2",
   "metadata": {},
   "source": [
    "# Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "5b00677d-e235-4cfc-8a21-a36b78f42fe1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 2s 5ms/step - loss: 0.0767 - accuracy: 0.9780\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_acc = model.evaluate(x_test_processed, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "ddce35a6-e56c-4af6-bc25-de245c3e822b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Loss:  0.07668498158454895\n",
      "Test Accuracy:  0.9779999852180481\n"
     ]
    }
   ],
   "source": [
    "print(\"Test Loss: \", test_loss)\n",
    "print(\"Test Accuracy: \", test_acc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f3a23a4-1d87-41ac-8cbb-c7a0a067e143",
   "metadata": {},
   "source": [
    "- Access Loss and Accuracy details from the training history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "64d47a33-6ea3-4688-b665-66ff193a64b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "training_loss = history.history['loss']\n",
    "training_accuracy = history.history['accuracy']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "c8ddf7b8-1dd0-4f3a-8dd0-08d9ac4b82fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x300 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create subplots for loss and accuracy\n",
    "plt.figure(figsize=(8, 3))\n",
    "# Loss subplot\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(training_loss, label='Training Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "\n",
    "# Accuracy subplot\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(training_accuracy, label='Training Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7ab7133-fa60-423b-a4eb-2d7f6d02c5e3",
   "metadata": {},
   "source": [
    "- As we can see in the graph, loss has been decreased with each epoch where accuracy has been increased"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d13a611-d600-429f-a026-67fbaab9675f",
   "metadata": {},
   "source": [
    "### Let's make predictions on the test test and check whether those predictions are correct or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "254f7210-8b21-4bee-ab53-6b89c0db4d92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 2s 7ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(x_test_processed)[0].argmax()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "90d6a731-5b1a-45cb-9ed2-1c500a89272a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "d7a64fdf-12c4-4e6a-9de0-45d128669b7b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 1s 4ms/step\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA7UAAAB/CAYAAAAwwuV6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAos0lEQVR4nO3deXxMZ/s/8GuySyQkEbKQhBKJLSKopYpSe0nt2gbPQ/FY+60Hbb9KF/VQXRRtVWv5PqWqaq/uSptaopYITSSxbwliTYgsM/fvDz/3dY7MJDPJzJiTfN6vV169zsx9zrk7l3Nm7pnr3EcnhBAEAAAAAAAAoEFOj7oDAAAAAAAAAGWFQS0AAAAAAABoFga1AAAAAAAAoFkY1AIAAAAAAIBmYVALAAAAAAAAmoVBLQAAAAAAAGgWBrUAAAAAAACgWRjUAgAAAAAAgGa5mNPIYDDQpUuXyNvbm3Q6na37VOkIISgnJ4eCg4PJyals3zMgR7ZljRwRIU+2hBxpA853jg85cnw43zk+5EgbcL5zfGbnSJjh/PnzgojwZ+O/8+fPm5MO5EijOUKekCP8WSdPyBFyhL/y5wh5Qo7wZ508IUeOkSOzfqn19vYmIqInqBe5kKs5q4AFiqiQ/qTv5etcFsiRbVkjR0TIky0hR9qA853jQ44cH853jg850gac7xyfuTkya1D74Kd0F3IlFx2SZXXi/n/KU7KAHNmYFXKkXB95sgHkSBtwvnN8yJHjw/nO8SFH2oDzneMzM0eYKAoAAAAAAAA0C4NaAAAAAAAA0CwMagEAAAAAAECzMKgFAAAAAAAAzcKgFgAAAAAAADTLrNmPAUpzZk5b1bLeQ8g4oPFVGe+N3mB0/cd++4eMvfdXkXGtRXus1UUAAAAAAKiA8EstAAAAAAAAaBYGtQAAAAAAAKBZGNQCAAAAAACAZuGaWiizG9sbyPhY8yVmrVMojD9+vPMXMl7TMkjG3/zSUdVOn5phQQ/BlnSxjWW8feuXqueaLp0o4zpv47poa3GuXk3GaUvqyVh5/My8Eivjo89HyFifkm7j3gEAAEBF4RJYS7Vc0CC41HVc0y/KOO1V/pxSPUUnY7/Ue6p1nBIOl7WL6u1YZSsAAAAAAAAAjwAGtQAAAAAAAKBZKD8GiyhLjnc3/9qsdZbe5PKDD/Y+LePwML7Vz8+NNsr4ee9MGb8zsoZqW/VmoPzYUVxp5SPjItKrnvO8ZKLOHMrFULe2jI92+kzGyrL+OTUPyjj62XYyroPyY6vRd26hWp647BsZf9qgvk32mTOkjYyrJ2VzX9JO2GR/wG4O51vWJc77VPVco4/Hyzh0/n4Zi6Ii23dMo1zC6si45rqbqud+P9hIxpGf8HP6v9Ns3S1yDgiQ8bWefBz7rjskY5Gfb/N+ANjbrRf4/eVaLy4NfiXmR1W74T7fl7qt5bdCZdzfe5OMfQd5mFynT0isyecsgV9qAQAAAAAAQLMwqAUAAAAAAADNQvkxlKqoC5cF/Bb9seIZVxktvBFBSjuHtOSFS1dkGHHjgIydPLgUYW5iUxm/VuMo79sXJVyO6kYzLjm+UKQuyfJfvtfe3amwXOpwyXHdZSg1dQRnu7urlv2cc22+z6zeBTIujOfvo/362HzXlZJLCM/y+fasL0y2S5nwiYx7LuogY5GTY5uOaZRyFtW3dm2QcUNXg6rdU9cCZaz/2/aXGylLjp//k8uM23hw2eSEo2N5hcN/27xPjs65hr+M0z7kUtNODThfFzsWyhgl24+WU3SUjI9P8pJxQreFMg5w/ovbl/P3zlHVzimWTJcc2wJ+qQUAAAAAAADNwqAWAAAAAAAANOuRlB9fe5FnEgyN53K641e4PKUgn0tbQ9ZyTETkeYFLvQxJKbboIijkhrjJWFmWoCw53tW3qWod/anSZyo88WaMjL/ye1/xDJf21f4R37s4EtG+uYwT+nwg445/TFK1q0/WuZF2ZXRuVjvVcmwPPse9G5Rg0baqtuMZxs+/rt5ujWQu7a+yZT9ByXSufB586qkku+/f+zCXcQ0e9buMd1bn8nT9zVt27VNFdqV7mIy7eRaabNfiwBAZB+RihnEll9ohMq627q6Mm7k5y7jhr+NU6zQYcYjsKXVOuIwHV+WZXlssnC7j4MN77Nklh3Nlovq9Y/aU/8q4t+fPRteJq/GMjIsuXrJNx8Asd+p6yzi9p3L29ipW24fyLidrzrayeP1qZJ1LqzBiAAAAAAAAAM3CoBYAAAAAAAA065GUH0+f9pWMB3jd4CceM7FCJ/XimSIuY/noamfrdcyE/Ve4DMnr/Woydtlx0Ob7dgTV/8sz2Q488IKMdTduy7go84zF2x3d61cZV3VyL6ElOIrrjbhcJcjZU8Yh37oaaw5lkDx2sWq5UOhNtCzdrug1vBCtfm7TnSAZr8iJk7HLb5XjvGapnGdbyHhRiDpHUZsnyrgBJdpk//m+QsaTfY/LeJc3z2xJKD8uFydPPqd1n/ynWeu4f+3LC0KYblgJ3WhfR8abwz822iZq5hXVsj3udyDa8snwRJ/PZNzx6CAZ11nBx1jZz8Da5RzBH8i/mLpQ9VxzNx46qOeuZpmfcslr0Fie0booM8sq/auslCX9qTP40pNae3Sqdj5r98nYKZ/PS+mFPIv++aLqMq7jclPGI4+NkPGNVJ7pmoio1l+8rep7zstY5PJlodVuPrq7NOCXWgAAAAAAANAsDGoBAAAAAABAsx5J+fGi14bKeFYzHlf7pvLP2jei+Kd0t2Y3Veu/22SjjD8M4lKv7Xeryri3Zy6VJk8UqJYT8/mmxJ08FLMdKvZRfwjfhDtiR6m7qHD0KeWb3fHMOzzz9ajq7yme4Zk9p2a2kbH3r6nq/Zdr71BeXcZzKfrmO9VlXHWXerZr5Mkyrru4FNhV51xCy9IdLuCCsDOFATJ+1uu6qt3gqlz2N/jLZTLuExJbrv1XJMrZvj+e/5GMV98OU7WLnMnnRVv922/b7ZiNtgwP5LfjUu45NZcbbXPXoP7c4PPVPqPtKiuXMC45vtrvntE2Ld/j2fIDz9tnZmFlyfHMNf9ntE3udi6T9bp2yuZ9cmSpr3BZvXK2anMlxvJlhul7+Zjp/+XLMq73jvouCYZ7xv+9VHbO1fmyx9bbT8t4c42tMm5/YCKZ4v7DXzKe1nukjPV/8+c256gGMvZLO8mxwfRnfntcKmAp/FILAAAAAAAAmoVBLQAAAAAAAGjWIyk/9vo2UREbb+NTwvqLAzvJeE77cF7nd55x691O9Uvth0ueet42r+RMGfv/sUHGTd14ZlfPM5jl1VI347nkePdwLjmu5sQlx3vzubwlaU6MjKvc3m/j3kFpnBs3lPHcmmtlvPw2z7ynx6yrFsuLay3jfwStl/HDsx2bM/txkx3jZBywg2cSd7/F677aSf0d5tFBi4xu68Kr7WRc+z/2KQ10VDde5Zn2a7twsdXLk3qr2rnesM2M0S5BXA65MvRHGRcKfB9tC6f7l15mOTAj7qFHLtmkL1p1/iO+DCyj9SoZz7zSXMYhK/+Wsb0uVbnYiS8va+/On/2a7OGZXkMXV+7znXOjCBn/2mWh4pkqqnbzr3GZ/oGboTJe99iPZEyEq5uMP3/+U97Oin6qdobTZy3pboXm5MGfj/O/5fLj12r8JuOGG8fLOHITH1NEpo8rZcmx6vHUjDL00vHgnREAAAAAAAA0C4NaAAAAAAAA0CwMagEAAAAAAECzHsk1teVVlHVZxl4bOFbWkHt9e83i7V4ezdd+Nnbjl+a963xNYfhKnubdEaezdkTZLfhWTcrraJVG7Bot44jNuI7WkVx82t/o4wdzlLc1ybNPZzROeX3ynA/4Njot3ZS3CTF9Xd+mO3zrn5k7B8g4avpxGetv3za6bsOMCNXy/r58LLZ251sp/PCvd2XczWO6jMPn8nWjIj/fZB+17tqL/D6wvukCGf/3VjMZu/5qm2toH5byFt8eRXlt9YgzXWWsv3LVLn2pDHq3OmL08VsGPr8VvlFL9ZwTrqlVEYJvx6j8N5t4LVzGznlXyBacvL1lnPZOI9Vzm/t+IGMD8dwooYOO2qQvWpTdmt/rw108ZTzm/JOqdhfa8C0znbx43oHYcXyrpn+/+I2Mn/fmfD+p+Ai4bcM51XZTevMcAkWZWZZ0vUJw9uXbKB1/m9+v06I+kfFBxVtv5Fs8HjH1vl/Z4JdaAAAAAAAA0CwMagEAAAAAAECzNFl+bC0uYXVUy0teWyJjVx2XAK7/iEu9/DP32r5jFUDBL1yaujfyfcUzXHsSvZen0o+aelLG9priH8xzu1Gh0ceTljSXcXXCcWEOg+KyBnXJsWn/PNtDxjlD+NYKERe4TN+cY0afkq5aHr+KbwN0YOxCGQc58z4OjeLHB2zk41UcSTVjj9rkFJct42AXvj3S8q84D7XJNrf+UJanExGt7vKZjPMFH4fnPuDSNK/8RIKyy+/VSsZLQj432uaC4lojp98P27pLFdL3kZtlPGpXZxmfywlStStYHkiWyOrAlzf1ejxJxluDP3moJZcct08aKmNfqhi3MrEGPZ/uyED8uiZ/1lTVzk/xfm+4c0fGQe/zefGbZ/i4Gub9Ha8s+HZKl/O5XJyISNyruJe1mOPSC3yrpLRnF8t46x0uS17e52kZ66/y52a4D7/UAgAAAAAAgGZhUAsAAAAAAACaVanLj4//T4hquZU7z9r3dwHPduiXcpegZC71wlXLb9dfL2NfxYzHypnbwt7mokn9jRs26xtYLr8nlw5t6cZlMG9lx8rYb0OyjLmgCMrrtcstVcu3R/OMlPoL1iuVC9/AZbavx7WR8bzAv6y2D61wDgiQ8cyI7Ubb1J5rm5JjpePjq6uWW7rzOfLjGzybq9cGlBxby+VWrqW2eea7l2TcgPDal6TmYr58Yecyfu/vXIVnWF8eulPGTsSfu4iIDB8IsoRyfWXJ7MPW5vCs1f6v8UdfvHcx7wGZRh+/1f2OatlvZenbmhW2VbFk/PezhMORquWIG5X7zhc5jxu/i8RHp7vIuEo6So5Lgl9qAQAAAAAAQLMwqAUAAAAAAADNqnTlx/m9uazy0MAPH3qWp37715QpMq6yp3KXRJjjsW8uqpZj3Ix/XzJsB8+4GnGk8pU5asWFp/jU0MyNS8hGnOFZEGveOW7XPlU0yhnWlZJbPFxCZ6PZOXVctufixEV4pvp16U2OA+Ns06VHRefJ/8a7e96Sceu/hss4kGw/43ON8Osmn1tzmsvSa1C6yXZgGbcY45e+pBbwZUeRi7hUH7Pzl8zlt4My/uiJp2T8drtwGV/oxue4E88sVa2/P5/PSy/8PI5K0+C/fE3T9vUrTLZ7N6W7jEOO/F3qdiujnA2Kmagbcziykbrk/o9WrWV8NaaqjEUfPn81ceXPzamFPHN7Y1c3GW/qyZc2ERHNaPMiL+xLpspmbftliiX+DP1to9UybvvBVBnX3cp3UHDedciWXdMM/FILAAAAAAAAmoVBLQAAAAAAAGhWpSs/PteTx/FVde6q54ad5psae/54RMaWzcVXedwY0VbGb9Z6/6Fn+bUdcaarjKOmn5AxyrgcV0CTKzLWK26W7rLF11hzMFPavzxlXCge7RFwpj/PqvxtAJeKFQpnRcx9DJ7N61a0GUMN12/K+O2rLWT83GMHZPxH0GMyLsrMstq+XcLqyHh3868fepbfr/L21VA8jvLj8rjXh8snD7T6VPEM/9tPK6wpYz1mHC2ToqzLMvbcyHHERm7Ta1wLMiWCSr/0y6kZz6CrnAl5TnYTVbuwKXxZQVGpW62cAreelnH6q1zaOs0/RdVuxma+FMPUjNNDTvaWcd5knl3+2bW7ZPwPn/OqdU5O5vPdY/vM7HQF0tqdZ2JXvvcq7yByfMjH3GYwt2myQ12qX+0vXie3NufI5xS3qZGsntX6gexmXjKuteuK6jlHPxfil1oAAAAAAADQLAxqAQAAAAAAQLMqRfmxk7e3jOM7/Cnj24Z7qnZX5taTsXs+ZuY1xiUkWMYdJvOMeFWd3I01JyKivSn1ZRxxA6+ro3KpGybj9xqul/Hnt7g80m/FXrv2qaKZ2WGbXffnUqe2jHNig1XPLf3HJ6Wuvz+fS5h0BRW3aM+QkyPjny9yOWNC869knPldNX78M770wlw3G3EJWNVwLoVsE3yG+1FCYbcO18FYTV4NLjM2Ndv39IP9ZVyXKt9MrFpxbjbnT1kK+/M7T6raVT1fCetZLaS8rGLMtJdkvPK9D1TtIly5PJUUlyfV/5lnL46cyHdHMNzh8uV5vz0j41FxytJ/ovktuS79i2guXzYcsf3M846g7jZ+/dL7LC2h5X3Kc1da18/VT3Ylq9j/ik61/FLKUBn79XG8y2DwSy0AAAAAAABoFga1AAAAAAAAoFmVovw44w2+i/R3Nbjkrl/GAFU79+9RGlua1Ne4FHVzoOlSys5HB8kYMx5rQ8ZYLk9to6gmf/FQZxnXoWP27BKUU8qbgTL+u9sSs9bZkMuz7H76bz6OPVJLn4m0IvB9k0uuO74xTMabmqyS8fzZlpfhH8jnUjG94vvklm4FilbqUi+l0MVHZVzRZp+2t/y4m0YfTy24K+PaX7gabQOPXvYYLv9PbsOzwZ4pypNxlasFBGVXdT1fXvYPeln13PXBfJzcu8UfFqKm8cy4+jvGZ9Zt+AqXIndp0F/13C+NN8h49mw+R4aom1VYDScclnH39WNkPHwJf9b2dMqXcR/PqzI2dRlFebV2V1/38mfMGhk3XjBZxo9Nc4xL0/BLLQAAAAAAAGgWBrUAAAAAAACgWRjUAgAAAAAAgGZV2Gtqb73QRsbJQxbJ+GRRoYxz59dWreNOmbbvmMYd7PuhYsn0bXyqjeervopu3LBhj8BaDHXuGX0876aH0cfBMbnuCpLxf4I2lNDSuFUX28nYY1vluI5WZT9fu1qtFz8c34mvH7rZwPS5zxT/z41fc3RxI8/5cPDxVSbXV952CCznHPGYjA+0Wq18RkY/5DaRseuvB+3RLSiDu0/nGn18YNJoGdfceche3anwlNfX3l823s6cOVOU57Hbm5qon+RTIc1vxu9dnwR1krHytkMVjSji2+Ypzz9rI4ONNadFA/n2OnpX9XwM7f7N793zAq03X5CT4rfQ2tGON2bCL7UAAAAAAACgWRjUAgAAAAAAgGZVqPJjlxD+if6l19fJ2F3H/5tDj8TLOOAH3MLHVgprVZOxa0GIRevqr2bLWOTnq57TuXPZn3NADTJGH1BdxhlT3czap9Bz6UbkJMUtiG7fNmv9iuCTx1cbfTzkB9tMFV8ZOeu4LN/UFPy3n2tj9HEiojffWi7jzlWMl4srt1solAVh5uVRPHXRrHaVjfMuLmf032W97ead8eaFx023E+2by1i3O8l6HagkLneuKWNTx96SnU/LuAElGm0Dj95nsV/KOFPPt5fxX+j5KLoDZRTwmfrylsd7PifjxNivZDzl3+EyfmxqxS0/tpTXt6bPUdui+bZX8+J5rHNX8K2uYv/4l4zDvuBzYvZkPqbUl2o4PvxSCwAAAAAAAJqFQS0AAAAAAABolubLj3Uu/L8Q/d0FGQ+qek3Ga3K47KjW6zyO50JAsLbt364o87rtDg+TcfZlH9VzvgE8c56yPMWaGs2cKON6043PWFpR3HumtYyf8FCWAmn+1OCQ5q0bKOPBoxYabfPHgo9Vy+oSYuXjpe/P1LoPa7JjnIwbEGYNtSvFpJVOJXzPjJLj8rnnpzP6+MF8LseLms+fIYqMNYZH5sKrPCt7e3c+R+3L55JjZ8x4rC0G9fuT//ucy+wv82ScOpTfE5/5ariMxcG/bdg5bQv9SXHpHl91SZ46viQvtSNfzhQfxpdefB/+k2JLpt+TzmX5ybgBnSlTP60Nv9QCAAAAAACAZmFQCwAAAAAAAJql/RrD6IYyfLvml0abfDx3kIyrH6nY5aS21i/leRnvaPKtTfaxJ2atxesoZ3QrFMYLy3slj5TxrSTjMycTEYX8WXkKz8715RpW5Szhb2U3lXHVLXwTcDMqXqEE9dbxzN77X/CQcWt34zMZl9f+fN7HsqyOqudujA+UceRpxYzfNukJmKQ4qAy4KMZmapqY1Xvr7RgZK2feB8fy/LAdMjYoDppRB0bKOIyOytjZn0sjiYiopr8M9akZ1u8glJvT74dl3On/psk45Z9cfpzzDpcl+wzimeMNOXxpGhC5HuB/420O8SV9+1oY/3z9ZfgviiX+vTNfFKra9UkZKuPIySdl7CifG/BLLQAAAAAAAGgWBrUAAAAAAACgWZosP3ZuFCHjMV9vMdqm0YoJMg7/cp/N+1RZVOl+WsaN5/IswcLMf0nekddlbM7sxY0T/qFaFue8jLar920uL+w/arSNL2UYjSsTZx/1bNIz2n9vtN1XPzwp43pFKNm3Fn1KuoxnvTxaxuef4bLT9J6fWW1/41fwrMZ13tnz0LM3rLYfKDuDh+mS46v6fJPPQcl07u6q5X7BR4y2u1ZQVcYiH6+31hj0/NvMlYk8Q3Lv0QmqdptPBck4pL/t+wXlU3/ZeRl/OYgvlfmjKV/21iP6nzJ2+jPJLv3SCmU5duAkXxk/s6KvjF8L3y7jtu5cQLwhly/P+9/vh6i2W/9/eDzlKCXHSvilFgAAAAAAADQLg1oAAAAAAADQLE2WHx8fr/gp3fO20Ta1d/FsuCQwZ6st1H2tfGWpfSi29H1Qcrn2AWqGh8rrUu4Gy7jrxZYybjCXb2ruiCUmFUGVLftlHKG4iuLJYRNU7VxHXpbxj43XybjbMZ6F0LCqpoyFjtcNT7oqY+TRMa3usVTGqQXqUuRhq6bLOJQeLh+HEunV/+KXpT4h45fanZHxrvP1ZRxCfxNoS+qTK2VseJI/6zX+45+qdvXfuCNjnAsdX9H5CzL+5lmeuT/+V34PzJ7Gdw2o+ad9+qVFRWfO8cJTHE6ePF7GOa14VunImTwLfP2z2rp8E7/UAgAAAAAAgGZhUAsAAAAAAACapYny43vPtFYt73jmfcWSp307A6BhD8/umcYVx+RGZ2WM8qxHx2ftQ+U+inulP0t8LvSiU4pGypghj47vrdM8G+WdT0JUz4VuQMlxWYmiItVy+Ctcfhr1n3gZ65K87dYnKLuf/pdLUFNe5ZmM9yZGyjjyo0syfiwrTbW+/t49Am3Sp/LdKoac6ibjbTFfyHhUGy6lpX24bM0ctRbx+0stxeNFxZtqBn6pBQAAAAAAAM3CoBYAAAAAAAA0C4NaAAAAAAAA0CxNXFN7qb2zajnUxfh1tGty+LYWrrf5lj64oQ8AADikLnzrCi+6UEJDKA/9idMyDh30CDsCZeKxjW+BdnUbP16feA4CLV8LCOa5+yx/ok/cw7ckvNHQS8a+2roLDVgRfqkFAAAAAAAAzcKgFgAAAAAAADRLE+XHJfnPtUYy3ts9XMYi8+gj6A0AAAAAAFibPvuajJdF1JOxL+19FN0BB4NfagEAAAAAAECzMKgFAAAAAAAAzdJE+XG9V9RlBb1eaWGiZZbtOwMAAAAAAAAOA7/UAgAAAAAAgGaZ9UutEPfvC1VEhbjpqw0UUSER8etcFsiRbVkjR8r1kSfrQ460Aec7x4ccOT6c7xwfcqQNON85PnNzZNagNicnh4iI/qTvy9ktKElOTg5Vq1atzOsSIUe2Vp4cPVifCHmyJeRIG3C+c3zIkePD+c7xIUfagPOd4ystRzphxlcTBoOBLl26RN7e3qTT6azaQbj/zUNOTg4FBweTk1PZKsKRI9uyRo6IkCdbQo60Aec7x4ccOT6c7xwfcqQNON85PnNzZNagFgAAAAAAAMARYaIoAAAAAAAA0CwMagEAAAAAAECzMKgFAAAAAAAAzXLoQW14eDgtXLhQLut0Otq8eXO5tmmNbQBDjrQBeXJ8yJHjQ44cH3KkDciT40OOHB9ypObQg9qHZWZmUs+ePc1q+8Ybb1Dz5s3LtQ1r0ul0Rv8WLFhg977YklZzVFhYSDNmzKCmTZuSl5cXBQcH0/Dhw+nSpUt27Ye9aDVPREQbN26kbt26kb+/P+l0OkpKSrJ7H+xByzkSQtCsWbMoKCiIqlSpQl27dqWMjAy798PWtJwjpXHjxpFOp1N9OKootJyjynKuI9J2npRwLN3naDm6fPkyjRw5koKDg8nT05N69OiB9yQHy9HIkSOLjZF69Ohh0TZsPqgtKCiw2rYCAwPJ3d39kW+jLDIzM1V/K1asIJ1ORwMGDLB7Xx6GHBHdvXuXDh06RK+//jodOnSINm7cSGlpadS3b1+79qMkyNN9d+7coSeeeILmz59v932XBjm6791336VFixbR0qVLKTExkby8vKh79+507949u/flYciR2qZNm2jfvn0UHBz8yPrwMOToPkc+1xEhTw/DsWSfbVhKCEFxcXF06tQp2rJlCx0+fJjCwsKoa9eudOfOHbv2xRjkiPXo0UM1Vlq7dq1lGxAW6Nixo5gwYYKYMGGC8PHxEf7+/mLmzJnCYDDINmFhYeKtt94S8fHxwtvbW4wYMUIIIURCQoJ44oknhIeHh6hdu7aYNGmSyM3NletdvnxZ9OnTR3h4eIjw8HCxevVqERYWJj788EPZhojEpk2b5PL58+fF0KFDha+vr/D09BSxsbFi3759YuXKlYKIVH8rV640uo3k5GTRuXNn4eHhIfz8/MSLL74ocnJy5PMjRowQ/fr1EwsWLBCBgYHCz89PjB8/XhQUFFjy0hXTr18/8dRTT5VrG8YgR9bL0f79+wURibNnz5ZrO8YgT+XP0+nTpwURicOHD5dp/dIgR2XLkcFgEIGBgWLBggXysZs3bwp3d3exdu1as7djDuSofMfRhQsXREhIiDh27Fix/zdrQY4c/1wnBPKEY6ni5igtLU0QkTh27Jh8TK/Xi4CAAPH555+bvR1zIEdlP44ebKc8LB7UVq1aVUyZMkUcP35crF69Wnh6eoply5bJNmFhYcLHx0e899574sSJE/LPy8tLfPjhhyI9PV3s3r1bxMTEiJEjR8r1evbsKaKjo8XevXvFgQMHRLt27USVKlVMJisnJ0fUq1dPdOjQQSQkJIiMjAyxbt06sWfPHnH37l0xdepU0bhxY5GZmSkyMzPF3bt3i20jNzdXBAUFif79+4ujR4+KHTt2iLp168p/YELcf5F9fHzEuHHjRGpqqti2bVux/+fZs2eLsLAws1/HrKws4eLiItasWWP+i28m5Mg6ORJCiF9++UXodDpx69Yti9YzB/JU/jzZY1CLHFmeo5MnTxrNy5NPPikmT55sWRJKgRyV/TjS6/Wic+fOYuHChfJ1stUHceTIsc91QiBPOJYqbo6Sk5MFEYkTJ06oHq9du7ZqX9aAHJX9OBoxYoSoVq2aCAgIEBEREWLcuHEiOzvbotff4kFtVFSU6huHGTNmiKioKLkcFhYm4uLiVOuNGjVKjBkzRvVYQkKCcHJyEnl5efJblP3798vnU1NTBRGZTNZnn30mvL29xbVr14z2dfbs2SI6OrrY48ptLFu2TPj6+qq+Cdm+fbtwcnISWVlZQoj7L3JYWJgoKiqSbQYNGiSGDBkilxcvXmzRr67z588Xvr6+Ii8vz+x1zIUc3VfeHOXl5YkWLVqI5557zux1LIE83VeePNljUIscWZ6j3bt3CyISly5dUj0+aNAgMXjwYJPrlQVydF9ZjqO5c+eKp59+Wr52tvwgjhw59rlOCOTpARxL91WkHBUUFIjQ0FAxaNAgcf36dZGfny/mzZsniEh069bN5HplgRzdV5bjaO3atWLLli0iOTlZbNq0SURFRYlWrVqptlsai6+pbdOmDel0Ornctm1bysjIIL1eLx9r2bKlap0jR47QqlWrqGrVqvKve/fuZDAY6PTp05SamkouLi4UGxsr14mMjKTq1aub7EdSUhLFxMSQn5+fpf8LUmpqKkVHR5OXl5d8rH379mQwGCgtLU0+1rhxY3J2dpbLQUFBdOXKFbk8ceJE2rFjh9n7XbFiBT3//PPk4eFR5r6XBDkqX44KCwtp8ODBJISgTz/9tMx9Lw3yVP5jydaQI+TogYqUo4MHD9JHH31Eq1atUr12toIcOf5xRIQ8EeFYeqAi5cjV1ZU2btxI6enp5OfnR56enrRz507q2bMnOTlZf2oh5Khs57uhQ4dS3759qWnTphQXF0ffffcd/fXXX7Rr1y6z++tidksLKP/niYhyc3Np7NixNHny5GJtQ0NDKT093eJ9VKlSpcz9s5Srq6tqWafTkcFgKNO2EhISKC0tjdatW2eNrpUZcmTcgwHt2bNn6bfffiMfHx9rdbFMkCfHhxypBQYGEtH92SaDgoLk45cvXzY606I9IEdqCQkJdOXKFQoNDZWP6fV6mjp1Ki1cuJDOnDljra6aDTnSBuRJDceS7VnjWIqNjaWkpCS6desWFRQUUEBAAD3++OPFBpf2ghyVrl69elSjRg06ceIEdenSxax1LP6KIjExUbW8b98+atCggWqE/rAWLVpQSkoK1a9fv9ifm5sbRUZGUlFRER08eFCuk5aWRjdv3jS5zWbNmlFSUhJdv37d6PNubm6qb0WMiYqKoiNHjqhmP9u9ezc5OTlRw4YNS1y3rJYvX06xsbEUHR1tk+0TIUdl9WBAm5GRQb/++iv5+/tbdfsPQ54cH3Jkubp161JgYKDqW9nbt29TYmIitW3b1mr7eQA5slx8fDwlJydTUlKS/AsODqZp06bRTz/9ZLX9PIAcaQPyZDkcS46fI6Vq1apRQEAAZWRk0IEDB6hfv35W3wdyZB0XLlyga9euqb4cL43Fg9pz587Ryy+/TGlpabR27VpavHgxTZkypcR1ZsyYQXv27KGJEydSUlISZWRk0JYtW2jixIlERNSwYUPq0aMHjR07lhITE+ngwYM0evToEr9lGDZsGAUGBlJcXBzt3r2bTp06RRs2bKC9e/cS0f0bEp8+fZqSkpIoOzub8vPzi23jQQnwiBEj6NixY7Rz506aNGkSxcfHU61atcx+TZYsWWLWtwi3b9+m9evX0+jRo83edlkgR8WVlqPCwkIaOHAgHThwgNasWUN6vZ6ysrIoKyvLqtOtKyFPxZlzLF2/fp2SkpIoJSWFiO6f2JOSkigrK8vs/ZgLOSqutBzpdDp66aWXaM6cObR161Y6evQoDR8+nIKDgykuLs7s/ZgLOSqutBz5+/tTkyZNVH+urq4UGBhokw8qyFFxjnauI0KejMGxZJyWckREtH79etq1a5e8rc/TTz9NcXFx1K1bN7P3Yy7kqLjScpSbm0vTpk2jffv20ZkzZ2jHjh3Ur18/ql+/PnXv3t3s/Vg8qB0+fDjl5eVR69atacKECTRlyhQaM2ZMies0a9aMfv/9d0pPT6cOHTpQTEwMzZo1S3Uvr5UrV1JwcDB17NiR+vfvT2PGjKGaNWua3Kabmxv9/PPPVLNmTerVqxc1bdqU5s2bJ78JGTBgAPXo0YM6d+5MAQEBRu915OnpST/99BNdv36dWrVqRQMHDqQuXbrQkiVLLHpNsrOz6eTJk6W2+/rrr0kIQcOGDbNo+5ZCjoorLUcXL16krVu30oULF6h58+YUFBQk//bs2WPRvsyFPBVnzrG0detWiomJod69exPR/eswYmJiaOnSpRbtyxzIUXHm5Gj69Ok0adIkGjNmDLVq1Ypyc3Ppxx9/tMk8AshRcea+J9kLclSco53riJAnY3AsGae1HGVmZlJ8fDxFRkbS5MmTKT4+3vJ7oJoJOSqutBw5OztTcnIy9e3blyIiImjUqFEUGxtLCQkJFt0zV/f/Z7oyS6dOnah58+a0cOFCs3cA9oUcaQPy5PiQI8eHHDk+5EgbkCfHhxw5PuTo0bL+tF8AAAAAAAAAdoJBLQAAAAAAAGiWReXHAAAAAAAAAI4Ev9QCAAAAAACAZmFQCwAAAAAAAJqFQS0AAAAAAABoFga1AAAAAAAAoFkY1AIAAAAAAIBmYVALAAAAAAAAmoVBLQAAAAAAAGgWBrUAAAAAAACgWRjUAgAAAAAAgGb9P/ILLkHZcU+fAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1200x500 with 9 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = model.predict(x_test_processed)\n",
    "plt.figure(figsize=(12, 5))\n",
    "for i in range(9):\n",
    "    plt.subplot(1, 9, i+1)\n",
    "    prediction = predictions[i].argmax()\n",
    "    image =plt.imshow(x_test_processed[i])\n",
    "    plt.xlabel('prediction: '+str(prediction))\n",
    "    plt.xticks([])  # Hide the x-axis scale and ticks\n",
    "    plt.yticks([])  # Hide the y-axis scale and ticks\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
