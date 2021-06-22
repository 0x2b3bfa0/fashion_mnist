import io
import os
from datetime import datetime
import itertools

import idx2numpy
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

EPOCHS = 2
INITIAL_EPOCH = 0
CHECKPOINT_FOLDER = 'model'
CHECKPOINT_NAME = 'seq.h5'

# Load the data.
train_images = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

test_images = idx2numpy.convert_from_file("data/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels-idx1-ubyte")

# Pre-process images
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Labels of classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def model_path():
    return os.path.join(CHECKPOINT_FOLDER, CHECKPOINT_NAME)

def create_model():
    global INITIAL_EPOCH
    try:
        model = tf.keras.models.load_model(model_path())
        with open(model_path() + '.epoch', 'r') as fh:
            INITIAL_EPOCH = int(fh.readline());
        
        print('Model found. Resuming...')
        return model
    except:
        print('No checkpoints found.')

    # Build classifier
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# Train the classifier
model = create_model();

def save_model(epoch, logs):
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    model.save(model_path())

    with open(model_path() + '.epoch', 'w') as outfile:
        outfile.write(str(epoch) + "\n")

def log_metrics(epoch, logs):
    with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(logs['val_acc']) + "\n")
        outfile.write("Loss: " + str(logs['loss']) + "\n")

def log_confusion_matrix(epoch, logs):
    test_pred = np.argmax(model.predict(test_images), axis=1)
    
    cm = confusion_matrix(test_labels, test_pred)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    tick_marks = np.arange(len(class_names))

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Epoch ' + str(epoch))

    plt.savefig('confusion_matrix.png')

history = model.fit(train_images,
          train_labels,
          epochs=EPOCHS,
          initial_epoch=INITIAL_EPOCH,
          verbose=0,
          callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metrics),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
          ],
          validation_data=(test_images, test_labels))

log_metrics(INITIAL_EPOCH, history.history)
log_confusion_matrix(INITIAL_EPOCH, None)
