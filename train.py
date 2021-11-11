import os
import itertools
import json

import idx2numpy
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

S3_BUCKET = os.getenv('S3_BUCKET', 's3://daviddvctest/cache') 
EPOCHS = os.getenv('EPOCHS', 1) 
CHECKPOINT_FOLDER = os.getenv('CHECKPOINT_FOLDER', 'output')
TB_LOG_DIR = os.getenv('TB_LOG_DIR', os.path.join(CHECKPOINT_FOLDER, 'tblogs'))

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

def sync_s3(push=False):
    print('Retrieving cache...')

    s3_path = S3_BUCKET

    if os.environ.get('CI', False):
        repo = os.environ.get('GITHUB_REPOSITORY', False)
        run_id = os.environ.get('GITHUB_RUN_ID')

        if not repo:
            repo = os.environ.get('CI_PROJECT_NAME', False)
            run_id = os.environ.get('CI_PIPELINE_ID')

        if not repo:
            repo = os.environ.get('BITBUCKET_REPO_FULL_NAME', False)
            run_id = os.environ.get('BITBUCKET_BUILD_NUMBER')  
 
        s3_path = os.path.join(S3_BUCKET, repo, run_id)

    command = f'aws s3 sync {s3_path} {CHECKPOINT_FOLDER}'
    if push:
        command = f'aws s3 sync {CHECKPOINT_FOLDER} {s3_path}'
            
    os.system(command)

def model_path():
    return os.path.join(CHECKPOINT_FOLDER, 'seq.h5')

def model_path_info():
    return model_path() + '.epoch'

def model_metrics():
    return os.path.join(CHECKPOINT_FOLDER, 'metrics.json')

def model_cmatrix():
    return os.path.join(CHECKPOINT_FOLDER, 'confusion_matrix.png')

def create_model(checkpoints_path):
    try:
        model = tf.keras.models.load_model(checkpoints_path)
        print('Model found. Resuming...')
        return model
    except:
        print('Failed loading checkpoints. Starting from zero...')

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

sync_s3()

try: 
    with open(model_path_info(), 'r') as fh:
        initial_epoch = int(fh.readline());       
except:
    initial_epoch = 0

if (initial_epoch == EPOCHS):
    print('Nothing to do. Model is already trained')
    exit()

model = create_model(model_path());

def save_model(epoch, logs):
    print(f'Saving epoch: {str(epoch)}')

    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    model.save(model_path())

    with open(model_path_info(), 'w') as fh:
        fh.write(f'{str(epoch+1)}\n')

    sync_s3(push=True)

def log_metrics(epoch, logs):
    print(epoch, logs)

    with open(model_metrics(), 'w') as fh:
        json.dump({ 
            'accuracy': logs.get('accuracy'), 
            'loss': logs.get('loss')
        }, fh)


    os.system('echo "# CML report\N:wave: from TPI" > report.md && cat output/metrics.json >> report.md && cml-publish output/confusion_matrix.png --md >> report.md && cml-send-comment --token ' + os.environ.get('REPO_TOKEN') + ' report.md')


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
    plt.title(f'Epoch {str(epoch)}')

    plt.savefig(model_cmatrix())

model.fit(train_images,
          train_labels,
          epochs=EPOCHS,
          initial_epoch=initial_epoch,
          verbose=0,
          callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR, histogram_freq=1),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_metrics),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model)
          ],
          validation_data=(test_images, test_labels))
