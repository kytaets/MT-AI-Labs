import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# HYPERPARAMS
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
MODEL_PATH = "mnist_cnn.h5"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train[:-10000], x_train[-10000:]
y_train, y_val = y_train[:-10000], y_train[-10000:]

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)

y_train_cat = utils.to_categorical(y_train, 10)
y_val_cat = utils.to_categorical(y_val, 10)
y_test_cat = utils.to_categorical(y_test, 10)

# 2. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08
)
datagen.fit(x_train)

# 3. Model definition
def build_model():
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inp, out)
    return model

model = build_model()
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
cb = [
    callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

# 4. Training
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val_cat),
    callbacks=cb
)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save final model
model.save(MODEL_PATH)

# 5. Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# 6. Utilities for noisy / corrupted images
def add_gaussian_noise(images, mean=0.0, std=0.2):
    noisy = images + np.random.normal(mean, std, images.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def random_erase(images, erase_prob=0.5, size_fraction=0.2):
    out = images.copy()
    N, H, W, C = out.shape
    for i in range(N):
        if random.random() < erase_prob:
            h_erase = int(H * size_fraction * random.uniform(0.5, 1.0))
            w_erase = int(W * size_fraction * random.uniform(0.5, 1.0))
            y = random.randint(0, H - h_erase)
            x = random.randint(0, W - w_erase)
            out[i, y:y+h_erase, x:x+w_erase, :] = 0.0
    return out

# 7. Test examples
num_examples = 12
idx = np.random.choice(len(x_test), num_examples, replace=False)
originals = x_test[idx]
labels = y_test[idx]

noisy = add_gaussian_noise(originals, std=0.35)
erased = random_erase(originals, erase_prob=1.0, size_fraction=0.3)

def predict_and_show(images, title):
    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    plt.figure(figsize=(12,3))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.axis('off')
        plt.imshow(images[i].squeeze(), cmap='gray')
        col = 'green' if pred_labels[i] == labels[i] else 'red'
        plt.title(f"p:{pred_labels[i]}\ntrue:{labels[i]}", color=col)
    plt.suptitle(title)
    plt.show()

predict_and_show(originals, "Original test images")
predict_and_show(noisy, "Gaussian noisy images (std=0.35)")
predict_and_show(erased, "Random erased images")

# 8. Confusion matrix
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 9. Save example images
os.makedirs("examples", exist_ok=True)
for i in range(6):
    plt.imsave(f"examples/orig_{i}.png", originals[i].squeeze(), cmap='gray')
    plt.imsave(f"examples/noisy_{i}.png", noisy[i].squeeze(), cmap='gray')
    plt.imsave(f"examples/erased_{i}.png", erased[i].squeeze(), cmap='gray')

print("Готово. Модель збережена як", MODEL_PATH)
