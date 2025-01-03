from ssd_model import build_ssd_model
from data_generator import data_generator
from ssd_loss import ssd_loss
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPUs available {len(gpus)}")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("No GPU detected")
    
# Building the model
model = build_ssd_model(input_shape=(300, 300, 3), num_classes=3)

# Creating data generators
train_gen = data_generator('ssd_dataset/train/images', 'ssd_dataset/train/annotations', batch_size=8, num_classes=3)
val_gen = data_generator('ssd_dataset/val/images', 'ssd_dataset/val/annotations', batch_size=8, num_classes=3)

# Compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=ssd_loss)


# Training the model
history = model.fit(
    train_gen,
    steps_per_epoch=5,  
    epochs=80,
    validation_data=val_gen,
    validation_steps=5  
)


# Saving the model
model.save("ssd_model.h5")


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


