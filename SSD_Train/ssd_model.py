import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Concatenate

def build_ssd_model(input_shape=(300, 300, 3), num_classes=3):
    """
    Builds an SSD300 model with a simple CNN backbone.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        num_classes (int): Number of object classes (excluding background).

    Returns:
        tf.keras.Model: SSD300 model.
    """
    # Input layer
    input_layer = Input(shape=input_shape)

    # Backbone CNN
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Feature map layers
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Prediction Heads
    # Classification Predictions
    class_preds = Conv2D(num_classes * 4, (3, 3), padding='same', activation='softmax')(conv5)  # 4 boxes per grid
    class_preds = Reshape((-1, num_classes))(class_preds)  # Flatten predictions

    # Localization Predictions
    loc_preds = Conv2D(4 * 4, (3, 3), padding='same')(conv5)  # 4 coordinates (x, y, w, h) per box
    loc_preds = Reshape((-1, 4))(loc_preds)  # Flatten predictions

    # Combine predictions
    output = Concatenate(axis=-1)([class_preds, loc_preds])

    # Create SSD model
    model = Model(inputs=input_layer, outputs=output)
    return model

# Instantiate the model
input_shape = (300, 300, 3)  # Input image size
num_classes = 3  # Number of object classes (e.g., mask, no mask, incorrect mask)
ssd_model = build_ssd_model(input_shape=input_shape, num_classes=num_classes)

# Print model summary
ssd_model.summary()
