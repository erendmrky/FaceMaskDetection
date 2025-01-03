import tensorflow as tf

def ssd_loss(y_true, y_pred):
    # Extract classification and localization parts
    num_classes = 3  # Update based on your dataset
    class_true = y_true[:, :, :num_classes]
    loc_true = y_true[:, :, num_classes:]

    class_pred = y_pred[:, :, :num_classes]
    loc_pred = y_pred[:, :, num_classes:]

    # Classification loss (only for positive anchors)
    classification_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(class_true, class_pred)

    # Localization loss (only for positive anchors)
    localization_loss = tf.keras.losses.MeanSquaredError()(loc_true, loc_pred)

    return classification_loss + localization_loss

if __name__ == "__main__":
    print("SSD Loss function ready.")
