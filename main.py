import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

def filter_out_corrupted_images():
    #### filter out corrupted images.
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("/home/diako/Downloads/kagglecatsanddogs_3367a/PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    print("Deleted %d images" % num_skipped)


def data_augmentation():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )
    return data_augmentation

def visualize_data_augmentation(train_ds, data_augmentation):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.axis("off")
            plt.imshow(augmented_images[1].numpy().astype("uint8"))
        plt.show()


def generate_dataset():
    image_size = (180, 180)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/home/diako/Downloads/kagglecatsanddogs_3367a/PetImages",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/home/diako/Downloads/kagglecatsanddogs_3367a/PetImages",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds
def visualize_data(train_ds):
    #Here are the first 9 images in the training dataset
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
        plt.show()

def standardizing_data(train_ds , data_augm):
    # normal from [0,255] to [0,1]
    augmented_train_ds = train_ds.map(
    lambda x, y: (data_augm(x, training=True), y))
    return augmented_train_ds


def make_model(data_augmn, input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmn(inputs)

    # Entry block
    # rescaling data in layers:(option 2) this option is best option fo gpu in term of speed and prformance.
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)



def main():
    # filter_out_corrupted_images()
    train_ds , val_ds = generate_dataset()
    # visualize_data(train_ds)
    data_augm = data_augmentation()
    # visualize_data_augmentation(train_ds, data_augm)
#rescaling data:(option 1) this option is not optimized.
    # augmented_train_ds = standardizing_data(train_ds , data_augm)

#Let's make sure to use buffered prefetching so
# we can yield data from disk without having I/O becoming blocking:
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
#Build a model
    model = make_model(data_augm, input_shape=(180, 180) + (3,), num_classes=2 )
    # keras.utils.plot_model(model, show_shapes=True)

# Train the model
    epochs = 5
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

# # Run inference and prediction on new data
#     img = keras.preprocessing.image.load_img(
#         "/home/diako/Downloads/kagglecatsanddogs_3367a/PetImages/Cat/6779.jpg", target_size=(180, 180)
#     )
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#
#     predictions = model.predict(img_array)
#     score = predictions[0]
#     print(
#         "This image is %.2f percent cat and %.2f percent dog."
#         % (100 * (1 - score), 100 * score)
#     )

if __name__=="__main__":
    main()