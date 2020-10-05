from typing import Iterator


import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import TensorBoard


import tensorflow_addons as tfa
from tensorflow.keras.losses import MeanAbsoluteError
from datetime import datetime
from tensorflow import keras
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0


def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """[summary]

    Parameters
    ----------
    data_generator : Iterator
        [description]
    df : pd.DataFrame
        [description]
    """
    # super hacky way of creating a small dataframe with one image
    series = df.iloc[1]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(  # type: ignore
        dataframe=df_augmentation_visualization,
        x_col="image_location",
        y_col="price",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=1,
    )

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        batch = next(iterator_visualizations)
        img = batch[0]  # type: ignore
        img = img[0, :, :, :]
        plt.imshow(img)
    plt.show()


def create_generators(df: pd.DataFrame) -> Tuple[Iterator, Iterator, Iterator]:
    """[summary]

    Parameters
    ----------
    df : pd.DataFrame
        [description]

    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        [description]
    """
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.2,
    )

    validation_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    # visualize image augmentations
    # visualize_augmentations(train_generator, df)

    # TODO: think about actually using the test data later?

    train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.25, random_state=1)
    print(train.shape)  # type: ignore
    print(val.shape)  # type: ignore
    print(test.shape)  # type: ignore

    print(train.describe())  # type: ignore
    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image_location",
        y_col="price",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=128,
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val,
        x_col="image_location",
        y_col="price",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=128,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test,
        x_col="image_location",
        y_col="price",
        class_mode="raw",
        target_size=(224, 224),
        batch_size=128,
    )
    return train_generator, validation_generator, test_generator


def small_cnn() -> Sequential:
    """[summary]

    Returns
    -------
    Sequential
        [description]
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))

    return model


def run_small_cnn(
    model_name: str,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
):
    """[summary]

    Parameters
    ----------
    model_name : str
        [description]
    train_generator : Iterator
        [description]
    validation_generator : Iterator
        [description]
    test_generator : Iterator
        [description]
    """
    logdir = "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model = small_cnn()
    model.summary()

    radam = tfa.optimizers.RectifiedAdam(learning_rate=0.01)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback],
        workers=6,
    )
    print(history.history)
    model.evaluate(
        test_generator,
        callbacks=[tensorboard_callback],
    )


def adapt_efficient_net():
    inputs = layers.Input(shape=(224, 224, 3))
    #! problematic
    # x = img_augmentation(inputs)

    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=5e-2)
    #! does not learn at the moment
    radam = tfa.optimizers.RectifiedAdam(learning_rate=5)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    # optimizer = ranger
    model.compile(optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError()])
    return model


def run_efficient_net(model_name, train_generator, validation_generator, test_generator):
    model = adapt_efficient_net()
    model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        # callbacks=[tensorboard_callback],
        workers=6,
    )


def run():

    df = pd.read_pickle("./data/df.pkl")
    df["image_location"] = "./data/processed_images/" + df["zpid"] + ".png"
    # df = df.iloc[0:1000]

    train_generator, validation_generator, test_generator = create_generators(df)

    # Larger variants of EfficientNet do not guarantee improved performance, especially for tasks with less data or fewer classes. In such a case, the larger variant of EfficientNet chosen, the harder it is to tune hyperparameters.
    # TODO: use small variant of EfficientNet
    # TODO: reprogram efficient net example
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    # TODO: try ranger as optimizer in keras example
    # TODO: change code to regression and custom dataset
    # TODO: use latest weights https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#using-the-latest-efficientnet-weights
    # TODO add early stoppng
    # run_small_cnn("small_cnn", train_generator, validation_generator, test_generator)
    # TODO add more naming options for learning rate etc for TB
    run_efficient_net("eff", train_generator, validation_generator, test_generator)

    # tensorboard --logdir logs/scalars


if __name__ == "__main__":
    run()
