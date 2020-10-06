from typing import Iterator, List, Union, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import plot_model


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


def mean_baseline(train, val):
    y_hat = train["price"].mean()
    val["y_hat"] = y_hat
    mae = MeanAbsoluteError()
    mape = MeanAbsolutePercentageError()
    print("================================")
    print(mae(val["price"], val["y_hat"]).numpy())
    print(mape(val["price"], val["y_hat"]).numpy())
    print("================================")
    print("================================")
    print("================================")


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

    train, val = train_test_split(df, test_size=0.2, random_state=1)
    train, test = train_test_split(train, test_size=0.125, random_state=1)

    mean_baseline(train, val)
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


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping]]:
    logdir = "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,
        patience=10,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )
    return [tensorboard_callback, early_stopping_callback]


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


def run_model(
    model_name: str,
    model_function: Model,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
):
    """[summary]

    Parameters
    ----------
    model_name : str
        [description]
    model_function : Model
        [description]
    lr : float
        [description]
    train_generator : Iterator
        [description]
    validation_generator : Iterator
        [description]
    test_generator : Iterator
        [description]
    """

    callbacks = get_callbacks(model_name)
    model = model_function
    model.summary()
    plot_model(model, to_file=model_name + ".jpg", show_shapes=True)

    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers=6,
    )
    # print(history.history)
    model.evaluate(
        test_generator,
        callbacks=callbacks,
    )


def adapt_efficient_net() -> Model:
    inputs = layers.Input(shape=(224, 224, 3))

    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4  # 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model


def run():

    df = pd.read_pickle("./data/df.pkl")
    df["image_location"] = "./data/processed_images/" + df["zpid"] + ".png"
    # df = df.iloc[0:2000]

    train_generator, validation_generator, test_generator = create_generators(df)

    run_model(
        model_name="small_cnn",
        model_function=small_cnn(),
        lr=0.001,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    run_model(
        model_name="eff_net",
        model_function=adapt_efficient_net(),
        lr=0.5,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    # wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b0.tar.gz
    # tar -xf noisy_student_efficientnet-b0.tar.gz
    # python efficientnet_weight_update_util.py --model b0 --notop --ckpt noisy_student_efficientnet-b0/model.ckpt --o efficientnetb0_notop.h5
    # tensorboard --logdir logs/scalars

    # mean baselines 188311.890625 28.71662139892578
    # val error small 61s 1s/step - loss: 184070.1875 - mean_absolute_error: 184073.7188 - mean_absolute_percentage_error: 25.3885 - val_loss: 182803.8438 - val_mean_absolute_error: 180425.0000 - val_mean_absolute_percentage_error: 27.3221
    # val error eff netEpoch 4/100 60/6066s 1s/step - loss: 273727.2812 - mean_absolute_error: 274309.0312 - mean_absolute_percentage_error: 34.1479 - val_loss: 183513.7344 - val_mean_absolute_error: 181217.1719 - val_mean_absolute_percentage_error: 24.0258
    # Test error small - loss: 184779.7656 - mean_absolute_error: 183014.6562 - mean_absolute_percentage_error: 26.9627 # epoch 17, 17m30s
    # Test error eff net 656ms/step - loss: 183692.4531 - mean_absolute_error: 185714.2031 - mean_absolute_percentage_error: 24.1210 epoch 4, 3m40s


if __name__ == "__main__":
    run()
