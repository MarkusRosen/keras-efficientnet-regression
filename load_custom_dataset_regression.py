from typing import Iterator, List, Union, Tuple, Dict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
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
    series = df.iloc[2]

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
    plt.close()


def get_mean_baseline(train, val):
    y_hat = train["price"].mean()
    val["y_hat"] = y_hat
    mae = MeanAbsoluteError()
    mae = mae(val["price"], val["y_hat"]).numpy()
    mape = MeanAbsolutePercentageError()
    mape = mape(val["price"], val["y_hat"]).numpy()

    print(mae)
    print("mean baseline MAPE: ", mape)

    return mape


def split_data(df):
    train, val = train_test_split(df, test_size=0.2, random_state=1)
    train, test = train_test_split(train, test_size=0.125, random_state=1)

    print("shape train: ", train.shape)  # type: ignore
    print("shape val: ", val.shape)  # type: ignore
    print("shape test: ", test.shape)  # type: ignore

    print("Descriptive statistics of train:")
    print(train.describe())  # type: ignore
    return train, val, test


def create_generators(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[Iterator, Iterator, Iterator]:
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


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """[summary]

    Parameters
    ----------
    model_name : str
        [description]

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        [description]
    """
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

    model_checkpoint_callback = ModelCheckpoint(
        "./data/models/" + model_name,
        monitor="val_mean_absolute_percentage_error",
        verbose=0,
        save_best_only=True,
        mode="min",
        save_freq="epoch",
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback]  # , model_checkpoint_callback


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
) -> Dict:
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

    model.evaluate(
        test_generator,
        callbacks=callbacks,
    )
    return history


def adapt_efficient_net() -> Model:
    inputs = layers.Input(shape=(224, 224, 3))

    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="efficientnetb0_notop.h5")
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(1, name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model


def plot_results(model_history_small_cnn, model_history_eff_net, mean_baseline):

    dict1 = {
        "MAPE": model_history_small_cnn.history["mean_absolute_percentage_error"],
        "type": "training",
        "model": "small_cnn",
    }
    dict2 = {
        "MAPE": model_history_small_cnn.history["val_mean_absolute_percentage_error"],
        "type": "validation",
        "model": "small_cnn",
    }
    dict3 = {
        "MAPE": model_history_eff_net.history["mean_absolute_percentage_error"],
        "type": "training",
        "model": "eff_net",
    }
    dict4 = {
        "MAPE": model_history_eff_net.history["val_mean_absolute_percentage_error"],
        "type": "validation",
        "model": "eff_net",
    }
    history_data = [dict1, dict2, dict3, dict4]
    s1 = pd.DataFrame(dict1)
    s2 = pd.DataFrame(dict2)
    s3 = pd.DataFrame(dict3)
    s4 = pd.DataFrame(dict4)
    df = pd.concat([s1, s2, s3, s4], axis=0).reset_index()
    grid = sns.relplot(data=df, x=df["index"], y="MAPE", hue="model", col="type", kind="line", legend=False)
    grid.set(ylim=(20, 100))
    for ax in grid.axes.flat:
        ax.axhline(y=mean_baseline, color="lightcoral", linestyle="dashed")
        ax.set(xlabel="Epoch")
    labels = ["small_cnn", "eff_net", "mean_baseline"]

    plt.legend(labels=labels)
    plt.savefig("training_validation.png")
    plt.show()


def run():

    df = pd.read_pickle("./data/df.pkl")
    df["image_location"] = "./data/processed_images/" + df["zpid"] + ".png"
    # df = df.iloc[0:2000]
    train, val, test = split_data(df)
    mean_baseline = get_mean_baseline(train, val)
    train_generator, validation_generator, test_generator = create_generators(df, train, val, test)

    small_cnn_history = run_model(
        model_name="small_cnn",
        model_function=small_cnn(),
        lr=0.001,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    eff_net_history = run_model(
        model_name="eff_net",
        model_function=adapt_efficient_net(),
        lr=0.5,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )

    plot_results(small_cnn_history, eff_net_history, mean_baseline)

    # wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b0.tar.gz
    # tar -xf noisy_student_efficientnet-b0.tar.gz
    # python efficientnet_weight_update_util.py --model b0 --notop --ckpt noisy_student_efficientnet-b0/model.ckpt --o efficientnetb0_notop.h5
    # tensorboard --logdir logs/scalars

    # mean baselines 188311.890625 28.71662139892578
    # 17m30s
    # 3m40s
    """ small cnn
    Epoch 15/100
60/60 [==============================] - 48s 805ms/step - loss: 185176.3281 - mean_absolute_error: 185115.5625 - mean_absolute_percentage_error: 25.4920 - val_loss: 187887.1094 - val_mean_absolute_error: 185792.2969 - val_mean_absolute_percentage_error: 28.6558
Epoch 16/100
60/60 [==============================] - 48s 792ms/step - loss: 184898.2656 - mean_absolute_error: 184811.8281 - mean_absolute_percentage_error: 25.4640 - val_loss: 184086.2188 - val_mean_absolute_error: 183725.7188 - val_mean_absolute_percentage_error: 28.2318
Epoch 17/100
60/60 [==============================] - 48s 805ms/step - loss: 184179.1094 - mean_absolute_error: 183871.7031 - mean_absolute_percentage_error: 25.3496 - val_loss: 186079.0469 - val_mean_absolute_error: 185818.1875 - val_mean_absolute_percentage_error: 28.3344
Epoch 18/100
60/60 [==============================] - 48s 803ms/step - loss: 184758.2344 - mean_absolute_error: 184760.5469 - mean_absolute_percentage_error: 25.4764 - val_loss: 184336.3438 - val_mean_absolute_error: 180909.9375 - val_mean_absolute_percentage_error: 27.1706
Epoch 19/100
60/60 [==============================] - 48s 793ms/step - loss: 185315.5625 - mean_absolute_error: 185343.5938 - mean_absolute_percentage_error: 25.5804 - val_loss: 184969.5781 - val_mean_absolute_error: 185502.6406 - val_mean_absolute_percentage_error: 27.9391
Epoch 20/100
60/60 [==============================] - 47s 789ms/step - loss: 184386.4531 - mean_absolute_error: 184337.3125 - mean_absolute_percentage_error: 25.4744 - val_loss: 183334.1406 - val_mean_absolute_error: 180248.4844 - val_mean_absolute_percentage_error: 27.1078
Epoch 21/100
60/60 [==============================] - 47s 782ms/step - loss: 184395.2188 - mean_absolute_error: 184380.7969 - mean_absolute_percentage_error: 25.4457 - val_loss: 187023.0625 - val_mean_absolute_error: 186611.1406 - val_mean_absolute_percentage_error: 28.2821
Epoch 22/100
60/60 [==============================] - 48s 807ms/step - loss: 183725.0000 - mean_absolute_error: 183510.2031 - mean_absolute_percentage_error: 25.3401 - val_loss: 182207.5938 - val_mean_absolute_error: 181847.2812 - val_mean_absolute_percentage_error: 28.3565
Epoch 23/100
60/60 [==============================] - 50s 825ms/step - loss: 183497.7656 - mean_absolute_error: 183333.9844 - mean_absolute_percentage_error: 25.3179 - val_loss: 183527.0938 - val_mean_absolute_error: 190079.0781 - val_mean_absolute_percentage_error: 27.6561
Epoch 24/100
60/60 [==============================] - 51s 847ms/step - loss: 183842.5781 - mean_absolute_error: 183972.0938 - mean_absolute_percentage_error: 25.3546 - val_loss: 185380.4062 - val_mean_absolute_error: 183492.7188 - val_mean_absolute_percentage_error: 28.2040
Epoch 25/100

# test error
9/9 [==============================] - 5s 561ms/step - loss: 187197.9062 - mean_absolute_error: 186815.7500 - mean_absolute_percentage_error: 27.8397

"""


"""eff_net
 2/60 [>.............................] - ETA: 3:16 - loss: 710118.0625 - mean_absolute_error: 715287.6250 - mean_absolute_percentage_error: 99.9999WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 1.7491s vs `on_train_batch_end` time: 5.0077s). Check your callbacks.
60/60 [==============================] - 62s 1s/step - loss: 705718.3125 - mean_absolute_error: 705919.2500 - mean_absolute_percentage_error: 99.7793 - val_loss: 697795.8125 - val_mean_absolute_error: 689881.5000 - val_mean_absolute_percentage_error: 99.0928
Epoch 2/100
60/60 [==============================] - 52s 873ms/step - loss: 675638.8750 - mean_absolute_error: 675420.8750 - mean_absolute_percentage_error: 95.0084 - val_loss: 633470.1875 - val_mean_absolute_error: 626153.2500 - val_mean_absolute_percentage_error: 88.8780
Epoch 3/100
60/60 [==============================] - 62s 1s/step - loss: 541324.6250 - mean_absolute_error: 541999.3750 - mean_absolute_percentage_error: 73.9094 - val_loss: 426912.6562 - val_mean_absolute_error: 431120.6250 - val_mean_absolute_percentage_error: 56.7776
Epoch 4/100
60/60 [==============================] - 55s 921ms/step - loss: 273624.1875 - mean_absolute_error: 274123.1562 - mean_absolute_percentage_error: 34.1725 - val_loss: 183415.8906 - val_mean_absolute_error: 183893.3594 - val_mean_absolute_percentage_error: 23.9141
Epoch 5/100
60/60 [==============================] - 55s 918ms/step - loss: 185995.6406 - mean_absolute_error: 186042.1094 - mean_absolute_percentage_error: 25.4941 - val_loss: 180898.1094 - val_mean_absolute_error: 180924.5312 - val_mean_absolute_percentage_error: 25.4735
Epoch 6/100
60/60 [==============================] - 57s 948ms/step - loss: 185689.1875 - mean_absolute_error: 185715.0312 - mean_absolute_percentage_error: 25.5540 - val_loss: 180900.4062 - val_mean_absolute_error: 175182.1875 - val_mean_absolute_percentage_error: 24.2947
Epoch 7/100
60/60 [==============================] - 56s 940ms/step - loss: 185837.9375 - mean_absolute_error: 185846.5156 - mean_absolute_percentage_error: 25.4831 - val_loss: 180961.9062 - val_mean_absolute_error: 183433.9375 - val_mean_absolute_percentage_error: 24.7834

# testing error
9/9 [==============================] - 6s 635ms/step - loss: 183612.8750 - mean_absolute_error: 185082.6719 - mean_absolute_percentage_error: 23.9706
"""

# TODO: write comments and docstrings


if __name__ == "__main__":
    run()
