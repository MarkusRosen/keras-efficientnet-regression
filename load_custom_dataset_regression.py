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
    print("================================")
    print(mae)
    print(mape)
    print("================================")
    print("================================")
    print("================================")
    return mape


def split_data(df):
    train, val = train_test_split(df, test_size=0.2, random_state=1)
    train, test = train_test_split(train, test_size=0.125, random_state=1)

    print(train.shape)  # type: ignore
    print(val.shape)  # type: ignore
    print(test.shape)  # type: ignore

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

    print(model_history_small_cnn.history["mean_absolute_percentage_error"])
    print(model_history_small_cnn.history["val_mean_absolute_percentage_error"])
    # history_data = {
    #    "small_cnn_train": model_history_small_cnn.history["mean_absolute_percentage_error"],
    #    "small_cnn_val": model_history_small_cnn.history["val_mean_absolute_percentage_error"],
    #   "eff_net_train": model_history_eff_net.history["mean_absolute_percentage_error"],
    #   "eff_net_val": model_history_eff_net.history["val_mean_absolute_percentage_error"],
    # }
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

    print(df)


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
    # val error small 61s 1s/step - loss: 184070.1875 - mean_absolute_error: 184073.7188 - mean_absolute_percentage_error: 25.3885 - val_loss: 182803.8438 - val_mean_absolute_error: 180425.0000 - val_mean_absolute_percentage_error: 27.3221
    # val error eff netEpoch 4/100 60/6066s 1s/step - loss: 273727.2812 - mean_absolute_error: 274309.0312 - mean_absolute_percentage_error: 34.1479 - val_loss: 183513.7344 - val_mean_absolute_error: 181217.1719 - val_mean_absolute_percentage_error: 24.0258
    # Test error small - loss: 184779.7656 - mean_absolute_error: 183014.6562 - mean_absolute_percentage_error: 26.9627 # epoch 17, 17m30s
    # Test error eff net 656ms/step - loss: 183692.4531 - mean_absolute_error: 185714.2031 - mean_absolute_percentage_error: 24.1210 epoch 4, 3m40s

    # TODO: write comments and docstrings
    # TODO: screenshot tensorboard
    # TODO: seaborn relplot training/validation curves of MAPE


if __name__ == "__main__":
    run()
