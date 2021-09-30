from __future__ import print_function, division

from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Conv1DTranspose

import pandas as pd
import numpy as np
from collections import OrderedDict

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Layer,
    MultiHeadAttention,
    LayerNormalization,
    Embedding,
)

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import random

random.seed(10)
np.random.seed(10)
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class SequenceLengthError(Exception):
    pass


class ApplianceNotFoundError(Exception):
    pass


# This code is inspired from :
# https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, att_weights = self.att(
            inputs, inputs, return_attention_scores=True
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "att": self.att,
                "ffn": self.ffn,
                "layernorm1": self.layernorm1,
                "layernorm2": self.layernorm2,
                "dropout1": self.dropout1,
                "dropout2": self.dropout2,
            }
        )
        return config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        # x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "token_emb": self.token_emb,
                "pos_emb": self.pos_emb,
            }
        )
        return config


class LPpool(Layer):
    def __init__(self, pool_size, strides=None, padding="same"):
        super(LPpool, self).__init__()
        self.avgpool = tf.keras.layers.AveragePooling1D(pool_size, strides, padding)

    def call(self, x):
        x = tf.math.pow(tf.math.abs(x), 2)
        x = self.avgpool(x)
        x = tf.math.pow(x, 1.0 / 2)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "avgpool": self.avgpool,
            }
        )
        return config


class MultiBERT(Disaggregator):
    def __init__(self, params):

        self.MODEL_NAME = "MultiBERT"
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 481)
        self.n_epochs = params.get("n_epochs", 10)
        self.model = None
        self.mains_mean = []
        self.mains_std = []
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.appliances = []
        self.embed_dim = params.get("embed_dim", 64)  # Embedding size for each token
        self.num_heads = params.get("num_heads", 12)  # Number of attention heads
        self.ff_dim = params.get(
            "ff_dim", 128
        )  # Hidden layer size in feed forward network inside transformer
        self.layers = params.get("layers", 4)  # Number of transformer blocks
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **load_kwargs,
    ):

        print(f"...............{self.MODEL_NAME} partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, "train"
            )
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))

        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append(app_df_values)
            self.appliances.append(app_name)
        appliance_power = np.stack(new_train_appliances, axis=-1)
        train_appliances = new_train_appliances

        if train_main.size > 0:
            # Sometimes chunks can be empty after dropping NANS
            if len(train_main) > 10:
                # Do validation when you have sufficient samples
                self.model = self.return_network(appliance_power.shape[-1])
                filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                    "_".join("All".split()),
                    current_epoch,
                )
                checkpoint = ModelCheckpoint(
                    filepath,
                    monitor="val_loss",
                    verbose=1,
                    save_best_only=True,
                    mode="min",
                )
                train_x, v_x, train_y, v_y = train_test_split(
                    train_main, appliance_power, test_size=0.15, random_state=10
                )
                self.model.fit(
                    train_x,
                    train_y,
                    validation_data=(v_x, v_y),
                    epochs=self.n_epochs,
                    callbacks=[checkpoint],
                    batch_size=self.batch_size,
                )
                self.model.load_weights(filepath)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method="test"
            )

        test_predictions = []
        for test_mains_df in test_main_list:

            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape(
                (-1, self.sequence_length, 1)
            )

            predictions = self.model.predict(
                test_main_array, batch_size=self.batch_size
            )
            print(predictions.shape)

            for app, appliance in enumerate(self.appliances):
                prediction = self.appliance_params[appliance]["mean"] + (
                    predictions[:, :, app] * self.appliance_params[appliance]["std"]
                )
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(
                    valid_predictions > 0, valid_predictions, 0
                )
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype="float32")
            test_predictions.append(results)
        return test_predictions

    def return_network(self, num_appliances):
        """Creates the BERT module"""
        maxlen = self.sequence_length  # maxlength for attention

        model = Sequential()
        model.add(
            Conv1D(
                self.embed_dim // 4,
                5,
                activation="linear",
                input_shape=(self.sequence_length, 1),
                padding="same",
                strides=1,
            )
        )
        model.add(
            Conv1D(
                self.embed_dim // 2,
                5,
                activation="linear",
                padding="same",
                strides=1,
            )
        )
        model.add(
            Conv1D(
                self.embed_dim,
                5,
                activation="linear",
                padding="same",
                strides=1,
            )
        )
        model.add(LPpool(pool_size=2))

        # Token and Positional embedding and Encoder part of the transformer
        model.add(TokenAndPositionEmbedding(maxlen, self.embed_dim))
        for _ in range(self.layers):
            model.add(TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim))

        # Fully connected layer
        model.add(
            Conv1DTranspose(
                self.ff_dim, kernel_size=4, strides=2, output_padding=1, padding="same"
            )
        )
        model.add(Dense(128, activation="tanh"))
        model.add(Dense(num_appliances))
        model.summary()
        model.compile(loss="mse", optimizer="adam", metrics=["mse"])
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == "train":
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                mains_mean = np.mean(new_mains)
                mains_std = np.std(new_mains)
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(
                    new_mains,
                    (units_to_pad, units_to_pad),
                    "constant",
                    constant_values=(0, 0),
                )
                new_mains = np.array(
                    [new_mains[i : i + n] for i in range(len(new_mains) - n + 1)]
                )
                new_mains = (new_mains - mains_mean) / mains_std
                self.mains_mean.append(mains_mean)
                self.mains_std.append(mains_std)
                processed_mains_lst.append(pd.DataFrame(new_mains))
            appliance_list = []
            for (app_name, app_df_lst) in submeters_lst:

                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]["mean"]
                    app_std = self.appliance_params[app_name]["std"]
                else:
                    print("Parameters for ", app_name, " were not found!")
                    raise ApplianceNotFoundError()

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(
                        new_app_readings,
                        (units_to_pad, units_to_pad),
                        "constant",
                        constant_values=(0, 0),
                    )
                    new_app_readings = np.array(
                        [
                            new_app_readings[i : i + n]
                            for i in range(len(new_app_readings) - n + 1)
                        ]
                    )
                    new_app_readings = (
                        new_app_readings - app_mean
                    ) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))

                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for i, mains in enumerate(mains_lst):
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                # new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array(
                    [new_mains[i : i + n] for i in range(len(new_mains) - n + 1)]
                )
                new_mains = (new_mains - self.mains_mean[i]) / self.mains_std[i]
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self, train_appliances):

        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {"mean": app_mean, "std": app_std}})
