from __future__ import print_function, division

from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
)

import pandas as pd
import numpy as np
from collections import OrderedDict
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Layer,
    LayerNormalization,
    Embedding,
    Conv1DTranspose,
)

from tensorflow_addons.layers import MultiHeadAttention

from tensorflow.keras.losses import KLDivergence, MeanSquaredError, MeanAbsoluteError

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
        self.att = MultiHeadAttention(num_heads=num_heads, head_size=embed_dim)
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
        attn_output = self.att([inputs, inputs])
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


class PositionEmbedding(Layer):
    def __init__(self, maxlen, hidden_size):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = Embedding(input_dim=maxlen + 1, output_dim=hidden_size)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
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


class BERT4NILM(Disaggregator):
    def __init__(self, params):

        self.MODEL_NAME = "BERT4NILM"
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 481)
        self.n_epochs = params.get("n_epochs", 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise (SequenceLengthError)

        self.cutoffs = {
            "aggregate": 6000.0,
            "fridge": 400.0,
            "washer_dryer": 3500.0,
            "microwave": 1800.0,
            "dishwasher": 1200.0,
            "television": 2000.0,
        }

        self.threshold = {
            "fridge": 50.0,
            "washer_dryer": 20.0,
            "microwave": 200.0,
            "dishwasher": 10.0,
            "television": 50.0,
        }

        self.min_on = {
            "fridge": 10,
            "washer_dryer": 300,
            "microwave": 2,
            "dishwasher": 300,
            "television": 1,
        }

        self.min_off = {
            "fridge": 2,
            "washer_dryer": 26,
            "microwave": 5,
            "dishwasher": 300,
            "television": 1,
        }

        self.c0 = {
            "fridge": 1e-6,
            "washer_dryer": 0.001,
            "microwave": 1.0,
            "dishwasher": 1.0,
            "television": 1.0,
        }

    def partial_fit(
        self,
        train_main,
        train_appliances,
        do_preprocessing=True,
        current_epoch=0,
        **load_kwargs
    ):

        print("...............BERT partial_fit running...............")
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
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network(
                    self.cutoffs[appliance_name],
                    self.threshold[appliance_name],
                    self.c0[appliance_name],
                )
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                        "_".join(appliance_name.split()),
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
                        train_main, power, test_size=0.15, random_state=10
                    )
                    model.fit(
                        train_x,
                        train_y,
                        validation_data=(v_x, v_y),
                        epochs=self.n_epochs,
                        callbacks=[checkpoint],
                        batch_size=self.batch_size,
                    )
                    model.load_weights(filepath)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):

        if model is not None:
            self.models = model

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

            for appliance in self.models:

                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array, batch_size=self.batch_size)

                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured

                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i : i + l] += prediction[i].flatten()
                    counts_arr[i : i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                #################
                prediction = self.appliance_params[appliance]["mean"] + (
                    sum_arr * self.appliance_params[appliance]["std"]
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

    def return_network(self, cutoff, threshold, c0):
        """Creates the BERT module"""
        hidden_size = 256  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = (
            4 * hidden_size
        )  # Hidden layer size in feed forward network inside transformer
        maxlen = self.sequence_length  # maxlength for attention
        dropout_rate = 0.15
        num_layers = 2

        model = Sequential()
        model.add(
            Conv1D(
                filters=hidden_size,
                kernel_size=5,
                activation="linear",
                input_shape=(self.sequence_length, 1),
                padding="same",
                strides=1,
            )
        )
        model.add(LPpool(pool_size=2))

        # Token and Positional embedding and Encoder part of the transformer
        model.add(PositionEmbedding(int(maxlen / 2), hidden_size))
        model.add(LayerNormalization())
        model.add(Dropout(dropout_rate))

        # Transformer layers
        for _ in range(num_layers):
            model.add(
                TransformerBlock(hidden_size, num_heads, ff_dim, rate=dropout_rate)
            )

        # Conv1dTranspose
        model.add(
            Conv1DTranspose(
                hidden_size, kernel_size=4, strides=2, output_padding=1, padding="same"
            )
        )

        # Fully connected layers
        model.add(Dense(128, activation="tanh"))
        model.add(Dense(1))

        model.summary()
        model.compile(
            loss=self.bert_loss(cutoff, threshold, c0),
            optimizer="adam",
            metrics=["mse", "mae"],
        )
        return model

    def bert_loss(self, cutoff, threshold, c0):
        def loss(y_true, y_pred):
            labels = y_true / cutoff
            status = BERT4NILM.compute_status(y_true, threshold)
            logits_energy = BERT4NILM.cutoff_energy(y_pred * cutoff, cutoff)
            logits_status = BERT4NILM.compute_status(logits_energy, threshold)

            mask = status >= 0
            labels_masked = tf.reshape(
                tf.boolean_mask(labels, mask), [-1, y_true.shape[-1]]
            )
            logits_masked = tf.reshape(
                tf.boolean_mask(y_pred, mask), [-1, y_true.shape[-1]]
            )
            status_masked = tf.reshape(
                tf.boolean_mask(status, mask), [-1, y_true.shape[-1]]
            )
            logits_status_masked = tf.reshape(
                tf.boolean_mask(logits_status, mask), [-1, y_true.shape[-1]]
            )

            kl = KLDivergence()
            mse = MeanSquaredError()
            l1_on = MeanAbsoluteError()

            kl_loss = kl(
                tf.math.log(
                    tf.nn.softmax(tf.squeeze(logits_masked) / 0.1, axis=-1) + 1e-9
                ),
                tf.nn.softmax(tf.squeeze(labels_masked) / 0.1, axis=-1),
            )
            mse_loss = mse(
                tf.cast(logits_masked, dtype=tf.float32),
                tf.cast(labels_masked, dtype=tf.float32),
            )
            margin_loss = BERT4NILM.soft_margin_loss(
                tf.cast((logits_status_masked * 2 - 1), dtype=tf.float32),
                tf.cast((status_masked * 2 - 1), dtype=tf.float32),
            )
            total_loss = kl_loss + mse_loss + margin_loss

            on_mask = tf.cast((status >= 0), dtype=tf.int32) * tf.cast(
                (
                    (
                        tf.cast((status == 1), dtype=tf.int32)
                        + tf.cast(
                            (
                                status
                                != tf.reshape(logits_status, [-1, status.shape[-1]])
                            ),
                            dtype=tf.int32,
                        )
                    )
                    >= 1
                ),
                dtype=tf.int32,
            )
            if tf.reduce_sum(on_mask) > 0:
                total_size = tf.cast(
                    tf.reduce_prod(tf.constant([self.batch_size, on_mask.shape[-1]])),
                    dtype=tf.float32,
                )
                logits_on = tf.boolean_mask(
                    tf.reshape(y_pred, [-1, on_mask.shape[-1]]), on_mask
                )
                labels_on = tf.boolean_mask(
                    tf.reshape(labels, [-1, on_mask.shape[-1]]), on_mask
                )
                l1_loss_on = l1_on(
                    tf.reshape(logits_on, [-1]), tf.reshape(labels_on, [-1])
                )
                total_loss = total_loss + c0 * l1_loss_on / total_size

            return total_loss

        return loss

    @staticmethod
    def cutoff_energy(data, cutoff):
        cutoff = tf.constant([cutoff for _ in range(data.shape[1])])
        data = tf.where(data < 5.0, 0.0, data)
        data = tf.math.minimum(data, cutoff)
        return data

    @staticmethod
    def compute_status(data, threshold):
        threshold = tf.constant([threshold for _ in range(data.shape[1])])
        status = tf.cast((data >= threshold), dtype=tf.int32) * 1
        return status

    @staticmethod
    def soft_margin_loss(x, y):
        top = tf.math.log(tf.math.exp(-x * y) + 1.0)
        return tf.reduce_sum(top) / tf.cast(tf.size(x), dtype=tf.float32)

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == "train":
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
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
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):

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
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                # new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array(
                    [new_mains[i : i + n] for i in range(len(new_mains) - n + 1)]
                )
                new_mains = (new_mains - self.mains_mean) / self.mains_std
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
