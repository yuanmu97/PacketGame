import tensorflow as tf
from tensorflow.keras import layers


def build_conv1d_twoview(inp_len1=5, inp_len2=5, conv_units=[32, 32]):
    """
    build two-view neural network 

    Args:
        inp_len1, inp_len2 (int): input length of two views, 5 by default
        conv_units (list of int): number of conv1d units
            by default, two conv1d layers with 32 units
    Return:
        tf.keras.Model instance
    """
    inp1 = layers.Input(shape=(None, inp_len1), name="View1-Indepdendent")
    inp2 = layers.Input(shape=(None, inp_len2), name="View2-Predicted")
    x1 = inp1
    x2 = inp2 
    for u in conv_units:
        x1 = layers.Conv1D(u, 1, activation="relu")(x1)
        x2 = layers.Conv1D(u, 1, activation="relu")(x2)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x = layers.Concatenate()([x1, x2])
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=[inp1, inp2], outputs=out)


def build_ensemble_threeview(inp_len1=5, inp_len2=5, inp_len3=1, conv_units=[32, 32], dense_unit=128):
    """
    build three-view neural network 

    Args:
        inp_len1, inp_len2, inp_len3 (int): input length of three views, 5 for inp1 & inp2 and 1 for inp3 by default
        conv_units (list of int): number of conv1d units
            by default, two conv1d layers with 32 units
        dense_unit (int): number of dense units, 128 by default
    Return:
        tf.keras.Model instance
    """
    inp1 = layers.Input(shape=(None, inp_len1), name="View1-Indepdendent")
    inp2 = layers.Input(shape=(None, inp_len2), name="View2-Predicted")
    inp3 = layers.Input(shape=(inp_len3), name="View3-Temporal")
    x1 = inp1
    x2 = inp2 
    x3 = inp3
    for u in conv_units:
        x1 = layers.Conv1D(u, 1, activation="relu")(x1)
        x2 = layers.Conv1D(u, 1, activation="relu")(x2)
    x1 = layers.GlobalMaxPooling1D()(x1)
    x2 = layers.GlobalMaxPooling1D()(x2)
    x = layers.Concatenate()([x1, x2])
    out = layers.Dense(1, activation="sigmoid")(x)
    x4 = layers.Concatenate()([out, x3])
    x4 = layers.Dense(dense_unit, activation="relu")(x4)
    out2 = layers.Dense(1, activation='sigmoid')(x4)
    return tf.keras.Model(inputs=[inp1, inp2, inp3], outputs=out2)