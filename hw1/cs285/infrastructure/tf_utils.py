import tensorflow as tf


def build_mlp(input_shape, output_size, n_layers, size, activation=tf.keras.activations.tanh, output_activation=None):
    """
        Builds a feedforward neural network

        arguments:
            input_shape: input tensor shape

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            Multi-level perceptron model.
    """
    if n_layers > 0:
        layers = [tf.keras.layers.Dense(size, activation=activation, input_shape=input_shape, name='dense0')]
        layers.extend(tf.keras.layers.Dense(size, activation=activation, name=f'dense{i}') for i in range(1, n_layers))
        layers.append(tf.keras.layers.Dense(output_size, activation=output_activation, name='final'))
    else:
        layers = [tf.keras.layers.Dense(output_size, activation=output_activation, input_shape=input_shape, name='final')]
    return tf.keras.Sequential(layers, name='mlp')


def configure_tf_devices(use_gpu, gpu_memory_limit=1024, allow_gpu_growth=True, which_gpu=0):
    if use_gpu:
        all_gpus = tf.config.experimental.list_physical_devices('GPU')
        active_gpu = all_gpus[which_gpu]
        tf.config.experimental.set_visible_devices([active_gpu], 'GPU')
        if allow_gpu_growth:
            tf.config.experimental.set_memory_growth(active_gpu, True)
        else:
            tf.config.experimental.set_virtual_device_configuration(
                active_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])
        tf.config.set_soft_device_placement(True)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.debugging.set_log_device_placement(False)
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
