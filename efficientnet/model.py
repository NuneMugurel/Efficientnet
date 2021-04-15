"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import tensorflow as tf
from keras import layers
from keras.models import Model
import math
import collections

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

blocks_args = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

params_dict = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    'efficientnet-experimental-width': (1.19, 1.0, 224, 0.2),
    'efficientnet-experimental-resolution': (1.0, 1.0, 267, 0.2),
    'efficientnet-experimental-depth': (1.0, 1.8, 224, 0.4)
}

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def mb_conv_block(inputs, block_args, activation, dropout_rate=None, prefix=''):
    """Mobile Inverted Residual Bottleneck."""
    # https://paperswithcode.com/method/inverted-residual-block
    # 

    num_filters = block_args.input_filters * block_args.expand_ratio

    # region Expansion phase
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(num_filters, 
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)
    # endregion

    # region Squeeze and Excitation phase
    # GlobalAverage + 2xDense + multiply with original
    target_shape = (1, 1, num_filters)

    residual = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
    residual = layers.Reshape(target_shape, name=prefix + 'se_reshape')(residual)

    se_reduced_filters = int(math.ceil(block_args.se_ratio * block_args.input_filters))
    residual = layers.Conv2D(se_reduced_filters,
                             kernel_size=1,
                             activation=activation,
                             padding='same',
                             use_bias=True,
                             name=prefix + 'se_reduce')(residual)
    residual = layers.Conv2D(num_filters,
                             kernel_size=1,
                             activation='sigmoid',
                             padding='same',
                             use_bias=True,
                             name=prefix + 'se_expand')(residual)
                
    x = layers.multiply([x, residual], name=prefix + 'se_excite')
    # endregion

    # region Output phase
    x = layers.Conv2D(block_args.output_filters, 
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project_conv')(inputs)
    x = layers.BatchNormalization(name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.add([x, inputs], name=prefix + 'add')
    # endregion

    return x

def EfficientNet(model_name,
                 dropout_connection_rate=0.2,
                 depth_divisor=8,
                 load_weights=False,
                 num_classes=23):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on part of ImageNet.
    # Arguments
        model_name: name of the model, efficientnet-b<0-8 or l2>
        dropout_connection_rate: float between 0 and 1, dropout rate at mb_conv connections
        depth_divisor: int
        load_weights: whether to load or not the weights of the previously trained model
        num_classes: number of classes from the ImageNet dataset
    # Returns
        A Keras model
    """
    
    model_params = params_dict[model_name]
    width_coefficient = model_params[0]
    depth_coefficient = model_params[1]
    resolution = model_params[2]
    dropout_rate = model_params[3] # dropout of the classifier

    activation = tf.keras.activations.swish

    # region Build bottom
    input_tensor = layers.Input(shape=(resolution, resolution, 3), dtype='float32', name='input_tensor')

    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor),
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias='false',
                      name='stem_conv')(input_tensor)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    #endregion

    # region Build blocks
    for idx, block_arg in enumerate(blocks_args):
        block_arg = block_arg._replace(
            input_filters = round_filters(block_arg.input_filters, width_coefficient, depth_divisor),
            output_filters = round_filters(block_arg.output_filters, width_coefficient, depth_divisor),
            num_repeat = round_repeats(block_arg.num_repeat, depth_coefficient)
        )

        dropout_connection_rate_current = dropout_connection_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x,
                          block_arg,
                          activation=activation,
                          dropout_rate=dropout_connection_rate_current,
                          prefix=f'block{idx + 1}a_'
                          )
        block_num += 1

        if block_arg.num_repeat > 1:
            block_arg = block_arg._replace(input_filters=block_arg.output_filters, strides=[1, 1])
            for idx2 in range(block_arg.num_repeat - 1):
                dropout_rate_blocks = dropout_connection_rate * float(block_num) / num_blocks_total
                print(dropout_rate_blocks)
                x = mb_conv_block(x,
                                  block_arg,
                                  activation=activation,
                                  dropout_rate=dropout_rate_blocks,
                                  prefix=f'block{idx + 1}{idx2 + 1}_'
                                  )
                block_num += 1
    # endregion

    # region Build top
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      name='top_conv')(x)
    x = layers.BatchNormalization(name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    
    x = layers.GlobalAveragePooling2D(name='top_avg_pool')(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = layers.Dense(num_classes,
                     activation='softmax',
                     name='probs')(x)
    # endregion

    model = Model(input_tensor, x, name=model_name)

    if load_weights:
        model.load_weights('./MnasEfficientNet.h5')

    return model

def EfficientNetB0(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b0',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB1(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b1',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB2(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b2',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB3(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b3',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB4(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b4',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB5(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b5',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB6(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b6',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB7(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b7',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetB8(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-b8',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetL2(load_weights=False,
                   num_classes=23):
    return EfficientNet(model_name='efficientnet-l2',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetExperimentalWidth(load_weights=False,
                             num_classes=23):
    return EfficientNet(model_name='efficientnet-experimental-width',
                        load_weights=load_weights,
                        num_classes=num_classes)

def EfficientNetExperimentalResolution(load_weights=False,
                             num_classes=23):
    return EfficientNet(model_name='efficientnet-experimental-resolution',
                        load_weights=load_weights,
                        num_classes=num_classes)  

def EfficientNetExperimentalDepth(load_weights=False,
                             num_classes=23):
    return EfficientNet(model_name='efficientnet-experimental-depth',
                        load_weights=load_weights,
                        num_classes=num_classes)    







    
