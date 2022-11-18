from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D , AveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf
import typing
import os 

tf.config.run_functions_eagerly(True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

@tf.function
def make_conv_block(X:tf.Tensor,activation:str = 'relu',kernel_size:int = 3 ,num_filters:int = 32)->tf.Tensor:
    """
    

    Parameters
    ----------
    X : tf.Tensor
        Input Tensor.
    activation : str, optional
        DESCRIPTION. The default is 'relu'.
    kernel_size : int, optional
        Kernel size of conv layer. The default is 3.

    Returns
    -------
    X : tf>tensor
        output Tensor.

    """
    X = Conv2D(kernel_size = kernel_size, padding = 'same' ,strides = (1,1),filters = num_filters)(X)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)
    X = MaxPooling2D(pool_size = (2,2),strides = (2,2))(X)
    return X


@tf.function
def make_dense_block(X:tf.Tensor , output_units:int ,dropout:float = 0.5 , activation:str = 'relu') -> tf.Tensor:
    """
    

    Parameters
    ----------
    X : tf.Tensor
        input tensor.
    output_units : int
        output units in fcl.
    dropout : float, optional
        dropout rate. The default is 0.5.
    activation : str, optional
        DESCRIPTION. The default is 'relu'.

    Returns
    -------
    X : tf.Tensor
        output.

    """
    X = Dense(units = output_units)(X)
    X = BatchNormalization()(X)
    X = Activation(activation)(X)
    X = Dropout(dropout)(X)
    return X


@tf.function
def build_model(input_shape:typing.Tuple[int] = (128,128,3) , num_classes:int = 4) ->Model:
    """
    

    Parameters
    ----------
    input_size : typing.Tuple, optional
        DESCRIPTION. The default is 128.
    num_classes : int, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    Model
        DESCRIPTION.

    """
    x_input = Input(input_shape)
    x =make_conv_block(X = x_input , num_filters  = 32)
    x =make_conv_block(X = x , num_filters = 64)
    x =make_conv_block(X = x , num_filters = 128)
    x =make_conv_block(X = x , num_filters = 256)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = make_dense_block(X = x , output_units = 256)
    x = make_dense_block(X = x , output_units = 128)
    x = make_dense_block(X = x , output_units = 64)
    x = Dense(units = num_classes)(x)
    
    model = Model(inputs = x_input , outputs = x)
    
    return model
    

   # return model

if __name__ == '__main__':
    input_shape = (128,128,3)
    model = build_model(input_shape = input_shape)
    model.summary()
