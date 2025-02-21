# [24.175098419189453, 0.9136666655540466]  cnnae_cbam_2019_v3.h5
# [17.06353187561035, 0.9056487083435059]  cnnae_cbam_2020_v3.h5
#[17.84626007080078, 0.9061633944511414] cnnae_cbam_2020_v4.h5-有多个residual connection
# [31.188888549804688, 0.8532350659370422] cnnae_cbam_2021_v4.h5
from __future__ import print_function
# import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten,Add,Dropout,Concatenate,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
# from tensorflow.keras import backend as K
from keras import backend as K
from tensorflow.keras.models import Model
import attention_module
import importlib
importlib.reload(attention_module)
from attention_module import attach_attention_module


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 layername='encoder'):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-2),
                  name=layername)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_ae(input_shape, num_classes=1, attention_module=None):
    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    
    x = resnet_layer(inputs=inputs,
                     num_filters=64,
                     conv_first=True)
    x = attach_attention_module(x, attention_module)
    avg_pool= AveragePooling2D(pool_size=7)(x)
    max_pool = MaxPooling2D(pool_size=7)(x)
    x = Add()([avg_pool,max_pool])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    
    other_input = Input(shape=(input_shape[-1],))
    other_input_encoded = Dense(128, name='denseinput')(other_input)
    other_input_encoded = BatchNormalization()(other_input_encoded)
    other_input_encoded = Activation('relu')(other_input_encoded)
    y = Concatenate()([x, other_input])
    
    dense0 = Dense(256, name='encoder0')(y)#,kernel_regularizer=regularizers.l2(0.01)
    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(128, name='encoder1')(dense0)
    #dense1 = Dropout(0.01)(dense1) 
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    
    dense2 = Dense(64, name='encoder2')(dense1)
    #dense2 = Dropout(0.01)(dense2) 
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    
    dense3 = Dense(32, name='encoder3')(dense2)
    #dense3 = Dropout(0.01)(dense3) 
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)
    
    dense4 = Dense(16,name='encoder4')(dense3)
    #dense4 = Dropout(0.01)(dense4) 
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4)
    
    dense5 = Dense(32, name='decoder1')(dense4)
    #dense3 = Dropout(0.01)(dense3) 
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)
    
    shortcut1=dense3
    dense5 = Add()([dense5, shortcut1])
        
    dense6 = Dense(64,name='decoder2')(dense5)
    #dense6 = Dropout(0.01)(dense6) 
    dense6 = BatchNormalization()(dense6)
    dense6= Activation('relu')(dense6)
    
    shortcut2=dense2
    dense6 = Add()([dense6, shortcut2])
    
            
    dense7 = Dense(128, name='decoder3')(dense6)
    #dense7 = Dropout(0.01)(dense7) 
    dense7 = BatchNormalization()(dense7)
    dense7 = Activation('relu')(dense7)
    
    shortcut3=dense1
    dense7 = Add()([dense7, shortcut3])
    
    # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
    # #dense8 = Dropout(0.01)(dense8) 
    # dense8 = BatchNormalization()(dense8)
    dense8 = Dense(256, name='decoder4')(dense7)
    dense8 = BatchNormalization()(dense8)
    dense8 = Activation('relu')(dense8)
    
    shortcut4=dense0
    dense8 = Add()([dense8, shortcut4])
    
    # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
    # #dense8 = Dropout(0.01)(dense8) 
    # dense8 = BatchNormalization()(dense8)

    outputs= Dense(num_classes)(dense8)
    # outputs = Dense(num_classes)(y)
    # Instantiate model.
    model = Model(inputs=[inputs, other_input], outputs=outputs)
    return model

# [24.954687118530273, 0.9110957384109497]  cnnae_cbam_2019_v2.h5
# from __future__ import print_function
# import keras
# from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from tensorflow.keras.layers import AveragePooling2D, Input, Flatten,Add,Dropout,Concatenate,MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2
# from keras import backend as K
# from tensorflow.keras.models import Model
# import attention_module
# import importlib
# importlib.reload(attention_module)
# from attention_module import attach_attention_module


# def resnet_layer(inputs,
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True,
#                  layername='encoder'):
#     """2D Convolution-Batch Normalization-Activation stack builder

#     # Arguments
#         inputs (tensor): input tensor from input image or previous layer
#         num_filters (int): Conv2D number of filters
#         kernel_size (int): Conv2D square kernel dimensions
#         strides (int): Conv2D square stride dimensions
#         activation (string): activation name
#         batch_normalization (bool): whether to include batch normalization
#         conv_first (bool): conv-bn-activation (True) or
#             bn-activation-conv (False)

#     # Returns
#         x (tensor): tensor as input to the next layer
#     """
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-2),
#                   name=layername)

#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)
#     return x


# def resnet_ae(input_shape, num_classes=1, attention_module=None):
#     inputs = Input(shape=input_shape)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     # x = attach_attention_module(inputs, attention_module)
#     x = resnet_layer(inputs=inputs,
#                      num_filters=64,
#                      conv_first=True)
#     avg_pool= AveragePooling2D(pool_size=7)(x)
#     max_pool = MaxPooling2D(pool_size=7)(x)
#     x = Add()([avg_pool,max_pool])
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Flatten()(x)
    
#     other_input = Input(shape=(42,))
#     other_input_encoded = Dense(128, name='denseinput')(other_input)
#     other_input_encoded = BatchNormalization()(other_input_encoded)
#     other_input_encoded = Activation('relu')(other_input_encoded)
#     y = Concatenate()([x, other_input])
    
#     dense0 = Dense(256, name='encoder0')(y)#,kernel_regularizer=regularizers.l2(0.01)
#     dense0 = BatchNormalization()(dense0)
#     dense0 = Activation('relu')(dense0)
#     dense1 = Dense(128, name='encoder1')(dense0)
#     #dense1 = Dropout(0.01)(dense1) 
#     dense1 = BatchNormalization()(dense1)
#     dense1 = Activation('relu')(dense1)
    
#     dense2 = Dense(64, name='encoder2')(dense1)
#     #dense2 = Dropout(0.01)(dense2) 
#     dense2 = BatchNormalization()(dense2)
#     dense2 = Activation('relu')(dense2)
    
    
#     dense3 = Dense(32, name='encoder3')(dense2)
#     #dense3 = Dropout(0.01)(dense3) 
#     dense3 = BatchNormalization()(dense3)
#     dense3 = Activation('relu')(dense3)
    
#     dense4 = Dense(16,name='encoder4')(dense3)
#     #dense4 = Dropout(0.01)(dense4) 
#     dense4 = BatchNormalization()(dense4)
#     dense4 = Activation('relu')(dense4)
    
#     dense5 = Dense(32, name='decoder1')(dense4)
#     #dense3 = Dropout(0.01)(dense3) 
#     dense5 = BatchNormalization()(dense5)
#     dense5 = Activation('relu')(dense5)
    
#     # shortcut1=dense3
#     # dense5 = Add()([dense5, shortcut1])
        
#     dense6 = Dense(64,name='decoder2')(dense5)
#     #dense6 = Dropout(0.01)(dense6) 
#     dense6 = BatchNormalization()(dense6)
#     dense6= Activation('relu')(dense6)
    
#     # shortcut2=dense2
#     # dense6 = Add()([dense6, shortcut2])
    
            
#     dense7 = Dense(128, name='decoder3')(dense6)
#     #dense7 = Dropout(0.01)(dense7) 
#     dense7 = BatchNormalization()(dense7)
#     dense7 = Activation('relu')(dense7)
    
#     # shortcut3=dense1
#     # dense7 = Add()([dense7, shortcut3])
    
#     # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
#     # #dense8 = Dropout(0.01)(dense8) 
#     # dense8 = BatchNormalization()(dense8)
#     dense8 = Dense(256, name='decoder4')(dense7)
#     dense8 = BatchNormalization()(dense8)
#     dense8 = Activation('relu')(dense8)
    
#     shortcut4=dense0
#     dense8 = Add()([dense8, shortcut4])
    
#     # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
#     # #dense8 = Dropout(0.01)(dense8) 
#     # dense8 = BatchNormalization()(dense8)

#     outputs= Dense(num_classes)(dense8)
#     # outputs = Dense(num_classes)(y)
#     # Instantiate model.
#     model = Model(inputs=[inputs, other_input], outputs=outputs)
#     return model
    

## 备份-最好的一次 cnnae_cbam_2019_v1.h5   23  0.91
# from __future__ import print_function
# import keras
# from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from tensorflow.keras.layers import AveragePooling2D, Input, Flatten,Add,Dropout,Concatenate,MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2
# from keras import backend as K
# from tensorflow.keras.models import Model
# import attention_module
# import importlib
# importlib.reload(attention_module)
# from attention_module import attach_attention_module


# def resnet_layer(inputs,
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True,
#                  layername='encoder'):
#     """2D Convolution-Batch Normalization-Activation stack builder

#     # Arguments
#         inputs (tensor): input tensor from input image or previous layer
#         num_filters (int): Conv2D number of filters
#         kernel_size (int): Conv2D square kernel dimensions
#         strides (int): Conv2D square stride dimensions
#         activation (string): activation name
#         batch_normalization (bool): whether to include batch normalization
#         conv_first (bool): conv-bn-activation (True) or
#             bn-activation-conv (False)

#     # Returns
#         x (tensor): tensor as input to the next layer
#     """
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-2),
#                   name=layername)

#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)
#     return x


# def resnet_ae(input_shape, num_classes=1, attention_module=None):
#     inputs = Input(shape=input_shape)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     # x = attach_attention_module(inputs, attention_module)
#     # x = resnet_layer(inputs=x,
#     #                  num_filters=64,
#     #                  conv_first=True)
#     avg_pool= AveragePooling2D(pool_size=7)(inputs)
#     max_pool = MaxPooling2D(pool_size=7)(inputs)
#     x = Add()([avg_pool,max_pool])
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Flatten()(x)
    
#     other_input = Input(shape=(42,))
#     other_input_encoded = Dense(128, name='denseinput')(other_input)
#     other_input_encoded = BatchNormalization()(other_input_encoded)
#     other_input_encoded = Activation('relu')(other_input_encoded)
#     y = Concatenate()([x, other_input])
    
#     dense0 = Dense(256, name='encoder0')(y)#,kernel_regularizer=regularizers.l2(0.01)
#     dense0 = BatchNormalization()(dense0)
#     dense0 = Activation('relu')(dense0)
#     dense1 = Dense(128, name='encoder1')(dense0)
#     #dense1 = Dropout(0.01)(dense1) 
#     dense1 = BatchNormalization()(dense1)
#     dense1 = Activation('relu')(dense1)
    
#     dense2 = Dense(64, name='encoder2')(dense1)
#     #dense2 = Dropout(0.01)(dense2) 
#     dense2 = BatchNormalization()(dense2)
#     dense2 = Activation('relu')(dense2)
    
    
#     dense3 = Dense(32, name='encoder3')(dense2)
#     #dense3 = Dropout(0.01)(dense3) 
#     dense3 = BatchNormalization()(dense3)
#     dense3 = Activation('relu')(dense3)
    
#     dense4 = Dense(16,name='encoder4')(dense3)
#     #dense4 = Dropout(0.01)(dense4) 
#     dense4 = BatchNormalization()(dense4)
#     dense4 = Activation('relu')(dense4)
    
#     dense5 = Dense(32, name='decoder1')(dense4)
#     #dense3 = Dropout(0.01)(dense3) 
#     dense5 = BatchNormalization()(dense5)
#     dense5 = Activation('relu')(dense5)
    
#     # shortcut1=dense3
#     # dense5 = Add()([dense5, shortcut1])
        
#     dense6 = Dense(64,name='decoder2')(dense5)
#     #dense6 = Dropout(0.01)(dense6) 
#     dense6 = BatchNormalization()(dense6)
#     dense6= Activation('relu')(dense6)
    
#     # shortcut2=dense2
#     # dense6 = Add()([dense6, shortcut2])
    
            
#     dense7 = Dense(128, name='decoder3')(dense6)
#     #dense7 = Dropout(0.01)(dense7) 
#     dense7 = BatchNormalization()(dense7)
#     dense7 = Activation('relu')(dense7)
    
#     # shortcut3=dense1
#     # dense7 = Add()([dense7, shortcut3])
    
#     # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
#     # #dense8 = Dropout(0.01)(dense8) 
#     # dense8 = BatchNormalization()(dense8)
#     dense8 = Dense(256, name='decoder4')(dense7)
#     dense8 = BatchNormalization()(dense8)
#     dense8 = Activation('relu')(dense8)
    
#     shortcut4=dense0
#     dense8 = Add()([dense8, shortcut4])
    
#     # dense8 = Dense(256, activation='relu',name='decoder4')(dense7)
#     # #dense8 = Dropout(0.01)(dense8) 
#     # dense8 = BatchNormalization()(dense8)

#     outputs= Dense(num_classes)(dense8)
#     # outputs = Dense(num_classes)(y)
#     # Instantiate model.
#     model = Model(inputs=[inputs, other_input], outputs=outputs)
#     return model
    

# #         #dense1 = Dropout(0.01)(dense1) 
# #         other_input = BatchNormalization()(other_input)
# #         other_input = Activation('relu')(other_input)
        
# #         other_input = Dense(64, name='encoder2')(other_input)
# #         #dense1 = Dropout(0.01)(dense1) 
# #         other_input = BatchNormalization()(other_input)
# #         other_input = Activation('relu')(other_input)
#         # 将最后一层输出和其他矩阵进行相加
        
#         # y = Concatenate()([y, other_input])
#         # y = BatchNormalization()(y)
#         # y = Activation('relu')(y)
# # def resnet_ae(input_shape, num_classes=1, attention_module=None):
# #     # num_filters = 16
# #     inputs = Input(shape=input_shape)
# #     dense0 = resnet_layer(inputs=inputs,
# #                      num_filters=256,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder0')
# #     dense1 = resnet_layer(inputs=dense0,
# #                      num_filters=128,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder1')
# #     dense2 = resnet_layer(inputs=dense1,
# #                      num_filters=64,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder2')
# #     dense3 = resnet_layer(inputs=dense2,
# #                      num_filters=32,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder3')
# #     dense4 = resnet_layer(inputs=dense3,
# #                      num_filters=16,
# #                      kernel_size=1,
# #                      strides=1,layername='encoder4')
# #     dense5 = resnet_layer(inputs=dense4,
# #                      num_filters=32,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder5')
# #     dense6 = resnet_layer(inputs=dense5,
# #                      num_filters=64,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder6')
# #     dense7 = resnet_layer(inputs=dense6,
# #                      num_filters=128,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder7')
# #     dense8 = resnet_layer(inputs=dense7,
# #                      num_filters=256,
# #                      kernel_size=3,
# #                      strides=1,layername='encoder8')
    
# #     shortcut4=dense0  
# #     if attention_module is not None:
# #         dense8 = attach_attention_module(dense8, attention_module)
# #         print(f'attention dims y.shape {shortcut4.shape}')
    
    
# #     dense8 = Add()([dense8, shortcut4])
    
# #     y = BatchNormalization()(dense8)
# #     y = Activation('relu')(y)
# #     y= AveragePooling2D(pool_size=7)(y)
# #     y = Flatten()(y)
    
# #     outputs = Dense(num_classes)(y)
# #     # Instantiate model.
# #     model = Model(inputs=inputs, outputs=outputs)
# #     return model
    
    