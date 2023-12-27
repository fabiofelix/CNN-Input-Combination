
import os, glob, keras.backend as K
from http.client import NON_AUTHORITATIVE_INFORMATION
from keras import optimizers, layers
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.regularizers import l2
from enum import IntEnum

import quantification

class Combine_Type(IntEnum):
  NONE     = 0
  BN       = 1
  DENSE    = 2
  BN_DENSE = 3
  THREE_INPUT          = 10
  THREE_INPUT_BN       = 11
  THREE_INPUT_DENSE    = 12
  THREE_INPUT_BN_DENSE = 13  
  THREE_CHANNEL          = 20
  THREE_CHANNEL_BN       = 21
  THREE_CHANNEL_DENSE    = 22
  THREE_CHANNEL_BN_DENSE = 23

HAND_CRAFTED_TYPES  = [ct.value for ct in Combine_Type if str(ct.value)[-1] != str(Combine_Type.NONE.value)]
BN_TYPES            = [ct.value for ct in Combine_Type if str(ct.value)[-1] == str(Combine_Type.BN.value)]
DENSE_TYPES         = [ct.value for ct in Combine_Type if str(ct.value)[-1] == str(Combine_Type.DENSE.value)]
BN_DENSE_TYPES      = [ct.value for ct in Combine_Type if str(ct.value)[-1] == str(Combine_Type.BN_DENSE.value)]
THREE_INPUT_TYPES   = [ct.value for ct in Combine_Type if ct.value >= Combine_Type.THREE_INPUT and ct.value < Combine_Type.THREE_CHANNEL]
THREE_CHANNEL_TYPES = [ct.value for ct in Combine_Type if ct.value >= Combine_Type.THREE_CHANNEL]     

class Model_type(IntEnum):
  CNN     = 0
  BIRDVOX = 1
  R50     = 2
  IV3     = 3

def my_CNN2D(dims, comb_type = Combine_Type.NONE):
  input_layer  = layers.Input(shape = dims)
  
  layer = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
  layer = layers.MaxPooling2D((2, 2), padding='same')(layer)
  layer = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layer)  
  layer = layers.MaxPooling2D((2, 2), padding='same')(layer)  
  layer = layers.Flatten()(layer)
  layer = layers.Dense(128, activation='relu')(layer)

  if comb_type == Combine_Type.NONE:
    layer = layers.Dense(128, activation='relu')(layer)  
  
  return input_layer, layer, "CNN", optimizers.SGD(lr=0.01, momentum = 0.9)

def Channel_AveragePooling2D(x):
  return K.mean(x, axis = -1)

def load_birdvox(dims, python_path = ""):
  python_path = "/home/fabio/Documents/Python/ambientes/birdvox3.6/lib/python3.6/site-packages"
  species_classifier = "birdvoxclassify/resources/models/birdvoxclassify-flat-multitask-convnet-v2_tv1hierarchical-2e7e1bbd434a35b3961e315cfe3832fc.h5" ##Cramer:2020

  birdvox = load_model(os.path.join(python_path, species_classifier), compile = False, custom_objects = None)

  input_layer = layers.Input(shape = dims)
  layer = input_layer

  if dims[-1] > 1:
    layer = layers.Lambda(Channel_AveragePooling2D)(layer)
    layer = layers.Reshape((dims[0], dims[1], 1))(layer)

  for i in range(1, len(birdvox.layers) - 4): #disconsider the input layers and the final dense layers
    layer = birdvox.layers[i](layer)

  return input_layer, layer

def my_BirdVox(dims, comb_type = Combine_Type.NONE):
  input_layer, layer = load_birdvox(dims)
  
  if comb_type == Combine_Type.NONE:
    layer = layers.Dense(64, activation = "relu", use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer = l2(0.001))(layer)
  else:
    layer = layers.Dense(128, activation = "relu", use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer = l2(0.001))(layer)

  return input_layer, layer, "BirdVox-Classify", optimizers.Adam(lr=0.0001)  

def my_ResNet50(dims, comb_type = Combine_Type.NONE):
  input_layer = None
  input_shape = dims

  if comb_type not in THREE_CHANNEL_TYPES:
  ##Combine a gray-scale image to generate a 3D tensor
    input_layer = layers.Input(shape = (dims[0], dims[1], 1))
    input_layer = layers.Concatenate()([input_layer, input_layer, input_layer])
    input_shape = None

  model = ResNet50(weights = "imagenet", include_top = False, input_tensor = input_layer, input_shape = input_shape)

  layer = layers.GlobalAveragePooling2D()(model.output)

  if comb_type in HAND_CRAFTED_TYPES:
    layer = layers.Dense(128, activation='relu')(layer)  

  return model.input, layer, "ResNet50", optimizers.Adam(lr=0.0001)

def my_InceptionV3(dims, comb_type = Combine_Type.NONE):
  input_layer = None
  input_shape = dims

  if comb_type not in THREE_CHANNEL_TYPES:
 ##Combine a gray-scale image to generate a 3D tensor
    input_layer = layers.Input(shape = (dims[0], dims[1], 1))
    input_layer = layers.Concatenate()([input_layer, input_layer, input_layer])
    input_shape = None

  model = InceptionV3(weights = "imagenet", include_top = False, input_tensor = input_layer, input_shape = input_shape)

  layer  = layers.GlobalAveragePooling2D()(model.output)

  if comb_type in HAND_CRAFTED_TYPES:
    layer = layers.Dense(128, activation='relu')(layer)
 
  return model.input, layer, "InceptionV3", optimizers.RMSprop(lr = 0.001)  

def add_handcrafted(input_cnn, output_cnn, comb_type):
  input_layers       = input_cnn
  input_handcrafted  = layers.Input(shape = (35, ))
  output_handcrafted = None

  if isinstance(input_layers, list):
    input_layers.append(input_handcrafted)
  else:
    input_layers = [input_cnn, input_handcrafted]

  if comb_type in BN_TYPES:
    output_handcrafted = layers.BatchNormalization(name = "bn_hc")(input_handcrafted)
  elif comb_type in DENSE_TYPES:  
    output_handcrafted = layers.Dense(128, activation='relu', name = "dense_hc")(input_handcrafted)
  elif comb_type in BN_DENSE_TYPES:    
    output_handcrafted = layers.BatchNormalization(name = "bn_hc")(input_handcrafted)
    output_handcrafted = layers.Dense(128, activation='relu', name = "dense_hc")(output_handcrafted)    

  concat_layers = output_cnn

  if isinstance(concat_layers, list):
    concat_layers.append(output_handcrafted)
  else:
    concat_layers = [concat_layers, output_handcrafted]

  concat       = layers.Concatenate()(concat_layers)
  output_layer = layers.Dense(128, activation='relu')(concat)  

  return input_layers, output_layer

def combine_3inputs(dims, inputs, outputs, comb_type):
  shared = Model(inputs, outputs)

  spec_input = layers.Input(shape = dims)  
  spec_model = shared(spec_input)

  mel_input  = layers.Input(shape = dims)  
  mel_model  = shared(mel_input)

  pcen_input = layers.Input(shape = dims)  
  pcen_model = shared(pcen_input)

  output_layer = [spec_model, mel_model, pcen_model]

  if comb_type == Combine_Type.THREE_INPUT:
    concat       = layers.Concatenate()([spec_model, mel_model, pcen_model]) 
    output_layer = layers.Dense(128, activation='relu')(concat)
  
  return [spec_input, mel_input, pcen_input], output_layer 

def build_model_name(name, comb_type, quant):
  if comb_type in THREE_INPUT_TYPES:
    name += "-3input"
  elif comb_type in THREE_CHANNEL_TYPES:
    name += "-3channel"

  if comb_type in BN_TYPES:
    name += "-hc_bn"
  elif comb_type in DENSE_TYPES:
    name += "-hc_dense"
  elif comb_type in BN_DENSE_TYPES:
    name += "-hc_bn+dense"

  return name + ("-CQ" if quant else "")

def build_model(model_type, dims, qt_labels, comb_type = Combine_Type.NONE, quant = None, show_summary = False):
  dims   = (dims[0], dims[1], 3 if comb_type in THREE_CHANNEL_TYPES else 1)
  inputs = outputs = model_name = model_opt = None

#==============================CREATING MODELS==============================#
  if model_type == Model_type.CNN:
    inputs, outputs, model_name, model_opt = my_CNN2D(dims, comb_type)
  elif model_type == Model_type.BIRDVOX:
    inputs, outputs, model_name, model_opt = my_BirdVox(dims, comb_type)
  elif model_type == Model_type.R50:
    inputs, outputs, model_name, model_opt = my_ResNet50(dims, comb_type)    
  elif model_type == Model_type.IV3:
    inputs, outputs, model_name, model_opt = my_InceptionV3(dims, comb_type)
  else:  
    raise Exception("Model [{}] is not defined.".format(model_type))

#==============================COMBINING INPUTS=============================#
  if comb_type in THREE_INPUT_TYPES:
    inputs, outputs = combine_3inputs(dims, inputs, outputs, comb_type)
  if comb_type in HAND_CRAFTED_TYPES:
    inputs, outputs = add_handcrafted(inputs, outputs, comb_type)
  ##THREE_CHANNELS is performed into each model function

#===========================================================================#
  outputs = layers.Dense(qt_labels, activation='softmax')(outputs)
  model   = Model(inputs, outputs, name = build_model_name(model_name, comb_type, quant))

  if quant is None:
    model.compile(loss='categorical_crossentropy', optimizer=model_opt, metrics=['acc'])    
  else:
    model.compile(loss = quantification.quant_loss(quant), optimizer=model_opt, metrics=['acc'])  

  if show_summary:
    model.summary()
  
  return model  

def my_load_model(weight_path, model_type, dims, qt_labels, comb_type = Combine_Type.NONE, quant = None, show_summary = False):
  model_name = glob.glob(os.path.join(weight_path, "*.hdf5"))

  if len(model_name) > 0:
    model_name = model_name[0]
  else:
    raise Exception("There is no hdf5 file with model in {}".format(weight_path))  

  model = None  
#==============================LOADING MODELS==============================#
  if model_type in [Model_type.CNN, Model_type.BIRDVOX]:
    model = load_model(model_name, compile = quant is None)

    if show_summary:
      model.summary()
  else:
    model = build_model(model_type, dims, qt_labels, comb_type, quant, show_summary)
    model.load_weights(model_name)

  return model  
