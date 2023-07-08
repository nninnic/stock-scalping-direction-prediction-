#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""


import Layers
import keras
from keras import backend as K


def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def BL(template, dropout=0.1, regularizer=None, constraint=None):
    """
    Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    regularizer: keras regularizer object
    constraint: keras constraint object
    
    outputs
    ------
    keras model object
    """
    print(template)
    inputs = keras.layers.Input(template[0])
    
    x = inputs
    for k in range(1, len(template)-1):
        x = Layers.BL(template[k], regularizer, constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    x = Layers.BL(template[-1], regularizer, constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    
    optimizer = keras.optimizers.Adam(0.01)
    
    model.compile(optimizer, 'categorical_crossentropy', ['acc'])
    
    return model

def TABL(template, dropout=0.1, projection_regularizer=None, projection_constraint=None,
         attention_regularizer=None, attention_constraint=None):
    """
    Temporal Attention augmented Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    projection_regularizer: keras regularizer object for projection matrices
    projection_constraint: keras constraint object for projection matrices
    attention_regularizer: keras regularizer object for attention matrices
    attention_constraint: keras constraint object for attention matrices
    
    outputs
    ------
    keras model object
    """
    
    inputs = keras.layers.Input(template[0])
    
    x = inputs
    for k in range(1, len(template)-1):
        x = Layers.BL(template[k], projection_regularizer, projection_constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    x = Layers.TABL(template[-1], projection_regularizer, projection_constraint,
                  attention_regularizer, attention_constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    
    optimizer = keras.optimizers.Adam(0.01, epsilon=1)
    
    # categorical_crossentropy sparse_categorical_crossentropy
    model.compile(optimizer, 'categorical_crossentropy', [
            keras.metrics.AUC(name='acc'),
            precision,
            recall,
            ]
        )
    
    return model


def BiN_TABL(template, dropout=0.1, 
        projection_regularizer=None, projection_constraint=None,attention_regularizer=None, attention_constraint=None,
        gamma1_regularizer=None,gamma1_constraint=None,gamma2_regularizer=None,gamma2_constraint=None):
    
    inputs = keras.layers.Input(template[0])
    x = inputs
    x = Layers.BiN(template[0],gamma1_regularizer,gamma1_constraint,gamma2_regularizer,gamma2_constraint)(x)
    for k in range(1, len(template)-1):
        x = Layers.BL(template[k], projection_regularizer, projection_constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)
    
    x = Layers.TABL(template[-1], projection_regularizer, projection_constraint,
                  attention_regularizer, attention_constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs = outputs)
    
    optimizer = keras.optimizers.Adam(lr=0.01,epsilon=1)
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            keras.metrics.AUC(name='acc'),
            precision,
            recall,
        ])
    
    return model
    

