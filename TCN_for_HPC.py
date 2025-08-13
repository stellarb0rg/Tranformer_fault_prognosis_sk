#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Temporal Convolutional Network (TCN) for Transformer Vibration Analysis
- Replaces RNNs with efficient dilated causal convolutions
- Incorporates spectral features and frequency domain analysis
- Optimized for both speed and accuracy
"""

import os
import time
import scipy.io as io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

layers = keras.layers
optimizers = keras.optimizers
callbacks = keras.callbacks

Model = keras.Model
Input = layers.Input
Dense = layers.Dense
Dropout = layers.Dropout
BatchNormalization = layers.BatchNormalization
Conv1D = layers.Conv1D
Add = layers.Add
Activation = layers.Activation
Lambda = layers.Lambda
Concatenate = layers.Concatenate
GlobalAveragePooling1D = layers.GlobalAveragePooling1D
MaxPooling1D = layers.MaxPooling1D
AveragePooling1D = layers.AveragePooling1D

Adam = optimizers.Adam
ReduceLROnPlateau = callbacks.ReduceLROnPlateau
EarlyStopping = callbacks.EarlyStopping
import math
import pickle

# Configure for GPU acceleration with memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Memory growth setting failed")

# Data parameters
file_ind = ['320','340','360','380','400','420','440']
Fs = 3000
st = 0.02  # stationary interval in seconds
L = int(st*Fs)  # block length
window = L
step = 1
delay = 0
batch_size = 32  # Larger batch size for faster convergence

# Function to convert numbers to readable string format
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

# Add FFT feature extraction function
def extract_spectral_features(x, axis=-1):
    # Compute FFT on the time dimension
    fft_features = tf.signal.rfft(x)
    # Get magnitude spectrum (absolute values of complex numbers)
    magnitudes = tf.abs(fft_features)
    return magnitudes

# Load and preprocess data
data_train_list = []
data_valid_list = []
data_test_list = []

print("Loading and preprocessing data...")
start_time = time.time()

for file in file_ind:
    f = io.loadmat('NoLoad_'+file+'V.mat')
    a = float(file)*np.ones((len(f['Data1_AI_0']), 1))
    b = np.double(f['Data1_AI_0'])
    N = len(b)
    I = np.floor(N/L)-1  # total number of observations (N/L)
    Ntest = int(np.floor(I/4))   # 1/4 of I for test
    Nvalid = int(np.floor(3*I/16))  # validation is 1/4 of the 3/4*I (training) = 3/16
    Ntrain = int(I-Nvalid-Ntest)
    train_ind_max = Ntrain*L
    valid_ind_max = train_ind_max+Nvalid*L
    test_ind_max = valid_ind_max+Ntest*L
    
    data_temp_train = np.concatenate((a[0:train_ind_max], b[0:train_ind_max]), axis=1)
    data_temp_valid = np.concatenate((a[train_ind_max:valid_ind_max], b[train_ind_max:valid_ind_max]), axis=1)
    data_temp_test = np.concatenate((a[valid_ind_max:test_ind_max], b[valid_ind_max:test_ind_max]), axis=1)
    data_train_list.append(data_temp_train)
    data_valid_list.append(data_temp_valid)
    data_test_list.append(data_temp_test)

data_train = np.concatenate(data_train_list, axis=0)
data_valid = np.concatenate(data_valid_list, axis=0)
data_test = np.concatenate(data_test_list, axis=0)

# Normalize using robust standardization for better performance
dmin = data_train.min(axis=0)
dmax = data_train.max(axis=0)
max_min = dmax - dmin
data_train = (data_train-dmin)/max_min
data_valid = (data_valid-dmin)/max_min
data_test = (data_test-dmin)/max_min

print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")

# Enhanced data generator with spectral features option
def generator(data, window, delay, min_index, max_index,
              shuffle=False, batch_size=batch_size, step=step, 
              spectral_features=False):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + window
    
    while 1:
        if shuffle:
            sample_ind = np.random.randint(
                    min_index, max_index//window, size=batch_size)
            rows = sample_ind*window
        else:
            if i >= max_index:
                i = min_index + window
            rows = np.arange(i, min(i + batch_size*window, max_index), window)
            i = rows[-1]+window
        
        # Base time series features
        samples = np.zeros((len(rows), window // step, (data.shape[-1]-1)))
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - window, rows[j], step)
            samples[j] = data[indices, 1:]
            targets[j] = data[rows[j]-1 + delay][0]
            
        # Add spectral features if requested (for advanced models)
        if spectral_features:
            # Compute FFT for each window
            fft_samples = np.zeros((len(rows), window//2+1, (data.shape[-1]-1)))
            for j in range(len(rows)):
                fft_samples[j] = np.abs(np.fft.rfft(samples[j], axis=0))
            
            yield [samples, fft_samples], targets
        else:
            yield samples, targets

# Setup generators (standard version for compatibility with evaluation code)
train_gen = generator(data_train, window=window, delay=delay, min_index=0,
                      max_index=None, shuffle=True, step=step, batch_size=batch_size)

val_gen = generator(data_valid, window=window, delay=delay, min_index=0,
                    max_index=None, shuffle=True, step=step, batch_size=batch_size)

test_gen = generator(data_test, window=window, delay=delay, min_index=0,
                    max_index=None, step=step, batch_size=batch_size)

val_steps = data_valid.shape[0]//(window*batch_size)
test_steps = data_test.shape[0]//(window*batch_size)

# Define a TCN (Temporal Convolutional Network) block
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.1):
    # First Conv layer with dilation
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, 
                   padding='causal', dilation_rate=dilation_rate)(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(dropout_rate)(conv1)
    
    # Second Conv layer with dilation
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size,
                   padding='causal', dilation_rate=dilation_rate)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(dropout_rate)(conv2)
    
    # Add residual connection if input and output shapes match
    if x.shape[-1] == filters:
        result = Add()([x, conv2])
    else:
        # Use 1x1 conv to match dimensions
        shortcut = Conv1D(filters=filters, kernel_size=1)(x)
        result = Add()([shortcut, conv2])
    
    return result

# Define frequency domain feature extraction module
def frequency_module(x, units):
    # Extract features in frequency domain using FFT
    fft_features = Lambda(extract_spectral_features)(x)
    
    # Process FFT features
    fft_conv = Conv1D(filters=units, kernel_size=3, padding='same')(fft_features)
    fft_conv = BatchNormalization()(fft_conv)
    fft_conv = Activation('relu')(fft_conv)
    
    # Pool to reduce dimensions
    fft_pool = AveragePooling1D(pool_size=4)(fft_conv)
    
    return fft_pool

# Units configuration from the original code
units_configs = [[16],[16,16,16],[16,16,16,16,16],[16,16,16,16,16,16,16],
         [32],[32,32,32],[32,32,32,32,32],[32,32,32,32,32,32,32],
         [64],[64,64,64],[64,64,64,64,64],[64,64,64,64,64,64,64], 
         [128],[128,128,128],[128,128,128,128,128],[128,128,128,128,128,128,128],
         [256],[256,256,256],[256,256,256,256,256],[256,256,256,256,256,256,256]]

# Process each units configuration
for num_units in units_configs:
    filenameFig = 'Fast_TCN_Voltage'
    
    for num_unit in num_units:
        filenameFig = filenameFig + '_' + str(num_unit)
    
    filename = filenameFig
    filename_model = filename + '.h5'
    
    print(f"\nBuilding model with units: {num_units}")
    model_start_time = time.time()
    
    # Build the TCN-based model
    input_layer = Input(shape=(window//step, 1))
    
    # Initial feature extraction
    x = Conv1D(filters=num_units[0], kernel_size=5, padding='causal')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
# Add frequency domain features if using larger units (more capacity)
if num_units[0] >= 64:
    freq_features = frequency_module(input_layer, num_units[0]//2)
    # Match dimensions and combine
    freq_features = Conv1D(filters=num_units[0], kernel_size=1)(freq_features)
    
    # Get dimensions
    target_length = window//step
    
    # Reshape to match input dimensions using 1D convolution
    # First, apply global pooling to remove dependency on input length
    freq_pooled = GlobalAveragePooling1D()(freq_features)
    # Then reshape and repeat to match target dimensions
    freq_expanded = layers.RepeatVector(target_length)(freq_pooled)
    
    # Apply another Conv1D to ensure proper feature matching
    freq_features = Conv1D(filters=num_units[0], kernel_size=1, padding='same')(freq_expanded)
    
    # Combine time and frequency domain features
    x = Add()([x, freq_features])
    
    # Build TCN blocks with exponentially increasing dilation rates
    # This gives an exponentially growing receptive field with depth
    for i, units in enumerate(num_units):
        dilation_rate = 2**i  # Exponential dilation
        x = tcn_block(x, filters=units, kernel_size=3, 
                      dilation_rate=dilation_rate, 
                      dropout_rate=0.1 if i < len(num_units)-1 else 0.2)
    
    # Global pooling to reduce dimensions
    x = GlobalAveragePooling1D()(x)
    
    # Prediction head
    x = Dense(units=max(32, num_units[-1]//2), activation='relu')(x)
    x = BatchNormalization()(x)
    output_layer = Dense(units=1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Use mixed precision for faster computation when appropriate
    if tf.__version__ >= '2.4.0' and num_units[-1] >= 64:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    
    # Compile with efficient optimizer settings
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    print(f"Model built in {time.time() - model_start_time:.2f} seconds")
    model.summary()
    
    # Add callbacks for better training
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # Train the model
    print("Training model...")
    training_start_time = time.time()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=500,
        epochs=150,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    training_time = time.time() - training_start_time
    print(f"Model training completed in {training_time:.2f} seconds")
    
    # Save model and history
    model.save(filename + '.h5')
    
    with open(filename, 'wb') as handle:
        pickle.dump(history.history, handle)
    
    # Evaluation
    print("Evaluating model...")
    eval_start_time = time.time()
    
    data_test_for_evaluate = data_valid[:,1:].reshape((len(data_valid)//window, window, 1))
    targets_test = data_valid[:,:1].reshape((len(data_valid)//window, window, 1))
    predicted_targets = np.zeros((len(data_test_for_evaluate),))
    true_targets = np.zeros((len(data_test_for_evaluate),))
    
    for i in range(len(data_test_for_evaluate)):
        true_targets[i] = targets_test[i,window-1]
    
    target_mean = true_targets.mean(axis=0)
    
    # Batch prediction for faster evaluation
    predicted_targets = model.predict(data_test_for_evaluate, batch_size=64).flatten()
    
    # Calculate error metrics
    MSE = np.mean(np.square(predicted_targets-true_targets))
    MAE = np.mean(np.abs(predicted_targets-true_targets))
    
    RRSE = 100 * np.sqrt(MSE * len(true_targets) / (sum(abs(true_targets-target_mean)**2)))
    RAE = 100 * MAE * len(true_targets) / sum(abs(true_targets-target_mean))
    
    print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")
    print('MSE: {:.2e}'.format(MSE))
    print('MAE: {:.2e}'.format(MAE))
    print('RRSE: {:.2f}%'.format(RRSE))
    print('RAE: {:.2f}%'.format(RAE))
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    epoch_count = range(1, len(history.history['loss']) + 1)
    plt.plot(epoch_count, np.array(history.history['loss']), 'b--')
    plt.plot(epoch_count, np.array(history.history['val_loss']), 'r-')
    
    y = history.history['val_loss']
    ymin = min(y)
    xpos = y.index(min(y))
    xmin = epoch_count[xpos]
    
    # Format error for scientific notation
    string1 = 'MSE = ' + '{:.2e}'.format(float(ymin))
    string2 = '\n' + 'RAE = ' + f'{RAE:.2f}%' + '\n' + 'RRSE = ' + f'{RRSE:.2f}%'
    string3 = '\n' + 'Training Time = ' + '{:.1f}s'.format(training_time)
    string = string1 + string2 + string3
    
    ax.annotate(string, xy=(xmin, ymin), xycoords='data',
                xytext=(-80, 85), textcoords='offset points',
                bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                size=12,
                arrowprops=dict(arrowstyle="->"))
    
    plt.title('TCN $\mathit{N}$=' + str(len(num_units)) + ', $\mathit{M}$=$\mathit{L}$=' + str(num_units[0]))
    xint = range(min(epoch_count)-1, math.ceil(max(epoch_count)), 20)
    plt.xticks(xint)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend(['Training', 'Validation'], loc="best")
    
    filename1 = filename + '_loss'
    fig.set_size_inches(5.46, 3.83)
    fig.savefig(filename1 + '.pdf', bbox_inches='tight')
    
    # Save scores
    score = []
    score.append(ymin)  # MSE (keras)
    score.append(MSE)   # MSE calculated
    score.append(MAE)   # MAE
    score.append(RRSE)  # RRSE
    score.append(RAE)   # RAE
    score.append(training_time)  # Training time
    filenameTXT = filename + '.txt'
    np.savetxt(filenameTXT, score)
    
    # Clean up
    tf.keras.backend.clear_session()
    del model
    print(f"Completed model with units: {num_units}")