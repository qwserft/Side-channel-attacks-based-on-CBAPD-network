import os
import pickle
import random
import sys
import time

import h5py
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def CBAPD_model(classes=256):
        # 输入层
        input_shape = (4000, 1)
        img_input = tf.keras.Input(shape=input_shape)

        # 第一个卷积块
        x = tf.keras.layers.Conv1D(64, 11, padding='same', name='block1_conv1')(img_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('selu')(x)
        x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block1_pool')(x)

        # 第二个
        x = tf.keras.layers.Conv1D(128, 11, padding='same', name='block2_conv1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('selu')(x)
        x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block2_pool')(x)

        # 第三个
        x = tf.keras.layers.Conv1D(256, 11, padding='same', name='block3_conv1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('selu')(x)
        x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block3_pool')(x)

        # 第四个
        x = tf.keras.layers.Conv1D(512, 11, padding='same', name='block4_conv1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('selu')(x)
        x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block4_pool')(x)

        # 第五个
        x = tf.keras.layers.Conv1D(512, 11, padding='same', name='block5_conv1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('selu')(x)
        x = tf.keras.layers.AveragePooling1D(2, strides=2, name='block5_pool')(x)


        x = tf.keras.layers.Flatten(name='flatten')(x)

        x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        # x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        # x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

        inputs = img_input
        model = tf.keras.Model(inputs, x, name='CBAPD')
        optimizer = tf.keras.optimizers.RMSprop(lr=0.00001)
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer, metrics=['accuracy'])
        return model

def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100):
    # 检查文件路径是否存在
    check_file_exists(os.path.dirname(save_file_name))
    # 每个epoch保存一次模型
    save_model = tf.keras.callbacks.ModelCheckpoint(save_file_name)

    input_layer_shape = model.get_layer(index=0).input_shape

    if len(input_layer_shape) < 3:
        input_layer_shape = list(input_layer_shape[0])

    Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))


    print(Reshaped_X_profiling.shape)

    callbacks = [save_model]
    print(model.summary())
    history = model.fit(x=Reshaped_X_profiling, y=tf.keras.utils.to_categorical(Y_profiling, num_classes=256), batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)

    tf.keras.models.save_model(model, DPAv4_trained_models_folder + model_name)

    return history

root = "./"
root_1 = "/home/lyn/存储/"
ASCAD_data_folder = root+"ASCAD_dataset/"
ASCAD_trained_models_folder = root_1+"ASCAD_trained_models/"
history_folder = root+"training_history/"
start = time.time()

DPAv4_data_folder = root+"DPAv4_dataset/"
DPAv4_trained_models_folder = root_1+"DPAv4_trained_models/"


# Load the profiling traces
# (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(ASCAD_data_folder + "ASCAD.h5", load_metadata=True)

# 加载npy格式的数据
(X_profiling, Y_profiling), (X_attack, Y_attack),  = (np.load(DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy')), (np.load(DPAv4_data_folder + 'attack_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_labels_dpav4.npy'))

print(X_profiling.shape)
print(Y_profiling.shape)
print(X_attack.shape)
print(Y_attack.shape)
##################开始训练####################

# 给参数
nb_epochs = 50
batch_size = 200
input_size = 4000
learning_rate = 0.00001

# 给模型
model = CBAPD_model()

model_name = "CBAPD_dpav4_epoch50_batch200.h5"

print('\n Model name = '+model_name)

# 记录信息
history = train_model(X_profiling, Y_profiling, model, ASCAD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size)

end = time.time()

print('Temps execution = %d'%(end-start))

print("\n########训练完成########\n")

# 保存信息
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

