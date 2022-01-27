import sys
import random
import matplotlib.pyplot as plot
import numpy as np
from keras.models import load_model
import pickle

###################################################################

########################  LOADING DATA  ###########################

###################################################################


AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


###################################################################

##########################  FUNCTIONS  ############################

###################################################################

# 加载模型
def load_sca_model(model_file):
	try:
		model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model


# Compute the position of the key hypothesis key amongst the hypotheses
def rk_key(rank_array, key):
    key_val = rank_array[key]
    return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]


# Compute the evolution of rank
def rank_compute(prediction, att_plt, key, mask, offset, byte):
    """
    - prediction : predictions of the NN
    - att_plt : plaintext of the attack traces
    - key : Key used during encryption
    - mask : the known mask used during the encryption
    - offset : offset used for computing the Sbox output
    - byte : byte to attack
    """

    (nb_trs, nb_hyp) = prediction.shape

    idx_min = nb_trs
    min_rk = 255

    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs, 255)
    prediction = np.log(prediction + 1e-40)

    for i in range(nb_trs):
        for k in range(nb_hyp):
            key_log_prob[k] += prediction[i, (AES_Sbox[int(att_plt[i, byte]) ^ int(k)] ^ int(
                mask[int(offset[i, byte] + 1) % 16]))]  # Computes the hypothesis values

        rank_evol[i] = rk_key(key_log_prob, key[byte])
    # print(rank_evol)
    return rank_evol


# Performs attack
def perform_attacks(nb_traces, predictions, nb_attacks, plt, key, mask, offset, byte=0, shuffle=True, savefig=True,
                    filename='fig'):
    """
    Performs a given number of attacks to be determined

    - nb_traces : number of traces used to perform the attack
    - predictions : array containing the values of the prediction
    - nb_attacks : number of attack to perform
    - plt : the plaintext used to obtain the consumption traces
    - key : the key used to obtain the consumption traces
    - mask : the known mask used during the encryption
    - offset : offset used for computing the Sbox output
    - byte : byte to attack
    - shuffle : traces have to be shuffled ? (boolean, default = True)

    """

    (nb_total, nb_hyp) = predictions.shape

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in range(nb_attacks):
        print(i)

        if shuffle:
            l = list(zip(predictions, plt, offset))
            random.shuffle(l)
            sp, splt, soffset = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            soffset = np.array(soffset)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]
            att_offset = soffset[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt[:nb_traces]
            att_offset = offset[:nb_traces]

        rank_evolution = rank_compute(att_pred, att_plt, key, mask, att_offset, byte=byte)
        all_rk_evol[i] = rank_evolution

    rk_avg = np.mean(all_rk_evol, axis=0)

    # 将rank值和traces的数量存入文件
    dic = {'rank': rk_avg}
    with open("/home/lyn/存储/dpav4_test/CBAPD_dpa_rank.h5", 'wb') as f1:
        pickle.dump(dic, f1)

    if (savefig == True):
        plot.rcParams['figure.figsize'] = (20, 10)
        plot.ylim(-5, 200)
        plot.grid(True)
        plot.plot(rk_avg, '-', label='avg')
        plot.xlabel('Number of traces')
        plot.ylabel('Rank')

        legend = plot.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=1, ncol=1)
        title = plot.title(filename + '\n_' + str(nb_traces) + 'trs_' + str(nb_attacks) + 'att', loc='center')
        plot.show()
        # plot.savefig("/home/lyn/存储/dpav4_test/fig/CBAPD_dpav4_test.jpg")


    return (rk_avg)

# 给路径
root = "./"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"
root_1 = "/home/lyn/存储/"
DPAv4_data_folder = root+"DPAv4_dataset/"
DPAv4_trained_models_folder = root_1+"DPAv4_trained_models/"

model_file = "/home/lyn/存储/ASCAD_trained_models/CBAPD_dpav4_epoch50_batch200.h5"

# 给参数
nb_traces_attacks = 30
nb_attacks = 100

# 加载模型
model = load_sca_model(model_file)

# 加载数据
(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (np.load(DPAv4_data_folder + 'profiling_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'profiling_labels_dpav4.npy')), (np.load(DPAv4_data_folder + 'attack_traces_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_labels_dpav4.npy')), (np.load(DPAv4_data_folder + 'profiling_plaintext_dpav4.npy'), np.load(DPAv4_data_folder + 'attack_plaintext_dpav4.npy'))

Reshaped_X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

real_key = np.load(DPAv4_data_folder + "key.npy")
mask = np.load(DPAv4_data_folder + "mask.npy")
att_offset = np.load(DPAv4_data_folder + "attack_offset_dpav4.npy")
model_name = "CBAPD_DPA_v4_test"

# print(X_attack)
predictions = model.predict(X_attack)
#
# np.save(predictions_folder + 'predictions_' + '.npy', predictions)

avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, mask=mask, offset=att_offset, byte=0, filename=model_name)