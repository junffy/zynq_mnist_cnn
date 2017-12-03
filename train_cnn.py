# This code is based on 
# https://github.com/oreilly-japan/deep-learning-from-scratch
# This modified code also takes over the MIT License.


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.trainer import Trainer
from cnn_int import SimpleConvNet
from common.fwrite_cnn_weight import fwrite_cnn_weight
from common.fwrite_cnn_bias import fwrite_cnn_bias
from common.fwrite_af_weight import fwrite_af_weight

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 10, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        #conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

'''x_testn, t_testn = x_test[:500], t_test[:500]
test_accn = network.accuracy_msg(x_testn, t_testn)
print(test_accn)'''

'''train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)
print(train_acc, test_acc)
train_acc_int = network.accuracy_int(x_train, t_train)'''
#test_acc_int = network.accuracy_int(x_test, t_test)
#print(test_acc_int)

MAGNIFICATION = 2 ** (9-1)
fwrite_cnn_weight(network.params['W1'], 'conv1_weight.h', 'conv1_fweight', 'conv1_weight', MAGNIFICATION)
fwrite_cnn_bias(network.params['b1'], 'conv1_bias.h', 'conv1_fbias', 'conv1_bias', MAGNIFICATION)
fwrite_cnn_bias(network.params['b2'], 'af1_bias.h', 'af1_fbias', 'af1_bias', MAGNIFICATION)
fwrite_cnn_bias(network.params['b3'], 'af2_bias.h', 'af2_fbias', 'af2_bias', MAGNIFICATION)
fwrite_af_weight(network.params['W2'], 'af1_weight.h', 'af1_fweight', 'af1_weight', MAGNIFICATION)
fwrite_af_weight(network.params['W3'], 'af2_weight.h', 'af2_fweight', 'af2_weight', MAGNIFICATION)
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
