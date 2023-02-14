import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model('temp_best_model.hdf5')
layers = model.layers
w = np.asarray(layers[-1].get_weights()[0])

w = np.abs(w)

w_max = np.max(w)
w_min = np.min(w)

# norm = plt.Normalize(w_min,w_max)

# print((w[:,3].argsort()[::-1][:50] > 128).sum()/50)
# exit()

cm = plt.get_cmap('RdYlBu')

n_classes = int(w.shape[1])

classes = ["class "+str(i) for i in range(n_classes)]

f, axis = plt.subplots(nrows=2,ncols=1,figsize=(20,10))

axis[0].set_title("Heat map of Weight matrix of the connections with the supervised features",fontsize=20)
axis[1].set_title("Heat map of Weight matrix of the connections with the self-supervised features",fontsize=20)
sns.heatmap(w[:128].T,ax=axis[0],cmap="summer",yticklabels=classes,linewidth=0.005,linecolor="#222")
sns.heatmap(w[128:].T,ax=axis[1],cmap="summer",yticklabels=classes,linewidth=0.005,linecolor="#222")

plt.savefig("/home/hadi/stage_2022/internship mulhouse/pre-papers/TLIT/weights_heat_map/heat_map_w_DiatomSizeReduction.pdf")
plt.savefig("/home/hadi/stage_2022/internship mulhouse/pre-papers/TLIT/weights_heat_map/heat_map_w_DiatomSizeReduction.png")