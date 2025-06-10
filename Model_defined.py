from numba import cuda
cuda.close()
import keras
from keras import layers
from keras.layers import Dense,Input,ReLU,Dropout
from keras.models import Sequential, Model
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import os
import pickle
from keras.constraints import unit_norm, max_norm
from keras.models import Sequential, Model
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
e1 = pd.read_csv('./exp1.csv',index_col=0)
m1 = pd.read_csv("./me1.csv", index_col=0)
e2 = pd.read_csv("./exp2.csv", index_col=0)
m2 = pd.read_csv("./me2.csv", index_col=0)
en=pd.read_csv("./en.csv",index_col=0)
mn=pd.read_csv("./mn.csv",index_col=0)
eps=1e-10
weight = pd.read_csv("./genecor.csv", index_col=0)
weight=weight.loc[m1.index]
weight=weight.values.reshape(-1,1)
seed=1
tf.random.set_seed(seed)
sigmoid1 = weight*(m1+eps)*(e1+eps)
sigmoid2 = weight*(m2+eps)*(e2+eps)
sigmoid3= weight*(mn+eps)*(en+eps)
sigmoid = pd.concat([sigmoid1.T, sigmoid2.T,sigmoid3.T], axis=0)
label=pd.DataFrame([1]*sigmoid1.shape[1] + [2]*sigmoid2.shape[1] + [0]*sigmoid3.shape[1],columns=['label'], index = sigmoid.index)	
EPOCHS = 1000
BATCH_SIZE = 16
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=200,
    mode='max',
    restore_best_weights=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

####Define the model####
from keras.models import load_model
from keras.layers import Add, Convolution2D, Input,concatenate,BatchNormalization
def make_model(metrics=METRICS, output_bias=None, my_layers=None):
	x = Input(shape=(13295,),name='inputdata')
	layer1 = layers.Dense(my_layers[0], activation="relu", name="layer1",kernel_constraint=max_norm(3),bias_constraint=max_norm(3))(x)
	layer1 = BatchNormalization(name='BN1')(layer1)
	layer1_D = layers.Dropout(0.1)(layer1)
	layer2 = layers.Dense(my_layers[1], activation="relu", name="layer2",kernel_constraint=max_norm(3),bias_constraint=max_norm(3))(layer1_D)
	layer2 = BatchNormalization()(layer2)
	layer2_D = layers.Dropout(0.1)(layer2)
	z = concatenate([x,layer2_D])
	layer3=layers.Dense(3, activation='softmax',name='layer3')(z)
	model = Model(inputs = x, outputs = layer3)
	model.compile(
	optimizer=opt,
	loss=keras.losses.CategoricalCrossentropy(),#weighted_cce_d,,
	metrics=metrics)
	return model

####The performance of Skip model ####
result = []
latent_layers=[[2048,128]]
skfold=StratifiedKFold(n_splits=3,random_state=1,shuffle=True)

for i,(train_index, test_index) in enumerate(skfold.split(sigmoid,label)):
	os.makedirs("/myData/my/PANCANCER/LUNG_NEW_model/fold_"+str(i),exist_ok=True)
	pd.DataFrame(train_index).to_csv("./_model/fold_"+str(i)+"/Train.csv",index=False)
	pd.DataFrame(test_index).to_csv("./_model/fold_"+str(i)+"/Val.csv",index=False)

for mylayer in latent_layers:
	weighted_model = make_model(my_layers=mylayer)
	for folder in os.listdir("./_model/"):
		folder=os.path.join("./_model/",folder)
		print(folder)
		train_index = pd.read_csv(folder+"/Train.csv",header=0,index_col=None)
		test_index = pd.read_csv(folder+"/Val.csv",header=0,index_col=None)
		train_features= sigmoid.iloc[train_index['0']]
		train_labels=label.iloc[train_index['0']]
		train_labels=tf.keras.utils.to_categorical(train_labels, num_classes=3)
		test_features=sigmoid.iloc[test_index['0']]
		test_labels=label.iloc[test_index['0']]
		test_labels=tf.keras.utils.to_categorical(test_labels, num_classes=3)
		weighted_history = weighted_model.fit(
		train_features,
		train_labels,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		class_weight={0:0.25,1:0.5,2:0.25},
		#sample_weight=weight_samp,
		callbacks = [early_stopping],
		validation_data=(test_features, test_labels),
		shuffle=True,
		)
		train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
		test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
		weighted_results = weighted_model.evaluate(test_features, test_labels,
											   batch_size=BATCH_SIZE, verbose=0)
		predict_dump = pd.concat([pd.DataFrame(test_predictions_weighted), pd.DataFrame(test_labels)], axis=1)
		predict_dump.to_csv(folder+'/'+str(mylayer[0])+'_'+str(mylayer[1])+"_result_fold.csv",index=False)
		tf.keras.models.save_model(weighted_model, folder+'/'+'2048_128_13295features_orimodel/')
		n_classes=3
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_predictions_weighted[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(n_classes):
			mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		mean_tpr /= n_classes
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), test_predictions_weighted.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		roc_auc['layer0']=mylayer[0]
		roc_auc['layer1']=mylayer[1]
		result.append(roc_auc)
		print(roc_auc)	
####save results
pd.DataFrame(result).to_csv("./2048_128skip_ori_result.csv",index=False)


#####Three-cross valdation###
rom keras.models import load_model
test_result = []	
for folder in os.listdir("./_model/"):
	folder=os.path.join("./_model/",folder)
	print(folder)
	train_index = pd.read_csv(folder+"/Train.csv",header=0,index_col=None)
	test_index = pd.read_csv(folder+"/Val.csv",header=0,index_col=None)
	train_features= sigmoid.iloc[train_index['0']]
	train_labels=label.iloc[train_index['0']]
	train_labels=tf.keras.utils.to_categorical(train_labels, num_classes=3)
	test_features=sigmoid.iloc[test_index['0']]
	test_labels=label.iloc[test_index['0']]
	test_labels=tf.keras.utils.to_categorical(test_labels, num_classes=3)
	weighted_model = load_model(folder+"/2048_128_13295features_orimodel")
	test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
	predicttest_dump = pd.concat([pd.DataFrame(test_predictions_weighted), pd.DataFrame(test_labels)], axis=1)
	test_result.append(predicttest_dump)

test_result = pd.concat(test_result,axis=0)
test_result.to_csv("./13295features_test_result.csv",index=False)

