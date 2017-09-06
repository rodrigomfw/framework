# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)
import math
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, Activation, Dropout, Reshape, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adagrad, Adam
from sklearn.preprocessing import scale
from keras.models import load_model
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from random import sample
from copy import deepcopy
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import log_loss
import os


def read_data(DIR,ID,sop):
  """
  Inputs
    dir - data directory
    id - patient ID
    sop - seizure occurrence period

  Outputs
    X - 
    y - target matrix
  """
  feat = DIR+'dados_' + ID+ '.csv'
  targ = DIR+ 'target_'+ ID+ '_'+str(sop) +'.csv'
  X = pd.read_csv(feat, header=None)
  y = pd.read_csv(targ, header=None)
  X = X.as_matrix()
  y = y.as_matrix()
  X = X.astype(np.float32)
  y = y.astype(np.float32)
  y = y.squeeze()
  return X,y

def separate_data(X,y,ptr,pv,idx):
  """
  Inputs
    X - feature matrix
    y - target matrix
    ptr - percentage of seizures included in the training set
    pv - percentage of seizures included in the validation set
    idx - indexes of where each seizure begins and ends
  
  Outputs
    X_train - train feature matrix 
    y_train - train target matrix
    X_valid - validation feature matrix 
    y_valid - validation target matrix
    X_test - test feature matrix 
    y_test - test target matrix
  """

  ntr = int(len(idx)*ptr)
  nt  = ntr + int(len(idx)*pv)
  vtr = int(idx[ntr-1][1]+1)
  vt  = int(idx[nt][1]+1)
  X_train = X[:vtr,:]
  X_valid = X[vtr:vt,:]
  X_test =  X[vt:,:]
  y_train = y[:vtr,:]
  y_valid = y[vtr:vt,:]
  y_test =  y[vt:,:]
  return X_train, X_valid, X_test, y_train, y_valid, y_test

def metrics(pred,y):
  """
  Inputs
    pred - predictions matrix
    y - target matrix
  Outputs
    sens - sensitivity
    spec - specificity
    prec - precision
    acc - accuracy
    f1 - f1 score
  """
  tp = 0.
  tn = 0.
  fp = 0.
  fn = 0.   
  for i in range(len(pred)):
    if np.argmax(y_test[i,:]) == 1 and np.argmax(pred[i,:]) == 1:
      tp += 1
    elif np.argmax(y_test[i,:]) == 0 and np.argmax(pred[i,:]) == 0:
      tn += 1              
    elif np.argmax(y_test[i,:]) == 0 and np.argmax(pred[i,:]) == 1:
      fp += 1
    elif np.argmax(y_test[i,:]) == 1 and np.argmax(pred[i,:]) == 0:
      fn += 1
  sens = tp/(tp+fn)
  spec = tn/(tn+fp)
  rec = sens
  try:
      prec = tp/(tp+fp)
  except:
      prec = 0
  acc = (tp+tn)/(tp+tn+fp+fn)
  try:
      f1 = 2*prec*rec/(prec+rec)
  except:
      f1 = 0
  return sens,spec,prec,acc,f1

  
def idx_seizure(y):
  """
  Inputs
    y - target matrix
  Outputs
    idx - indexes of where each seizure begins and ends
  """
  idx = []
  for i in range(0,y.shape[0]):
    if y[i,1] == 1 and y[i-1,1] == 0 or (i==0 and y[0,1] == 1):
      for j in range(i,i+sop*3*60/5+1):    
        try:
          if y[j,1] == 1 and y[j+1,1] == 0:
            idx.append([i,j])
            break
        except IndexError:
          idx.append([i,j])
          break
  return idx


def create_seq(X,y,seq_length,overlap,predstep):
  """
  Inputs
    X - feature matrix
    y - target matrix
    seq_length - sequence length
    idx - indexes of where each seizure begins and ends
  Outputs
    X_seq - sequential feature matrix 
    y_seq - target matrix
  """
  nb_samples = X.shape[0]
  new_size = (nb_samples - nb_samples%overlap)/overlap-seq_length-predstep
  X_ = np.zeros((new_size, seq_length, X.shape[1]))
  y_ = np.zeros((new_size,y.shape[1]))
  for i in range(0,new_size):
    j = i*overlap
    X_[i,:,:] = X[j:j+seq_length,:]
    y_[i,:] = y[j+seq_length-1,:]
  return X_,y_

def create_seq_train(X,y,seq_length,overlap,predstep,idx_seizure):
  """
  Inputs
    X - feature matrix
    y - target matrix
    seq_length - sequence length
    idx - indexes of where each seizure begins and ends
  Outputs
    X_seq - sequential feature matrix 
    y_seq - target matrix
  """
  indexes_preict = []
  for idx in idx_seizure:
    for i in range(idx[0],idx[1]+101):
      if i >= seq_length-1 and i<y.shape[0]:
        indexes_preict.append(i)
  n = len(idx_seizure)*(sop*60/5 + 1)
  k = n - (len(indexes_preict)-n)
  k = k*2
  indexes = list(range(seq_length,y.shape[0]))
  indexes = [x for x in indexes if x not in indexes_preict]
  indexes = sample(indexes,k)
  indexes = indexes+indexes_preict
  new_size = len(indexes)
  X_ = np.zeros((new_size, seq_length, X.shape[1]))
  y_ = np.zeros((new_size,y.shape[1]))
  for i in range(len(indexes)):
    j = indexes[i]
    X_[i,:,:] = X[j-seq_length+1:j+1,:]
    y_[i,:] = y[j,:]
  X_,y_ = shuffle(X_,y_,random_state=0)
  return X_,y_

def build_net(structure,X_train,y_train,X_test,y_test,lr,d,w):
  """
  Inputs
    X_train - train feature matrix
    y_train - train target matrix
    X_valid - validation feature matrix
    y_valid - validation target matrix
    structure - list containing network architecture
    lr - learning rate
    d - dropout
    w - regularization strength
  Outputs
    model - LSTM model. The model is also saved to a file.
  """
  print('Build model...')
  model = Sequential()
  n = len(structure)
  for i in range(len(structure)):
    type = structure[i][0]
    if type == 'cnn':
      n_filter = structure[i][1]
      filter_length = structure[i][2]
      if i==0:
        model.add(Conv1D(nb_filter=n_filter,
                    filter_length=filter_length,
                    border_mode='valid',
                    activation='relu',
                    subsample_length=1,
                    input_shape=(seq_length, X_train.shape[2]),
                    kernel_regularizer = l2(0.001),
                    bias_regularizer = l2(0.001)))
      elif i>0:
        model.add(Conv1D(nb_filter=n_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1,
        kernel_regularizer = l2(0.001),
        bias_regularizer = l2(0.001)))
        
    if type == 'lstm':
      n_neuron = structure[i][1]
      if i==0 and n == 1:
          model.add(LSTM(n_neuron, input_shape=(seq_length, X_train.shape[2]),
                       return_sequences=False, dropout=d, recurrent_dropout=d,
                       kernel_regularizer=l2(w),recurrent_regularizer=l2(w)))
      elif i==0 and n != 1:
          model.add(LSTM(n_neuron, input_shape=(seq_length, X_train.shape[2]),
                         return_sequences=True, dropout=d, recurrent_dropout=d,
                       kernel_regularizer=l2(w),recurrent_regularizer=l2(w)))
      elif i>0 and i == n-1:  
          model.add(LSTM(n_neuron, return_sequences=False, dropout=d, recurrent_dropout=d,
                       kernel_regularizer=l2(w),recurrent_regularizer=l2(w)))
      else:
          model.add(LSTM(n_neuron, return_sequences=True, dropout=d, recurrent_dropout=d,
                       kernel_regularizer=l2(w),recurrent_regularizer=l2(w)))
          
    if type == 'dropout':
      p = structure[i][1]
      model.add(Dropout(p))
    if type == 'maxpool':
      p_length = structure[i][1]
      model.add(MaxPooling1D(pool_length=p_length))

  model.add(Dense(y_train.shape[1], activity_regularizer = l2(0.0)))
  model.add(Activation('softmax'))
  optimizer = Adam(lr=lr)

  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  checkpointer = ModelCheckpoint(filepath="tmp/weights1.hdf5", verbose=1, save_best_only=True)
  earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')

  callbacks = [History(),checkpointer,earlystopping]
  hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                   class_weight = {0:1, 1:1.*n_int/n_preict},validation_data=(X_test,y_test),callbacks=callbacks)
  model.load_weights('tmp/weights1.hdf5')
  f = open(('history/'+ID+'_'+str(seq_length)+'_'+
            str(sop)+'_'+str(lr)+'_'+str(d)+'_'+str(w)+'_'+str(structure)+'_history.txt'),'w')
  f.write(str(hist.history['loss']))
  f.write('\n')
  f.write(str(hist.history['val_loss']))
  f.close()
  model.save(model_dir+ID+'_'+str(seq_length)+'_'
               +str(sop)+'_'+str(lr)+'_'+str(d)+'_'+str(w)+'_'+str(structure)+'_model.h5')
  return model


def check_nan(X,y):
  """
  Inputs
    X - feature matrix
    y - target matrix
  Outputs
    X - feature matrix
    y - target matrix
  """
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if math.isnan(X[i,j]):
        X[i,j]=0
  return X,y     
           
def firing_power(pred,y,th,prob,n):
  """
  Inputs
    pred - predictions matrix
    y - target matrix
    th - threshold for alarm generation using the firing power filter
    n - size of the filter window
    prob - treshold of the probability at which a pattern is classified 
           as pre-ictal
  Outputs
    new_pred - prediction matrix after applying the filter
    new_y - target matrix 
  """
  size = pred.shape[0]  
  new_pred = np.zeros((size-n,2))
  new_y = deepcopy(y)
  new_y = new_y[n:,:]
  for i in range(n,size):
    fp = pred[pred[i-n:i,1]>=prob].shape[0]/float(n)
    if fp >= th:
      new_pred[i-n,1] = 1
    else:
      new_pred[i-n,0] = 1
  for i in range(new_pred.shape[0]):
    if new_pred[i,1] == 1:
      for j in range(1,sop*60/5+1):
        try:
          new_pred[i+j,1] = 0
          new_pred[i+j,0] = 1
        except IndexError:
          pass
  return new_pred, new_y


def performance(pred,y,idx):
  """
  Inputs
   pred - prediction matrix, after firing power application
   y - target matrix
   idx - indexes of where each seizure begins and ends
  Outputs
    sens - sensitivity
    fpr - false positive rate
  """
  fp = 0.
  tp = 0.
  for i in range(pred.shape[0]):
    if np.argmax(y_test[i,:]) == 0 and np.argmax(pred[i,:]) == 1:
      fp += 1
  FPR = fp/(y_test[y_test[:,0]==1].shape[0]-fp*sop*60/5)*720
  for i in idx:
    for j in range(i[0],i[1]+1):
      try:
        if np.argmax(pred[j,:]) == 1:
          tp += 1
          break
      except IndexError:
          break         
  sens = float(tp)/len(idx)
  return FPR, sens   


def validation(X_valid,y_valid,X_test,y_test,type,sop):
  """
  Inputs
    X_valid - validation feature matrix 
    y_valid - validation target matrix
    X_test - test feature matrix 
    y_test - test target matrix
    type - string with values 'scalp' or 'inv'
    sop - seizure occurrence period
  Outputs
    sensitivity and false positive  rate are saved into a file.
  """
  model = load_model(model_dir+ID+'_'+str(seq_length)+'_'
               +str(sop)+'_'+str(lr)+'_'+str(d)+'_'+str(w)+'_'+str(structure)+'_model.h5')
  
  idx_valid = idx_seizure(y_valid)
  pred_valid = model.predict(X_valid,batch_size=batch_size)

  idx_test = idx_seizure(y_test)
  pred_test = model.predict(X_test,batch_size=batch_size)
  
  th_ = [0.3,0.4,0.5,0.6,0.7,0.8]
  prob_ = [0.5,0.6,0.7,0.8]
  n_ = [sop]
  for th in th_:
    for prob in prob_:
      for n in n_:               
        pred_valid_,y_valid_ = firing_power(pred_valid,y_valid,th,prob,n)
        idx_valid = idx_seizure(y_valid_)
        FPR_v,s_v = performance(pred_valid_, y_valid_, idx_valid) 

        pred_test_,y_test_ = firing_power(pred_test,y_test,th,prob,n)
        idx_test = idx_seizure(y_test_)
        FPR_t,s_t = performance(pred_test_, y_test_, idx_test)         
        
        f = open('results/FP_results_'+type+'.csv', 'a')
        f.write(str(ID)+','+str(sop)+','+str(d)+','+str(w)+','+str(th)+','+str(prob)+','+str(n)+','+str(FPR_v)+','+
                str(s_v)+','+str(FPR_t)+','+str(s_t)+'\n')
        f.close()
        
def validation_point(X_train,y_train,X_valid,y_valid,X_test,y_test,type,sop):
  """
  Inputs
    X_train - train feature matrix 
    y_train - train target matrix
    X_valid - validation feature matrix 
    y_valid - validation target matrix
    X_test - test feature matrix 
    y_test - test target matrix
    type - string with values 'scalp' or 'inv'
    sop - seizure occurrence period
  Outputs
    sensitivity, specificty, precision, f1 score and weighted accuracy are saved into a file.
  """
  f = open('results/point_results_'+type+'.csv','a')
  
  model = load_model(model_dir+ID+'_'+str(seq_length)+'_'
   +str(sop)+'_'+str(lr)+'_'+str(d)+'_'+str(w)+'_'+str(structure)+'_model.h5')
  

  pred = model.predict(X_train,batch_size = batch_size)
  senst,spect,prect,acct,f1t = metrics(pred,y_train)  
  senst = round(senst,3)
  spect = round(spect,3)
  prect = round(prect,3)
  acct = round(acct,3)
  f1t = round(f1t,3)
  
  pred = model.predict(X_valid_1,batch_size = batch_size)
  sensv,specv,precv,accv,f1v = metrics(pred,y_valid_1)
  sensv = round(sensv,3)
  specv = round(specv,3)
  precv = round(precv,3)
  accv = round(accv,3)
  f1v = round(f1v,3) 
  
  pred = model.predict(X_test,batch_size = batch_size)
  sensts,spects,prects,accts,f1ts = metrics(pred,y_test)  
  sensts = round(sensts,3)
  spects = round(spects,3)
  prects = round(prects,3)
  accts = round(accts,3)
  f1ts = round(f1ts,3) 
  
  f.write(ID+','+str(sop)+','+str(d)+','+str(w)+
          ','+str(senst)+','+str(spect)+','+str(prect)+','+str(acct)+','+str(f1t)+
          ','+str(sensv)+','+str(specv)+','+str(precv)+','+str(accv)+','+str(f1v)+
          ','+str(sensts)+','+str(spects)+','+str(prects)+','+str(accts)+','+str(f1ts)+'\n')  
  f.close()

  
  
"""
Here is a script that exemplifies how we used the above functions to train and evaluate models for 105 patients
"""
if __name__ == "__main__":   
 
  type = 'inv'
  model_dir = 'models/'
  overlap = 1
  predstep = 0 # do not change
  patients = pd.read_csv('info_'+type+'.csv')
  patients = patients.sort('min_dis', ascending=False)
  IDs = list(patients['ID'])
  IDs = [str(i) for i in IDs]
  DIR = 'feat_target_'+type+'/' 
  sop_ = [10,20,30,40]#seizure occurence period, multiply by 5/60 to convert to minutes
  lr_ = [0.001] #values for the learning rate
  d_ = [0.6,0.7,0.8] 
  w_ = [0.1,0.01,0.001] #values for the regularization strength
  seq_length_ = [50]
  structure_ = [[['lstm',200],['lstm',200]]]
#%%
  batch_size = 128 #batch size
  nb_epoch = 60 #number of epochs  
  for ID in IDs:  
    try:
      for structure in structure_:
        for seq_length in seq_length_:
          for sop in sop_:
            
            np.random.seed(14337)
            X,y = read_data(DIR,ID,sop)    #%%                   
            X,y = check_nan(X,y) #check for rows with missing values and remove them
            y[y==3]=1
            y[y==4]=1
  
            y = np_utils.to_categorical(y)
            y = y[:,1:]
            X = scale(X) #normalize the data
            idx = idx_seizure(y)
            
            X_train, X_valid, X_test, y_train, y_valid, y_test = separate_data(X,y,0.6,0.2,idx)            
            del X,y
            X_test,y_test = create_seq(X_test,y_test,seq_length,overlap,predstep)
            X_valid_1,y_valid_1 = create_seq(X_valid,y_valid,seq_length,overlap,predstep)
            
            idx_valid = idx_seizure(y_valid)
            idx_test  = idx_seizure(y_test)
            idx_train = idx_seizure(y_train)
            
            try:
              X_valid_2,y_valid_2 = create_seq_train(X_valid,y_valid,seq_length,overlap,predstep,idx_valid)
            except:
              X_valid_2,y_valid_2 = X_valid_1,y_valid_1
            X_train,y_train = create_seq_train(X_train,y_train,seq_length,overlap,predstep,idx_train)
            n_preict = len(y_train[y_train[:,1]==1])
            n_int = len(y_train[y_train[:,0]==1])  
            
            n_preict_valid = len(y_valid[y_valid[:,1]==1])
            n_int_valid = len(y_valid[y_valid[:,0]==1])  
            
            n_seiz = len(idx)
            n_train_seiz = len(idx_train)
            n_valid_seiz = len(idx_valid)
            n_test_seiz = len(idx_test)
            valid_dur = len(y_valid)*5/3600.
            test_dur = len(y_test)*5/3600.
            valid_dur = round(valid_dur,1)
            test_dur = round(test_dur,1)
            print('ID: '+ID)
            print('Nº seizures = ' + str(n_seiz))
            print('Nº train seizures = ' + str(n_train_seiz))
            print('Nº validate seizures = ' + str(n_valid_seiz))
            print('Nº test seizures = ' + str(n_test_seiz))
            print('Valid duration = ' + str(valid_dur))
            print('Test duration = ' + str(test_dur))
  
            if n_test_seiz == 0:
              break

            for lr in lr_:
              for d in d_:
                for w in w_:
                  if os.path.isfile(model_dir+ID+'_'+str(seq_length)+'_'
                   +str(sop)+'_'+str(lr)+'_'+str(d)+'_'+str(w)+'_'+str(structure)+'_model.h5'):
                    print("model exists")
                    continue
                  
                  if d==0 and w == 0: 
                    continue
                  model = build_net(structure,X_train,y_train,X_test,y_test,lr,d,w)
                  validation_point(X_train,y_train,X_valid_1,y_valid_1,X_test,y_test,type,sop)
                  validation(X_valid_1,y_valid_1,X_test,y_test,type,sop)
                  pass
            del X_train,y_train,X_valid_1,y_valid_1,X_test,y_test,X_valid_2,y_valid_2
    except:
        print('error: ',ID)
