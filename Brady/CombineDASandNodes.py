#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pywt
import subprocess
from sklearn.model_selection import KFold
import scipy.sparse as sp
wvt = 'db12'


# In[44]:


def reconstruction(w, wvt_lens, wvt):
    starts = np.hstack([0,np.cumsum(wvt_lens)])
    wcoef = [w[starts[i]:starts[i+1]] for i in range(len(wvt_lens))]
    return pywt.waverec(wcoef, wvt)

# In[4]:


das_data = np.load("Data/filtdas.npy")


# In[5]:


nodal_data = np.load("Data/nodal_rotated_data.npy")
nodal_data_x = nodal_data[0]
nodal_data_y = nodal_data[1]


# In[6]:


best_offset = 17 # determined by fitting DAS reconstruction using all nodes to the DAS data


# In[7]:


das_data = das_data[:, (150-best_offset) : (550-best_offset)]
nodal_data_x = nodal_data_x[:, 150:550]
nodal_data_y = nodal_data_y[:, 150:550]


# In[67]:


das_stds = np.std(das_data, axis=1)
nodal_x_stds = np.std(nodal_data_x, axis=1)
nodal_y_stds = np.std(nodal_data_y, axis=1)


# In[70]:


das_data = das_data / das_stds[:, np.newaxis]
nodal_data_x = nodal_data_x / nodal_x_stds[:, np.newaxis]
nodal_data_y = nodal_data_y / nodal_y_stds[:, np.newaxis]


# In[71]:


das_wvt_data = np.array([np.hstack(pywt.wavedec(d, wvt)) for d in das_data])
nodal_wvt_data_x = np.array([np.hstack(pywt.wavedec(d, wvt)) for d in nodal_data_x])
nodal_wvt_data_y = np.array([np.hstack(pywt.wavedec(d, wvt)) for d in nodal_data_y])


# In[49]:


wvt_tmp = pywt.wavedec(das_data[0], wvt)
wvt_lens = [len(wc) for wc in wvt_tmp]


# In[10]:


Gn_das = 5.487292209780819e-06
Gn_nodes = 0.0003260777891832482


# In[11]:


G_das = np.load("Data/G.npy")*Gn_das
G_nodes = np.load("Data/nodal_G.npy")*Gn_nodes


# In[21]:


np.random.seed(94899109)
permute_idx = np.random.permutation(nodal_data_x.shape[0])
kf = KFold(n_splits=5)


# In[79]:


# perform only node inversions

mse_results_nodes = np.zeros((5,5))

for fold, (train, test) in enumerate(kf.split(permute_idx)):
    for n_train in [1, 2, 3, 4, 5]:
        if n_train != 5:
            train_subset = permute_idx[train[:(n_train*(len(train)//5))]]
        else:
            train_subset  = permute_idx[train]
            
        G_nodes_train = G_nodes[train_subset]
        G_nodes_train_full = np.vstack([np.hstack([G_nodes_train/nodal_x_stds[train_subset][:, np.newaxis], np.zeros(G_nodes_train.shape)]),
                                  np.hstack([np.zeros(G_nodes_train.shape), G_nodes_train/nodal_y_stds[train_subset][:, np.newaxis]])])

        G_nodes_test= G_nodes[test]
        G_nodes_test_full = np.vstack([np.hstack([G_nodes_test/nodal_x_stds[test][:, np.newaxis], np.zeros(G_nodes_test.shape)]),
                                  np.hstack([np.zeros(G_nodes_test.shape), G_nodes_test/nodal_y_stds[test][:, np.newaxis]])])
        
        Gn_nodes_full = np.std(G_nodes_train_full)
        G_nodes_train_full = G_nodes_train_full / Gn_nodes_full 
        G_nodes_test_full = G_nodes_test_full / Gn_nodes_full
        nodal_wvt_data_train_full = np.vstack([nodal_wvt_data_x[train_subset],
                                         nodal_wvt_data_y[train_subset]])
        np.save("tmp/data.npy", nodal_wvt_data_train_full)
        np.save("tmp/G.npy",  G_nodes_train_full)
        subprocess.run(f"/home/jmuir/.local/bin/julia combinedinversion.jl {fold}_{n_train}_nodes", shell=True, env={"JULIA_NUM_THREADS" : "12"})
        res = sp.load_npz(f"Combined_Results/combined_results_{fold}_{n_train}_nodes.npz")
        nodal_wvt_data_test_pred = G_nodes_test_full @ res
        nodal_data_test_pred = np.real(np.array([reconstruction(w, wvt_lens, wvt) for w in nodal_wvt_data_test_pred]))

        
        nodal_data_test_full = np.vstack([nodal_data_x[test],
                                              nodal_data_y[test]])
        
        mse = np.mean(np.square(nodal_data_test_full - nodal_data_test_pred))
        print(f"Fold {fold}, percent_train {20*n_train}, mse = {mse}")
        mse_results_nodes[n_train-1, fold] = mse


# In[80]:


np.save("Combined_Results/node_mse.npy", mse_results_nodes)


# In[ ]:


# perform only node inversions

mse_results_nodes_das = np.zeros((5,5))

for fold, (train, test) in enumerate(kf.split(permute_idx)):
    for n_train in [1, 2, 3, 4, 5]:
        #This part is the same as before
        if n_train != 5:
            train_subset = permute_idx[train[:(n_train*(len(train)//5))]]
        else:
            train_subset  = permute_idx[train]
            
        G_nodes_train = G_nodes[train_subset]
        G_nodes_train_full = np.vstack([np.hstack([G_nodes_train/nodal_x_stds[train_subset][:, np.newaxis], np.zeros(G_nodes_train.shape)]),
                                  np.hstack([np.zeros(G_nodes_train.shape), G_nodes_train/nodal_y_stds[train_subset][:, np.newaxis]])])

        G_nodes_test= G_nodes[test]
        G_nodes_test_full = np.vstack([np.hstack([G_nodes_test/nodal_x_stds[test][:, np.newaxis], np.zeros(G_nodes_test.shape)]),
                                  np.hstack([np.zeros(G_nodes_test.shape), G_nodes_test/nodal_y_stds[test][:, np.newaxis]])])
        
        nodal_wvt_data_train_full = np.vstack([nodal_wvt_data_x[train_subset],
                                         nodal_wvt_data_y[train_subset]])
        
        
        
        #Now we add DAS as well
        G_das_train = G_das/das_stds[:, np.newaxis]
        wvt_data_train_full = np.vstack([nodal_wvt_data_train_full, 
                                         das_wvt_data])
        
        G_train_full = np.vstack([G_nodes_train_full, 
                                  G_das_train])

        Gn_train_full = np.std(G_train_full)
        G_train_full = G_train_full / Gn_train_full 
        G_nodes_test_full = G_nodes_test_full / Gn_train_full

        np.save("tmp/data.npy", wvt_data_train_full)
        np.save("tmp/G.npy",  G_train_full)
        
        #And then the rest is the same again
        subprocess.run(f"/home/jmuir/.local/bin/julia combinedinversion.jl {fold}_{n_train}_nodes_das", shell=True, env={"JULIA_NUM_THREADS" : "12"})
        res = sp.load_npz(f"Combined_Results/combined_results_{fold}_{n_train}_nodes_das.npz")
        nodal_wvt_data_test_pred = G_nodes_test_full @ res
        nodal_data_test_pred = np.real(np.array([reconstruction(w, wvt_lens, wvt) for w in nodal_wvt_data_test_pred]))

        
        nodal_data_test_full = np.vstack([nodal_data_x[test],
                                              nodal_data_y[test]])
        
        mse = np.mean(np.square(nodal_data_test_full - nodal_data_test_pred))
        print(f"Fold {fold}, percent_train {20*n_train}, mse = {mse}")
        mse_results_nodes_das[n_train-1, fold] = mse


# In[ ]:


np.save("Combined_Results/node_das_mse.npy", mse_results_nodes_das)
