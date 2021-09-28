import numpy as np
import matplotlib.pyplot as plt
from h5py import File

with File("Data/PoroTomo_iDAS16043_160314104148.decimated.mat", 'r') as file:
    dt = file["dt"][()][0,0]
    nch = int(file["nch"][()][0,0])
    npts =  int(file["npts"][()][0,0])
    t0 =  file["t0"][()][0,0]
    data = file["data"][()]
    
data = data[:, 1500:2500]

chan, x, y, z = np.loadtxt("Data/Surface_DAS_DTS_UTM_coordinates.csv", delimiter=',',skiprows=2, unpack=True)

xyselector = np.logical_and(x>10000,y>10000)

data = data[xyselector]
chan = chan[xyselector]
x = x[xyselector]
y = y[xyselector]
z = z[xyselector]


azimuth = np.rad2deg(np.arctan2(x[1:]-x[:-1],y[1:]-y[:-1]))
corners = np.hstack([0, np.abs(azimuth[1:]-azimuth[:-1]) > 10, 0])
near_corners_selector = np.array([True for i in range(len(corners))])
for i, c in enumerate(corners):
    if c == 1:
        near_corners_selector[i-5:i+6] = False
        
azimuth = np.hstack([azimuth, azimuth[-1]]) # add an azimuth for the final channel


data = data[near_corners_selector]
chan = chan[near_corners_selector]
x = x[near_corners_selector]
y = y[near_corners_selector]
z = z[near_corners_selector]
azimuth = azimuth[near_corners_selector]


from scipy.signal import butter, filtfilt, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype='low')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=5, axis=-1):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


data_r = butter_lowpass_filter(data, 10, 1/dt)[:,::5]


median_log_max_amp = np.log10(np.median(np.sort(np.abs(data_r), axis=1)[:,-10:], axis=1))

q1 = np.quantile(median_log_max_amp,0.01)
q99 = np.quantile(median_log_max_amp,0.99)


median_selector = np.logical_and(median_log_max_amp>q1, median_log_max_amp<q99)




# In[12]:


data = data[median_selector]
chan = chan[median_selector]
x = x[median_selector]
y = y[median_selector]
z = z[median_selector]
azimuth = azimuth[median_selector]


from scipy.spatial import ConvexHull
points = np.array([x,y]).T
hull = ConvexHull(points)
hullpoints = points[hull.vertices]
np.save("Data/convex_hull_points.npy", hullpoints)




corners = np.load("Data/rectangle_corners.npy")
vec1 = corners[:,0]-corners[:,1]
vec2 = corners[:,2]-corners[:,1]
l1 = np.sqrt(np.sum(np.square(vec1)))
l2 = np.sqrt(np.sum(np.square(vec2)))
t1 = np.linspace(0, l1, 33)
t2 = np.linspace(0, l2, 129)

t1 = (t1[1:]+t1[:-1])/2
t2 = (t2[1:]+t2[:-1])/2

domain_azimuth = np.rad2deg(np.arctan2(vec2[0], vec2[1]))
azimuth_corr = azimuth-domain_azimuth
rac = np.deg2rad(azimuth_corr)



points_corr = points-corners[:,1]
proj_1 = l1*(points_corr@vec1[:,np.newaxis] / np.sum(np.square(vec1))).flatten()
proj_2 = l2*(points_corr@vec2[:,np.newaxis] / np.sum(np.square(vec2))).flatten()


from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline as rbs
import scipy.sparse as sp

cscale = 2

generate_kernels = False

if generate_kernels:
    crv = loadmat("Data/G_32_128.mat")
    G_mat = np.reshape(crv["G_mat"].T, (crv["G_mat"].shape[1], 32,128))
    crvscales = crv["scales"].flatten()
    cvtscaler = (2.0**(cscale*crvscales))
    G1_rows = []
    G1_cols = []
    G1_vals = []
    G2_rows = []
    G2_cols = []
    G2_vals = []
    for j in range(G_mat.shape[0]):
        frame = rbs(t1,t2,G_mat[j])
        G1_col = (np.sin(rac)**2*frame.ev(proj_1,proj_2, dx=1) + 
                  np.sin(2*rac)*frame.ev(proj_1,proj_2, dy=1)/2)
        G1_selector = np.nonzero(np.log10(np.abs(G1_col))-np.max(np.log10(np.abs(G1_col))) > -2)[0] # if <1% max, get rid of it
        G1_rows += [i for i in G1_selector]
        G1_cols += [j for i in G1_selector]
        G1_vals += [G1_col[i]/cvtscaler[j] for i in G1_selector]
        
        
        G2_col = (np.cos(rac)**2*frame.ev(proj_1,proj_2, dy=1) + 
                  np.sin(2*rac)*frame.ev(proj_1,proj_2, dx=1)/2)
        G2_selector = np.nonzero(np.log10(np.abs(G2_col))-np.max(np.log10(np.abs(G2_col))) > -2)[0] # if <1% max, get rid of it
        G2_rows += [i for i in G2_selector]
        G2_cols += [j for i in G2_selector]
        G2_vals += [G2_col[i]/cvtscaler[j] for i in G2_selector]
        
    
    G1 = sp.coo_matrix((G1_vals, (G1_rows, G1_cols)), shape=(len(x), G_mat.shape[0]))
    G2 = sp.coo_matrix((G2_vals, (G2_rows, G2_cols)), shape=(len(x), G_mat.shape[0]))
    G = sp.hstack([G1,G2])
    Gn = np.sqrt(G.power(2).sum()/G.nnz - G.mean()**2)
    G = G / Gn
    Gevp = G_mat / cvtscaler[:,np.newaxis,np.newaxis] / Gn
    sp.save_npz("Data/G.npz", G)
    np.save("Data/G_evp.npy", Gevp)
    
if not generate_kernels:
    G = sp.load_npz("Data/G.npz")
    Gevp = np.load("Data/G_evp.npy")
        


# In[18]:


G = G.tocsc()astype(np.float32)


# In[ ]:


import pywt
import h2o4gpu
from tqdm import tqdm
import os

wvt = 'db8'

wvt_data = np.array([np.hstack(pywt.wavedec(d, wvt)) for d in data]).astype(np.float32).T
ncoefs, nstations = wvt_data.shape

actually_do_reconstruction = True

if actually_do_reconstruction:
    #model = celer.LassoCV(cv=5, n_alphas=20, max_epochs=50000, fit_intercept=False)
    model = h2o4gpu.ElasticNetH2O(n_threads=8, fit_intercept=False, n_lambdas=25, n_folds=5, alphas=[1.0])
    rows = []
    cols = []
    vals = []
    norms = []
    alphas = []

    os.makedirs("Results/", exist_ok=True)
    with tqdm(total=ncoefs) as pbar1:
        for i, d in enumerate(wvt_data):
            norm = np.std(d)
            norms += [norm]
            dn = d / norm
            model.fit(G, dn)
            vi = np.nonzero(model.coef_)[0]
            v = model.coef_[vi]
            rows += [vii for vii in vi] # 
            cols += [i for j in range(len(vi))] # the col index should all be the ith wavelet
            vals += [vv*norm for vv in v]
            #alphas += [model.alpha_]  
            alphas += [1e-2]
            pbar1.update(1)
        
    res = sp.coo_matrix((vals, (rows, cols)), shape=(G.shape[1], ncoefs))
    sp.save_npz("Results/fit.npz", res)
    np.save("Results/norm.npy", norms)
    np.save("Results/alphas.npy", alphas)

