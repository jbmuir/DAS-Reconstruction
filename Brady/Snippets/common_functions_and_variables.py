from scipy.signal import filtfilt, butter, hilbert, find_peaks
import cartopy.feature as feature
import cartopy.crs as ccrs
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.stats import linregress
import matplotlib.gridspec as gridspec


proj = ccrs.UTM(11)
geo = ccrs.Geodetic()
pc = ccrs.PlateCarree()

# Getting Box

states_provinces = feature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none',
    edgecolor='black')

borders = feature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none',
    edgecolor='black')


coastline = feature.GSHHSFeature(
    scale="full",
    levels=[1, 2],
    edgecolor='black')

def trlatlon(tr):
    network, station, location, channel = tr.id.split('.')
    chainfo = inventory.select(network=network,station=station,location=location,channel=channel)[0][0][0]
    tr.stats.latitude = chainfo.latitude
    tr.stats.longitude = chainfo.longitude

def box_selector(tr, minlon, maxlon, minlat, maxlat):
    trlatlon(tr)
    if (minlon < tr.stats.longitude < maxlon) and (minlat < tr.stats.latitude < maxlat):
        return True
    else:
        return False
        
def reconstruction(w, wvt_lens, wvt):
    starts = np.hstack([0,np.cumsum(wvt_lens)])
    wcoef = [w[starts[i]:starts[i+1]] for i in range(len(wvt_lens))]
    return pywt.waverec(wcoef, wvt)


def plot_sheet(dataset1, dataset2=None, savename=None, d1color='k',d2color='r'):
    """
    We are assuming that dataset1 and dataset2 are aligned 
    """
    nrows = 11
    ncols = 3
    letter = (8.5, 11)
    assert len(dataset1) <= nrows*ncols
    if dataset2 is not None:
        assert len(dataset1) == len(dataset2)

    def i2rc(i):
        return (i//ncols, i % ncols)

    #make sure that the time axis is plotted at the bottom of each column
    lr = [len(dataset1) // ncols for i in range(ncols)]
    for i in range(len(dataset1) % ncols):
        lr[i] += 1

    fig = plt.figure(figsize=letter)
    gs = gridspec.GridSpec(nrows, ncols)

    for (i, tr) in enumerate(dataset1):
        r, c = i2rc(i)
        ax = fig.add_subplot(gs[r, c])
        ax.plot(tr.times(), tr.data, color=d1color)
        if dataset2 is not None:
            ax.plot(dataset2[i].times(), dataset2[i].data, color=d2color)

        ax.text(1.0, 1.0, tr.id,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
        ax.yaxis.set_ticks([])
        if r != lr[c]-1:
            ax.xaxis.set_ticks([])

        for loc, spine in ax.spines.items():
            if loc == "bottom" and r == lr[c]-1:
                spine.set_position(('outward', 10))  # outward by 10 points
                spine.set_smart_bounds(True)
                ax.set_xlabel("Time (s)")
            else:
                spine.set_color('none')  # don't draw spine

    if savename is not None:
        fig.savefig(savename)


def filter_data(data, fmin, fmax, fs=1.0, axis=-1):
    b, a = butter(2, [fmin, fmax], btype='bandpass', fs=fs)
    return filtfilt(b, a, data, axis=axis)


def gaussian_filter(fc, alpha=20):
    stdev = fc / np.sqrt(alpha)
    return (fc-stdev, fc+stdev)


def laplacian_of_spline(s, ye, xe, yw, xw):
    d2sdx2 = s(ye, xe, dx=2)
    d2sdy2 = s(ye, xe, dy=2)
    return d2sdx2 + d2sdy2

def laplacian(A, ye, xe, yw, xw):
    spline = rbs(ye, xe, A)
    return laplacian_of_spline(spline, ye, xe, yw, xw)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def find_maximum(trace, period, fs=1.0, near_to=None):
    width = period*fs  # peaks must be seperated by at least this
    envelope = np.abs(hilbert(trace))
    peaks, _ = find_peaks(envelope, distance=width)
    if near_to is not None:
        sel = np.argmin(np.abs(peaks-near_to))
    else:
        sel = np.argmax(envelope[peaks])
    return peaks[sel]


def compute_phase_velocity(x, xl, period, fs=1.0, near_to=None, w=1.0):
    t = np.array([fs*i for i in range(len(x))])
    s = ius(t, x)
    x2t = s(t, nu=2)
    m = find_maximum(x, period, fs=fs, near_to=near_to)
    mm = max(int(m-w*fs*period), 0) # max required to avoid looping back to -ve indices with numpy
    mp = int(m+w*fs*period)
    res = linregress(xl[mm:mp], x2t[mm:mp])
    return (np.sqrt(res.slope), res.rvalue, m)


def cascade_compute_phase_velocity(xf, xlf, periods, near_to = None, fs=1.0, w=1.0):
    res = []
    res += [np.array([*compute_phase_velocity(xf[0],
                                              xlf[0], periods[0], near_to = near_to, fs=fs, w=w)])]
    for i in range(1, len(periods)):
        res += [np.array([*compute_phase_velocity(xf[i], xlf[i],
                                                  periods[i], near_to=res[i-1][2], fs=fs, w=w)])]

    return np.array(res)


def get_phase_velocities(xfxy, xlfxy, periods, fs=1.0, w=1.0):
    I, J, _, _  = xfxy.shape
    resinit = np.array([[find_maximum(xfxy[i,j,0], periods[0], fs=fs) for j in range(J)] for i in range(I)])
    nt0 = np.median(resinit)
    return np.array([[cascade_compute_phase_velocity(xfxy[i,j], xlfxy[i,j], periods, near_to=nt0, fs=fs, w=w) 
                for j in range(J)] for i in range(I)])


from matplotlib.collections import LineCollection

def particle_motion_plot(ds, start, end):
    fig = plt.figure(figsize=(15,12))
    gs = gridspec.GridSpec(3,2)
    ax_list = [fig.add_subplot(ss) for ss in gs]
    maxamp = np.max(np.abs(np.array([tr.data for tr in ds])))
    istart = np.argmin(np.abs(ds[0].times()-start))
    iend = np.argmin(np.abs(ds[0].times()-end))
    
    def plot_channel(ax, channel):
        tr = ds.select(channel=channel)[0]
        wtimes = tr.times()[istart:iend]
        norm=plt.Normalize(0, len(wtimes))
        ax.plot(tr.times(), tr.data, 'k')
        points = np.array([wtimes, tr.data[istart:iend]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('cool'), norm=norm)
        lc.set_array(np.array(range(len(wtimes))))
        lc.set_linewidth(3)
        ax.add_collection(lc)
        ax.set_ylabel(channel)
        ax.set_ylim(-maxamp,maxamp)

    plot_channel(ax_list[0],"HNZ")
    plot_channel(ax_list[2],"HNN")
    plot_channel(ax_list[4],"HNE")
    
    def plot_particle_motion(ax, chan1, chan2):
        tr1 = ds.select(channel=chan1)[0]
        tr2 = ds.select(channel=chan2)[0]
        wtimes = tr.times()[istart:iend]
        norm=plt.Normalize(0, len(wtimes))
        points = np.array([tr1.data[istart:iend], tr2.data[istart:iend]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.get_cmap('cool'), norm=norm)
        lc.set_array(np.array(range(len(wtimes))))
        lc.set_linewidth(3)
        ax.add_collection(lc)
        pm = np.max(np.abs(points))
        ax.set_xlim(-pm, pm)
        ax.set_ylim(-pm,pm)
        ax.set_xlabel(chan1)
        ax.set_ylabel(chan2)
        ax.set_aspect('equal')

    plot_particle_motion(ax_list[1], "HNZ","HNE")
    plot_particle_motion(ax_list[3], "HNZ","HNN")
    plot_particle_motion(ax_list[5], "HNE","HNN")