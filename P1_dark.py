#%% Import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from secsy import get_SECS_B_G_matrices
import EZIE.inversion_code.cases_new as cases
from importlib import reload
import sys
import copy
from matplotlib.patches import ConnectionPatch
from matplotlib import patches
from pysymmetry.visualization.polarsubplot import Polarsubplot
import matplotlib.patheffects as mpe
from scipy.interpolate import splrep, BSpline

sys.path.append('/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lib')
import P1_lib as P1

#%% Figure settings

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

#%% Define case

reload(cases) 

info = cases.cases_all['case_2_K0582_mlt03']
info['clevels'] = np.linspace(-1000, 1000, 21)
info['central_lat'] = 68
info['segment'] = 2

#%% Some info

OBSHEIGHT = info['observation_height']

d2r = np.pi / 180

LRES = 40. # spatial resolution of SECS grid along satellite track
WRES = 20. # spatial resolution perpendicular to satellite tarck
wshift = info['wshift'] # shift center of grid wshift km to the right of the satellite (rel to velocity)
timeres = info['timeres']
DT  = info['DT'] # size of time window [min]
RI  = (6371.2 + 110) * 1e3 # SECS height (m)
RE  = 6371.2e3

ground_file = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/articles/P1/Gamera_data/ground/case_2/gamera_dBs_00km_2016-08-09T09:18:00.txt'

info['clevels'] = np.linspace(np.min(info['clevels']), np.max(info['clevels']), 40)

#%% Show MHD

#mlt_shift = 0
mlt_shift = 3

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

#projection, grid = P1.get_projection(data, 800, 1400, rs, tm, RI, LRES, WRES, wshift)
projection, grid = P1.get_projection(data, rs, tm, RI, LRES, WRES, wshift)

lat_mhd, lon_mhd = np.meshgrid(np.linspace(50, 90, 500), np.linspace(0, 360, 2000))

mhdBu =  info['mhdfunc'](lat_mhd.flatten(), lon_mhd.flatten() + info['mapshift'], fn = info['mhd_B_fn'])

plt.ioff()
plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 15))
ax = plt.gca()
pax = Polarsubplot(ax, minlat = 50, linestyle = '--', linewidth = .5, color = 'grey')

cc = pax.tricontourf(lat_mhd.flatten(), ((lon_mhd/15+mlt_shift)%24).flatten(), 
                     mhdBu, cmap='bwr', levels=np.linspace(-np.max(abs(mhdBu)), np.max(abs(mhdBu)), 41))

for i, c in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
    pax.plot(data['lat_{}'.format(i+1)], (data['lon_{}'.format(i+1)]/15+mlt_shift)%24, linestyle='--', linewidth=1, color=c, zorder=9)    
    #pax.plot(data['lat_{}'.format(i+1)][70:225], (data['lon_{}'.format(i+1)][70:225]/15+mlt_shift)%24, linewidth=4, color=c, zorder=9)
    pax.plot(np.array(obs['lat'][0+i*81:81+i*81]), (np.array(obs['lon'][0+i*81:81+i*81])/15+mlt_shift)%24, linewidth=4, color=c, zorder=9)

pax.plot(grid.lat[0, :], (grid.lon[0, :]/15+mlt_shift)%24, color='k', linewidth=1.5, zorder=10)
pax.plot(grid.lat[-1, :], (grid.lon[-1, :]/15+mlt_shift)%24, color='k', linewidth=1.5, zorder=10)
pax.plot(grid.lat[:, 0], (grid.lon[:, 0]/15+mlt_shift)%24, color='k', linewidth=1.5, zorder=10)
pax.plot(grid.lat[:, -1], (grid.lon[:, -1]/15+mlt_shift)%24, color='k', linewidth=1.5, zorder=10)

for i in range(0, grid.shape[0], 6):
    pax.plot(grid.lat[i, :], (grid.lon[i, :]/15+mlt_shift)%24, color='grey', linewidth=1, zorder=8)

for i in range(0, grid.shape[1], 6):
    pax.plot(grid.lat[:, i], (grid.lon[:, i]/15+mlt_shift)%24, color='grey', linewidth=1, zorder=8)

args = dict(va='center', ha='center', fontsize=24, transform=ax.transAxes)
ax.text(0.5, 0, '0 MLT', **args)
ax.text(1, 0.5, '06', **args)
ax.text(0.5, 1, '12', **args)
ax.text(0, 0.5, '18', **args)

ofs = 0.08
ax.text(0.5+0.08, 0.5-0.08, '80', **args, color='k')
ax.text(0.5+0.16, 0.5-0.16, '70', **args, color='k')
ax.text(0.5+0.24, 0.5-0.24, '60', **args, color='k')
ax.text(0.485+0.32, 0.515-0.32, '50', **args, color='k')

cax = fig.add_axes([0.2, 0.06, 0.61, 0.015])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([-1000, -500, 0, 500, 1000])
#cax.set_xticklabels([])
cax.text(0.5, -2, '$\Delta$B$_r$ [nT]', fontsize=24, va='top', ha='center', transform=cax.transAxes)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/overview_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/overview_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Grid plot - Disputas

mlt_shift = 3
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
projection, grid = P1.get_projection(data, rs, tm, RI, LRES, WRES, wshift)

ximin, ximax = grid.xi_mesh.min(), grid.xi_mesh.max()
etamin, etamax = grid.eta_mesh.min(), grid.eta_mesh.max()

dxi = ximax - ximin
deta = etamax - etamin

plt.style.use('dark_background')
plt.ioff()
plt.figure(figsize=(15, 15))
ax = plt.gca()

# grid
for i in range(grid.xi_mesh.shape[0]):
    plt.plot(grid.xi_mesh[i, :], grid.eta_mesh[i, :], 'grey', linewidth=0.6)
for i in range(grid.xi_mesh.shape[1]):
    plt.plot(grid.xi_mesh[:, i], grid.eta_mesh[:, i], 'grey', linewidth=0.6)

plt.plot([ximin, ximin, ximax, ximax, ximin],
         [etamin, etamax, etamax, etamin, etamin], color='w', linewidth=2)

# sphere
for lat in np.arange(0, 90+5, 5):
    xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 1000), np.ones(1000)*lat)
    plt.plot(xi, eta, color='grey')
for lon in np.arange(0, 360+20, 20):
    xi, eta = grid.projection.geo2cube(np.ones(1000)*lon, np.linspace(0, 90, 1000))
    plt.plot(xi, eta, color='grey')

# tracks
f = (data.index >= t0) & (data.index <= t1)
for i, c in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
    xi, eta = grid.projection.geo2cube(data.loc[f, 'lon_{}'.format(i+1)], data.loc[f, 'lat_{}'.format(i+1)])
    plt.plot(xi, eta, color='k', linewidth=12)
    plt.plot(xi, eta, color='w', linewidth=10)
    plt.plot(xi, eta, color=c, linewidth=8)

plt.arrow(0.485, 0.8, 0, 0.1, width=.01, color='w', transform=ax.transAxes, zorder = 10)
ax.text(.5, .96, 'Satellite direction', ha='center', fontsize=20, transform=ax.transAxes)

#sticks
c_s = 'w'

plt.plot([grid.xi_mesh[0, 12], grid.xi_mesh[0, 36]], [grid.eta_mesh[12, 0]]*2, color=c_s, linewidth=3, zorder=8)
plt.plot([grid.xi_mesh[0, 12], grid.xi_mesh[0, 36]], [grid.eta_mesh[12, 0]]*2, color='k', linewidth=6, zorder=7)
for i in range(5):
    plt.plot([grid.xi_mesh[0, 12+i*5]]*2, [grid.eta_mesh[11, 0], grid.eta_mesh[13, 0]], color=c_s, linewidth=3, zorder=8)
    plt.plot([grid.xi_mesh[0, 12+i*5]]*2, [grid.eta_mesh[11, 0], grid.eta_mesh[13, 0]], color='k', linewidth=6, zorder=7)

plt.plot([grid.xi_mesh[0, 38]]*2, [grid.eta_mesh[15, 0], grid.eta_mesh[97, 0]], color=c_s, linewidth=3, zorder=8)
plt.plot([grid.xi_mesh[0, 38]]*2, [grid.eta_mesh[15, 0], grid.eta_mesh[97, 0]], color='k', linewidth=6, zorder=7)
ddxi = grid.xi_mesh[0,1] - grid.xi_mesh[0,0]
for i in range(9):
    plt.plot([grid.xi_mesh[0, 38]-.6*ddxi, grid.xi_mesh[0, 38]+.6*ddxi], [grid.eta_mesh[15+i*10, 0]]*2, color=c_s, linewidth=3, zorder=8)
    plt.plot([grid.xi_mesh[0, 38]-.6*ddxi, grid.xi_mesh[0, 38]+.6*ddxi], [grid.eta_mesh[15+i*10, 0]]*2, color='k', linewidth=6, zorder=7)

ax.text(.331, .15, '200 km', ha='center', fontsize=15, transform=ax.transAxes)
ax.text(.8, .23, '200 km', ha='center', fontsize=15, transform=ax.transAxes)

plt.xlim([ximin-0.1*dxi, ximax+0.1*dxi])
plt.ylim([etamin-0.1*deta, etamax+0.1*deta])

ax.set_aspect('equal')
ax.set_axis_off()

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/grid.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/grid.pdf', format='pdf', bbox_inches='tight')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/conference/eiscat_2024/presentation/figures/grid.png', bbox_inches='tight')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/conference/eiscat_2024/presentation/figures/grid.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Solve inversion for EZIE data alone

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
projection, grid = P1.get_projection(data, rs, tm, RI, LRES, WRES, wshift)
_, grid2 = P1.get_projection2(data, rs, tm, RI, LRES, WRES, wshift) # used to calculate true j

# Define the area that we are optimizing
xi, eta = grid.projection.geo2cube(obs['lon'], obs['lat'])

xi_max = np.max(xi)
xi_min = np.min(xi)
eta_max = np.max(eta)
eta_min = np.min(eta)

deta = eta_max - eta_min
eta_mid = eta_min + (deta)/2
eta_window = deta/3

LL = P1.get_roughening_matrix(grid)

Q = P1.get_covariance_matrix(obs)

# Hoyer test Kneedle
l1 = 10**0.8787878787878789
l2 = 10**3.245490870668853

m_org, scale = P1.do_inversion(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                    obs['lat'], obs['lon'], 
                    obs['Bu'], obs['Bn'], obs['Be'],
                    grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

R_org =     P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)
R_org_low = P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=10**-5, l2=0)

Cmp_org = P1.get_posterior_model_covariance(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

q = P1.error_covariance_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

#%% Solve inverseion for EZIE data and 1 ground mag

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

row_st = 54 #58
col_st = 27 #17
xi_st = grid.xi_mesh[row_st, col_st]
eta_st = grid.eta_mesh[row_st, col_st]

lat_st = grid.lat_mesh[row_st, col_st]
lon_st = grid.lon_mesh[row_st, col_st]

mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = ground_file)[0]
mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = ground_file)[0]
mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = ground_file)[0]

obs['lat'].append(lat_st)
obs['lon'].append(lon_st)

obs['Be'].append(mhdBe)
obs['Bn'].append(mhdBn)
obs['Bu'].append(mhdBu)

obs['cov_ee'].append(float(1))
obs['cov_nn'].append(float(1))
obs['cov_uu'].append(float(1))
obs['cov_en'].append(float(0))
obs['cov_eu'].append(float(0))
obs['cov_nu'].append(float(0))

r = np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3
r[-1] = 6371.2 * 1e3

Q = P1.get_covariance_matrix(obs)

# Hoyer test Kneedle
l1 = 10**0.09090909090909083
l2 = 10**1.7476896811054083

m_gmag, _ = P1.do_inversion(r, obs['lat'], obs['lon'], 
                         obs['Bu'], obs['Bn'], obs['Be'],
                         grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

R_gmag =        P1.get_resolution_matrix(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)
R_gmag_low =    P1.get_resolution_matrix(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=10**-5, l2=0)

Cmp_gmag = P1.get_posterior_model_covariance(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

#%% Solve inverseion for EZIE data and multiple G-mags

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

n_st = 10

np.random.seed(1337)
row_st = np.ones(n_st).astype(int)
while len(np.unique(row_st)) != n_st:
    row_st = np.random.uniform(10, 98, n_st).astype(int)
col_st = np.ones(n_st).astype(int)
while len(np.unique(col_st)) != n_st:    
    col_st = np.random.uniform(12, 37, n_st).astype(int)

xi_st_multi = grid.xi_mesh[row_st, col_st]
eta_st_multi = grid.eta_mesh[row_st, col_st]

lat_st = grid.lat_mesh[row_st, col_st]
lon_st = grid.lon_mesh[row_st, col_st]

mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = ground_file)[0]
mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = ground_file)[0]
mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = ground_file)[0]

obs['lat'].extend(lat_st)
obs['lon'].extend(lon_st)

obs['Be'].extend(mhdBe)
obs['Bn'].extend(mhdBn)
obs['Bu'].extend(mhdBu)

obs['cov_ee'].extend(np.ones(n_st))
obs['cov_nn'].extend(np.ones(n_st))
obs['cov_uu'].extend(np.ones(n_st))
obs['cov_en'].extend(np.zeros(n_st))
obs['cov_eu'].extend(np.zeros(n_st))
obs['cov_nu'].extend(np.zeros(n_st))

r = np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3
r[-n_st:] = 6371.2 * 1e3

Q = P1.get_covariance_matrix(obs)

# Hoyer test Kneedle
l1 = 10**-0.4545454545454546
l2 = 10**0.9661056602933654

m_multi, _ = P1.do_inversion(r, obs['lat'], obs['lon'], 
                          obs['Bu'], obs['Bn'], obs['Be'],
                          grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

R_multi =       P1.get_resolution_matrix(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)
R_multi_low =   P1.get_resolution_matrix(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=10**-5, l2=0)

Cmp_multi = P1.get_posterior_model_covariance(r, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

#%% Plot initial example 2

#plt.style.use('default')
plt.style.use('dark_background')

i = 2324
PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
vmax = np.max(abs(PSF))

### Ellipse fit
result = P1.do_nonlinear_PSF_fit_2(grid, PSF)

gauss_2d = result.eval(x=grid.xi.flatten(), y=grid.eta.flatten()).reshape(grid.shape)

e1 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     1.1775*result.best_values['sigmax'], 1.1775*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1.2, color='k')

e2 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     3*result.best_values['sigmax'], 3*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1.2, color='k')

marg_xi = np.sum(gauss_2d, axis=0)
xi_mu_e = np.argmax(marg_xi) + 1
q = ((marg_xi - np.min(marg_xi))/np.max(marg_xi)*1000).astype(int)
qq = []
for ii in range(len(q)):
    if q[ii] == 0:
        continue
    for j in range(q[ii]):
        qq.append(ii)
xi_sig_e = np.std(qq)

marg_eta = np.sum(gauss_2d, axis=1)
eta_mu_e = np.argmax(marg_eta) + 1
q = ((marg_eta - np.min(marg_eta))/np.max(marg_eta)*1000).astype(int)
qq = []
for ii in range(len(q)):
    if q[ii] == 0:
        continue
    for j in range(q[ii]):
        qq.append(ii)
eta_sig_e = np.std(qq)
### Ellipse fit done
        
### Olden
xi_mu, eta_mu, xi_sig, eta_sig = P1.get_resolution_metrics(R_org, grid)
xi_mu = xi_mu[i] + 1
eta_mu = eta_mu[i] + 1
xi_sig = xi_sig[i]*np.sqrt(2)
eta_sig = eta_sig[i]*np.sqrt(2)
###


plt.ioff()

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('bwr')
cmap = truncate_colormap(cmap, minval=0.5, maxval=1, n=1000)
fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = cmap, 
                                    clevels=np.linspace(0, vmax, 40), cbar=False,
                                    fg='k')


c_g = 'lightgrey'
for j in range(grid.xi_mesh.shape[0]):
    plt.plot(grid.xi_mesh[j, :], grid.eta_mesh[j, :], color=c_g, linewidth=0.6, zorder=1.1)
for j in range(grid.xi_mesh.shape[1]):
    plt.plot(grid.xi_mesh[:, j], grid.eta_mesh[:, j], color=c_g, linewidth=0.6, zorder=1.1)

plt.plot([ximin, ximin, ximax, ximax, ximin],
         [etamin, etamax, etamax, etamin, etamin], color='w', linewidth=2)

ax.add_patch(e1)
ax.add_patch(e2)

ax.arrow(0.05, 0.5, 0, 0.1, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k', zorder=1.2)
ax.arrow(0.05, 0.5, 0.1, 0, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k', zorder=1.2)
ax.text(0.15, 0.48, 'Cross-track', va='center', ha='center', transform=ax.transAxes, fontsize=16, color='k')
ax.text(0.1, 0.63, 'Along-track', va='center', ha='center', transform=ax.transAxes, fontsize=16, color='k')

    
cax = fig.add_axes([0.19, 0.2, 0.61, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([])
cax.set_xticklabels([])
#cax.text(0.486, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0, 1.1, '0', fontsize=18, va='bottom', ha='center', transform=cax.transAxes, color='k')
cax.text(0.5, 1.1, '|PSF|', fontsize=18, va='bottom', ha='center', transform=cax.transAxes, color='k')
cax.text(1, 1.1, 'max', fontsize=18, va='bottom', ha='center', transform=cax.transAxes, color='k')
    
ax_b = fig.add_axes([0.12, 0.07, 0.78, 0.1])
#ax_b = fig.add_axes([0.01, 0.04, 1.01, 0.1])

ax_r = fig.add_axes([0.9, 0.175, 0.15, 0.665])
ax_r_2 = ax_r.twinx()
    
PSF_xi = np.sum(PSF, axis=0)
PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
PSF_xi_g = np.sum(gauss_2d, axis=0)
PSF_eta_g = -np.sum(gauss_2d, axis=1)

ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='tab:blue', label='Projection of PSF')

ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='tab:blue')
    
ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
i_left, i_right, _ = P1.left_right(PSF_xi)
ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15, label='FWHM (This study)')
    
i_left, i_right, _ = P1.left_right(-PSF_eta)
ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    


#ax_b.plot([xi_mu]*2, [0, np.max(PSF_xi)], color='tab:green')
ax_b.fill_between([xi_mu-1.1775*xi_sig, xi_mu+1.1775*xi_sig], [0, 0], [np.max(PSF_xi)]*2, color='tab:green', alpha=0.4, zorder=-1, label='Oldenborger')

#ax_r_2.plot([0, np.min(PSF_eta)], [eta_mu]*2, color='tab:green')
ax_r_2.fill_between([np.min(PSF_eta), 0], [eta_mu-1.1775*eta_sig]*2, [eta_mu+1.1775*eta_sig]*2, color='tab:green', alpha=0.4, zorder=-1)

#ax_b.plot([xi_mu_e]*2, [0, np.max(PSF_xi)], '--', color='tab:orange')
ax_b.fill_between([xi_mu_e-1.1775*xi_sig_e, xi_mu_e+1.1775*xi_sig_e], [0, 0], [np.max(PSF_xi)]*2, color='tab:orange', alpha=0.4, zorder=-1, label='2D Gaussian')

#ax_r_2.plot([0, np.min(PSF_eta)], [eta_mu_e]*2, '--', color='tab:orange')
ax_r_2.fill_between([np.min(PSF_eta), 0], [eta_mu_e-1.1775*eta_sig_e]*2, [eta_mu_e+1.1775*eta_sig_e]*2, color='tab:orange', alpha=0.4, zorder=-1)

ax_b.legend(ncol=2, fontsize=13, bbox_to_anchor=[1.3, 0.8])

ax_b.set_xlim(1, grid.shape[1])
ax_r_2.set_ylim(1, grid.shape[0])
    
ax_b.spines.right.set_visible(False)
ax_b.spines.left.set_visible(False)
ax_b.spines.top.set_visible(False)
    
ax_r.spines.bottom.set_visible(False)
ax_r.spines.left.set_visible(False)
ax_r.spines.top.set_visible(False)
ax_r_2.spines.bottom.set_visible(False)
ax_r_2.spines.left.set_visible(False)
ax_r_2.spines.top.set_visible(False)
    
ax_b.set_yticks([])
ax_b.set_yticklabels([])
    
ax_r.set_xticks([])
ax_r.set_xticklabels([])
ax_r.set_yticks([])
ax_r.set_yticklabels([])

ax_b.set_xlabel('Grid cell', fontsize=15)
ax_r_2.set_ylabel('Grid cell', fontsize=15)
    
#ax.scatter(grid.xi[col, i], grid.eta[col, i], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
row = int(i/48)
col = i%48    
ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
ax.text(0.95, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12, color='k')
ax.text(0.95, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12, color='k')
ax.text(0.95, 0.35-0.02, '60$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12, color='k')
ax.text(0.95, 0.08-0.02, '50$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12, color='k')

con1 = ConnectionPatch(xyA = (0.1, 0.039), coordsA = ax.transData,
                      xyB = (-0.04, 0.14), coordsB = ax.transData,
                      arrowstyle = 'Simple, tail_width=0.2, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=-.18', zorder=1.2)

con2 = ConnectionPatch(xyA = (-0.04, 0.14), coordsA = ax.transData,
                      xyB = (0.1, 0.039), coordsB = ax.transData,
                      arrowstyle = 'Simple, tail_width=0.1, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=.18', zorder=1.2)

ax.add_artist(con1)
ax.add_artist(con2)

ax.text(0.84, 0.7, 'East/West\ndirection', va='center', ha='center', transform=ax.transAxes, fontsize=16, color='k')

## remove whitespace
#plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/ecample_init_small_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_init_small_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot solution - Bigger arrows - Disputas

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare_disputas(m_org, data, grid, grid2, 4, t0, t1, info, OBSHEIGHT, RI, RE)

axs[5].quiver(0.004, -0.192, 1e3, 0, linewidth = 2, scale=1e4, zorder=40, color='white', clip_on=False)
axs[5].text(0, -0.185, '1 [A/m]', va = 'top', ha = 'right', zorder = 101, size = 30)

axs[0].text(-.05, .5, 'Gamera (MHD)', va='center', ha='center', rotation=90, transform=axs[0].transAxes, fontsize=35)
axs[3].text(-.05, .5, 'EZIE', va='center', ha='center', rotation=90, transform=axs[3].transAxes, fontsize=35)

axs[1].text(0.95+0.1, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=axs[1].transAxes, fontsize=30, color='w')
axs[1].text(0.95+0.1, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=axs[1].transAxes, fontsize=30, color='w')
axs[1].text(0.95+0.1, 0.36-0.02, '60$^{\circ}$', va='center', ha='center', transform=axs[1].transAxes, fontsize=30, color='w')
axs[1].text(0.95+0.1, 0.11-0.02, '50$^{\circ}$', va='center', ha='center', transform=axs[1].transAxes, fontsize=30, color='w')

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/example_bigger_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/example_bigger_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot both solutions - Bigger arrows

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare_2_dark(m_org, m_gmag, data, grid, grid2, 4, t0, t1, info, OBSHEIGHT, RI, RE)
for ax in axs[0:3]:
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
for ax in axs[6:]:
    ax.scatter(xi_st, eta_st, 300, marker = '*', color='cyan', edgecolor='k', zorder=100)

axs[5].quiver(0.004, -0.192, 1e3, 0, linewidth = 2, scale=1e4, zorder=40, color='white', clip_on=False)
axs[5].text(0, -0.185, '1 [A/m]', va = 'top', ha = 'right', zorder = 101, size = 22)

axs[0].text(0.5, 1.05, 'Gamera (MHD)', va='center', ha='center', transform=axs[0].transAxes, fontsize=25)
axs[3].text(0.5, 1.05, 'EZIE', va='center', ha='center', transform=axs[3].transAxes, fontsize=25)
axs[6].text(0.5, 1.05, 'EZIE + ground magnetometer', va='center', ha='center', transform=axs[6].transAxes, fontsize=25)

axs[2].text(0.95+0.01, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12, color='k')
axs[2].text(0.95+0.01, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12, color='k')
axs[2].text(0.95+0.01, 0.35-0.02, '60$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12, color='k')
axs[2].text(0.95+0.01, 0.08-0.02, '50$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12, color='k')

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()
#%% Plot spatial resolution

xi_sig_gmag = np.zeros(R_gmag.shape[0])
eta_sig_gmag = np.zeros(R_gmag.shape[0])
xi_sig_org = np.zeros(R_gmag.shape[0])
eta_sig_org = np.zeros(R_gmag.shape[0])

xi_sig_gmag_f = np.zeros(R_gmag.shape[0])
eta_sig_gmag_f = np.zeros(R_gmag.shape[0])
xi_sig_org_f = np.zeros(R_gmag.shape[0])
eta_sig_org_f = np.zeros(R_gmag.shape[0])

for i in range(R_gmag.shape[0]):
    print(i)
    PSF = copy.deepcopy(R_gmag[:, i].reshape(grid.shape))
    PSF = abs(copy.deepcopy(R_gmag[:, i].reshape(grid.shape)))
    #PSF = copy.deepcopy(R_gmag[:, i].reshape(grid.shape))**2
    
    row = int(i/48)
    col = i%48
    xi_i = grid.xi[row, col]
    eta_i = grid.eta[row, col]
    
    inside = False
    if (xi_i >= xi_min) and (xi_i <= xi_max) and (eta_i >= eta_min) and (eta_i <= eta_max):
        inside = True
        
    args_xi = dict(inside=inside, x=grid.xi[row, :], x_min=xi_min, x_max=xi_max)
    args_eta = dict(inside=inside, x=grid.eta[:, col], x_min=eta_min, x_max=eta_max)
    
    PSF_xi = np.sum(PSF, axis=0)
    i_left, i_right, flag = P1.left_right(PSF_xi, **args_xi)
    xi_sig_gmag[i] = i_right - i_left
    xi_sig_gmag_f[i] = flag
        
    PSF_eta = np.sum(PSF, axis=1)
    i_left, i_right, flag = P1.left_right(PSF_eta, **args_eta)
    eta_sig_gmag[i] = i_right - i_left
    eta_sig_gmag_f[i] = flag
    
    
    PSF = copy.deepcopy(R_org[:, i].reshape(grid.shape))
    PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
    #PSF = copy.deepcopy(R_org[:, i].reshape(grid.shape))**2
    
    PSF_xi = np.sum(PSF, axis=0)
    i_left, i_right, flag = P1.left_right(PSF_xi, **args_xi)
    xi_sig_org[i] = i_right - i_left
    xi_sig_org_f[i] = flag
        
    PSF_eta = np.sum(PSF, axis=1)
    i_left, i_right, flag = P1.left_right(PSF_eta, **args_eta)
    eta_sig_org[i] = i_right - i_left
    eta_sig_org_f[i] = flag

plt.ioff()
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 1200, 50), cbar=False)
args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-300, 300, 16), cbar=False)
args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-300, 300, 60), cbar=False)


_, _, cc =      P1.plot_map_of_matrix(40* xi_sig_org, mask=xi_sig_org_f, ax=axs[0, 0], **args)
_, _, _ =       P1.plot_map_of_matrix(40* xi_sig_gmag, mask=xi_sig_gmag_f, ax=axs[0, 1], **args)
_, _, cc_diff = P1.plot_map_of_matrix(40*(xi_sig_org - xi_sig_gmag), mask=np.floor((xi_sig_org_f+xi_sig_gmag_f+eta_sig_org_f+eta_sig_gmag_f)/4), ax=axs[0, 2], **args_diff)

_, _, _ = P1.plot_map_of_matrix(20* eta_sig_org, mask=eta_sig_org_f, ax=axs[1, 0], **args)
_, _, _ = P1.plot_map_of_matrix(20* eta_sig_gmag, mask=eta_sig_gmag_f, ax=axs[1, 1], **args)
_, _, _ = P1.plot_map_of_matrix(20*(eta_sig_org - eta_sig_gmag), mask=np.floor((xi_sig_org_f+xi_sig_gmag_f+eta_sig_org_f+eta_sig_gmag_f)/4), ax=axs[1, 2], **args_diff)

for ax in axs[:, 1:].flatten():
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

cax = fig.add_axes([0.15, 0.07, 0.46, 0.02])
cax_diff = fig.add_axes([0.68, 0.07, 0.19, 0.02])

args_text = dict(va='center', ha='center', fontsize=20)

cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([0, 200, 400, 600, 800, 1000])
cax.text(0.5, -2.5, 'km', transform=cax.transAxes, **args_text)

cbar_diff = fig.colorbar(cc_diff, cax=cax_diff, orientation="horizontal")
cax_diff.set_xticks([-200, 0, 200])
#cax_diff.set_xticks([0, 250, 500])
cax_diff.text(0.5, -2.5, 'km', transform=cax_diff.transAxes, **args_text)

axs[0,0].text(-0.1, 0.5, 'Cross-track', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
axs[1,0].text(-0.1, 0.5, 'Along-track', transform=axs[1,0].transAxes, rotation='vertical', **args_text)

axs[0,0].text(0.5, 1.09, 'Without\nground magnetometer', transform=axs[0,0].transAxes, **args_text)
axs[0,1].text(0.5, 1.09, 'With\nground magnetometer', transform=axs[0,1].transAxes, **args_text)
axs[0,2].text(0.5, 1.05, 'Difference', transform=axs[0,2].transAxes, **args_text)

axs[0, 0].text(0.94, 0.88-0.03, '80$^{\circ}$', va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=12, color='w')
axs[0, 0].text(0.94, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=12, color='w')
axs[0, 0].text(0.94, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=12, color='w')
axs[0, 0].text(0.94, 0.08-0.03, '50$^{\circ}$', va='center', ha='center', transform=axs[0, 0].transAxes, fontsize=12, color='w')


plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot spatial resolution - Disputas

xi_sig_org = np.zeros(R_org.shape[0])
eta_sig_org = np.zeros(R_org.shape[0])

xi_sig_org_f = np.zeros(R_org.shape[0])
eta_sig_org_f = np.zeros(R_org.shape[0])

for i in range(R_org.shape[0]):
    print(i)
    
    row = int(i/48)
    col = i%48
    xi_i = grid.xi[row, col]
    eta_i = grid.eta[row, col]
    
    inside = False
    if (xi_i >= xi_min) and (xi_i <= xi_max) and (eta_i >= eta_min) and (eta_i <= eta_max):
        inside = True
        
    args_xi = dict(inside=inside, x=grid.xi[row, :], x_min=xi_min, x_max=xi_max)
    args_eta = dict(inside=inside, x=grid.eta[:, col], x_min=eta_min, x_max=eta_max)
    
    PSF = copy.deepcopy(R_org[:, i].reshape(grid.shape))
    #PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
    #PSF = copy.deepcopy(R_org[:, i].reshape(grid.shape))**2
    
    PSF_xi = np.sum(PSF, axis=0)
    i_left, i_right, flag = P1.left_right(PSF_xi, **args_xi)
    xi_sig_org[i] = i_right - i_left
    xi_sig_org_f[i] = flag
        
    PSF_eta = np.sum(PSF, axis=1)
    i_left, i_right, flag = P1.left_right(PSF_eta, **args_eta)
    eta_sig_org[i] = i_right - i_left
    eta_sig_org_f[i] = flag

f = (xi_sig_org_f == 1) & (eta_sig_org_f == 1)

plt.ioff()
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(15, 9))

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 700, 50), cbar=False)

_, _, cc =      P1.plot_map_of_matrix(40* xi_sig_org, mask=f, ax=axs[0], **args)
_, _, _ =       P1.plot_map_of_matrix(20* eta_sig_org, mask=f, ax=axs[1], **args)

cax = fig.add_axes([0.3, 0.07, 0.4, 0.02])

args_text = dict(va='center', ha='center', fontsize=20)

cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([0, 200, 400, 600])
cax.set_xticklabels(cax.get_xticks().astype(int), fontsize=15)
cax.text(0.5, -2.5, 'km', transform=cax.transAxes, **args_text)

axs[0].text(.5, 1.05, 'Cross-track', transform=axs[0].transAxes, **args_text)
axs[1].text(.5, 1.05, 'Along-track', transform=axs[1].transAxes, **args_text)

axs[0].text(0.94, 0.88-0.03, '80$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.08-0.03, '50$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')

#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/resolution.png', bbox_inches='tight')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/phd_defence/paper_III/resolution.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/conference/eiscat_2024/presentation/figures/resolution.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/conference/eiscat_2024/presentation/figures/resolution.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot spatial resolution - mag

#area_org = np.pi * (40*xi_sig_org/2) * (20*eta_sig_org/2)
#area_gmag = np.pi * (40*xi_sig_gmag/2) * (20*eta_sig_gmag/2)

area_org = (40*xi_sig_org + 20*eta_sig_org) / 2
area_gmag = (40*xi_sig_gmag + 20*eta_sig_gmag) / 2

mask = np.floor((xi_sig_org_f+xi_sig_gmag_f+eta_sig_org_f+eta_sig_gmag_f)/4)

plt.ioff()
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))

#args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 2e5, 50), cbar=False, mask=mask)
#args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-1e5, 1e5, 60), cbar=False, mask=mask)
args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 1200, 50), cbar=False, mask=mask)
args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-150, 150, 60), cbar=False, mask=mask)

_, _, cc =      P1.plot_map_of_matrix(area_org, ax=axs[0], **args)
_, _, _ =       P1.plot_map_of_matrix(area_gmag, ax=axs[1], **args)
_, _, cc_diff = P1.plot_map_of_matrix(area_org-area_gmag, ax=axs[2], **args_diff)

for ax in axs[1:].flatten():
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

cax = fig.add_axes([0.15, 0.07, 0.46, 0.02])
cax_diff = fig.add_axes([0.68, 0.07, 0.19, 0.02])

args_text = dict(va='center', ha='center', fontsize=20)

cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
#cax.set_xticks([0, 200, 400, 600, 800, 1000])
cax.text(0.5, -2.5, 'km', transform=cax.transAxes, **args_text)

cbar_diff = fig.colorbar(cc_diff, cax=cax_diff, orientation="horizontal")
#cax_diff.set_xticks([-200, 0, 200])
#cax_diff.set_xticks([0, 250, 500])
cax_diff.text(0.5, -2.5, 'km', transform=cax_diff.transAxes, **args_text)

#axs[0,0].text(-0.1, 0.5, 'Cross-track', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
#axs[1,0].text(-0.1, 0.5, 'Along-track', transform=axs[1,0].transAxes, rotation='vertical', **args_text)

axs[0].text(0.5, 1.09, 'Without\nground magnetometer', transform=axs[0].transAxes, **args_text)
axs[1].text(0.5, 1.09, 'With\nground magnetometer', transform=axs[1].transAxes, **args_text)
axs[2].text(0.5, 1.05, 'Difference', transform=axs[2].transAxes, **args_text)

axs[0].text(0.94, 0.88-0.03, '80$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')
axs[0].text(0.94, 0.08-0.03, '50$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=12, color='w')


plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_area_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_area_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%%

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
np.sqrt((40* xi_sig_org)**2  + (20* eta_sig_org)**2)
np.sqrt((40* xi_sig_gmag)**2 + (20* eta_sig_gmag)**2)



#%% Model variance projection into d

Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.lon_mesh.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

dCmp_org_e = Ge@Cmp_org@Ge.T
dCmp_org_n = Gn@Cmp_org@Gn.T
dCmp_org_u = Gu@Cmp_org@Gu.T

dCmp_gmag_e = Ge@Cmp_gmag@Ge.T
dCmp_gmag_n = Gn@Cmp_gmag@Gn.T
dCmp_gmag_u = Gu@Cmp_gmag@Gu.T

del Ge, Gn, Gu

Be_sig_org = np.sqrt(np.diag(dCmp_org_e))
Bn_sig_org = np.sqrt(np.diag(dCmp_org_n))
Bu_sig_org = np.sqrt(np.diag(dCmp_org_u))

Be_sig_gmag = np.sqrt(np.diag(dCmp_gmag_e))
Bn_sig_gmag = np.sqrt(np.diag(dCmp_gmag_n))
Bu_sig_gmag = np.sqrt(np.diag(dCmp_gmag_u))

plt.ioff()
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))
'''
args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(0.75, 2.25, 40), cbar=False, mesh=True)

_, _, cc = P1.plot_map_of_matrix(np.log10(Be_sig_org), ax=axs[0, 0], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bn_sig_org), ax=axs[0, 1], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bu_sig_org), ax=axs[0, 2], **args)

_, _, cc = P1.plot_map_of_matrix(np.log10(Be_sig_gmag), ax=axs[1, 0], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bn_sig_gmag), ax=axs[1, 1], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bu_sig_gmag), ax=axs[1, 2], **args)
'''
args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(10, 125, 24), cbar=False, mesh=True)

_, _, cc = P1.plot_map_of_matrix(Be_sig_org, ax=axs[0, 0], **args)
_, _, cc = P1.plot_map_of_matrix(Bn_sig_org, ax=axs[0, 1], **args)
_, _, cc = P1.plot_map_of_matrix(Bu_sig_org, ax=axs[0, 2], **args)

_, _, cc = P1.plot_map_of_matrix(Be_sig_gmag, ax=axs[1, 0], **args)
_, _, cc = P1.plot_map_of_matrix(Bn_sig_gmag, ax=axs[1, 1], **args)
_, _, cc = P1.plot_map_of_matrix(Bu_sig_gmag, ax=axs[1, 2], **args)

cax = fig.add_axes([0.15, 0.09, 0.73, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
#cax.set_xticks([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])

for ax in axs[1, :].flatten():
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

args_text = dict(va='center', ha='center', fontsize=20)

#cax.text(0.5, -2.5, 'log$_{10}$(B$_i$) [nT]', transform=cax.transAxes, **args_text)
cax.text(0.5, -2.5, 'nT', transform=cax.transAxes, **args_text)

axs[0,0].text(-0.1, 0.5, 'Without\nground magnetometer', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
axs[1,0].text(-0.1, 0.5, 'With\nground magnetometer', transform=axs[1,0].transAxes, rotation='vertical', **args_text)

axs[0,0].text(0.5, 1.05, '$\Delta$B$_{\phi}$', transform=axs[0,0].transAxes, **args_text)
axs[0,1].text(0.5, 1.05, '$\Delta$B$_{\u03b8}$', transform=axs[0,1].transAxes, **args_text)
axs[0,2].text(0.5, 1.05, '$\Delta$B$_{r}$', transform=axs[0,2].transAxes, **args_text)

axs[1, 0].text(0.94, 0.88-0.03, '80$^{\circ}$', va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=12)
axs[1, 0].text(0.94, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=12)
axs[1, 0].text(0.94, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=12)
axs[1, 0].text(0.94, 0.08-0.03, '50$^{\circ}$', va='center', ha='center', transform=axs[1, 0].transAxes, fontsize=12)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Model variance projection into d - with diff

Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.lon_mesh.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

dCmp_org_e = Ge@Cmp_org@Ge.T
dCmp_org_n = Gn@Cmp_org@Gn.T
dCmp_org_u = Gu@Cmp_org@Gu.T

dCmp_gmag_e = Ge@Cmp_gmag@Ge.T
dCmp_gmag_n = Gn@Cmp_gmag@Gn.T
dCmp_gmag_u = Gu@Cmp_gmag@Gu.T

del Ge, Gn, Gu

Be_sig_org = np.sqrt(np.diag(dCmp_org_e))
Bn_sig_org = np.sqrt(np.diag(dCmp_org_n))
Bu_sig_org = np.sqrt(np.diag(dCmp_org_u))

Be_sig_gmag = np.sqrt(np.diag(dCmp_gmag_e))
Bn_sig_gmag = np.sqrt(np.diag(dCmp_gmag_n))
Bu_sig_gmag = np.sqrt(np.diag(dCmp_gmag_u))

dBe_sig = Be_sig_org - Be_sig_gmag
dBn_sig = Bn_sig_org - Bn_sig_gmag
dBu_sig = Bu_sig_org - Bu_sig_gmag

plt.ioff()
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 16))

args_text = dict(va='center', ha='center', fontsize=20)

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(10, 125, 24), cbar=False, mesh=True)

_, _, cc = P1.plot_map_of_matrix(Be_sig_org, ax=axs[0, 0], **args)
_, _, cc = P1.plot_map_of_matrix(Bn_sig_org, ax=axs[0, 1], **args)
_, _, cc = P1.plot_map_of_matrix(Bu_sig_org, ax=axs[0, 2], **args)

_, _, cc = P1.plot_map_of_matrix(Be_sig_gmag, ax=axs[1, 0], **args)
_, _, cc = P1.plot_map_of_matrix(Bn_sig_gmag, ax=axs[1, 1], **args)
_, _, cc = P1.plot_map_of_matrix(Bu_sig_gmag, ax=axs[1, 2], **args)

cax = fig.add_axes([0.91, 0.4, .02, 0.45])
cbar = fig.colorbar(cc, cax=cax, orientation="vertical")
cax.text(4, 0.5, 'nT', transform=cax.transAxes, **args_text)

#args['clevels'] = np.linspace(0, 55, 24)
args['clevels'] = np.linspace(0, 28, 15)
args['extend'] = 'max'

_, _, cc = P1.plot_map_of_matrix(dBe_sig, ax=axs[2, 0], **args)
_, _, cc = P1.plot_map_of_matrix(dBn_sig, ax=axs[2, 1], **args)
_, _, cc = P1.plot_map_of_matrix(dBu_sig, ax=axs[2, 2], **args)

cax = fig.add_axes([0.91, 0.12, .02, 0.22])
cbar = fig.colorbar(cc, cax=cax, orientation="vertical")
cax.text(4, 0.5, 'nT', transform=cax.transAxes, **args_text)
#cax.set_yticks([0, 10, 20, 30, 40, 50])
cax.set_yticks([0, 6, 12, 18, 24])

cax.text(0.1, 1.1, 'max=54 nT', ha='left', va='center', fontsize=16, transform=cax.transAxes)

for ax in axs[1:, :].flatten():
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

axs[0,0].text(-0.1, 0.5, 'Without\nground magnetometer', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
axs[1,0].text(-0.1, 0.5, 'With\nground magnetometer', transform=axs[1,0].transAxes, rotation='vertical', **args_text)
axs[2,0].text(-0.1, 0.5, 'Difference', transform=axs[2,0].transAxes, rotation='vertical', **args_text)

axs[0,0].text(0.5, 1.05, '$\Delta$B$_{\phi}$', transform=axs[0,0].transAxes, **args_text)
axs[0,1].text(0.5, 1.05, '$\Delta$B$_{\u03b8}$', transform=axs[0,1].transAxes, **args_text)
axs[0,2].text(0.5, 1.05, '$\Delta$B$_{r}$', transform=axs[0,2].transAxes, **args_text)

axs[2, 0].text(0.94, 0.88-0.03, '80$^{\circ}$', va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=12)
axs[2, 0].text(0.94, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=12)
axs[2, 0].text(0.94, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=12)
axs[2, 0].text(0.94, 0.08-0.03, '50$^{\circ}$', va='center', ha='center', transform=axs[2, 0].transAxes, fontsize=12)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_diff_dark.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_diff_dark.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Combining lambda - Get main analysis information

#post_fix = 'AF'
post_fix = 'AF_abs'
#post_fix = 'AF_nd'
#post_fix = 'AF_abs_nd'
#post_fix = 'PSF'
#post_fix = 'PSF_abs'

#Rm_org = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}.npy'.format(post_fix))
#Rm_gmag = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}_c1.npy'.format(post_fix))
#Rm_multi = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}_c3.npy'.format(post_fix))

#Rm_org = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}.npy'.format(post_fix))
#Rm_gmag = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}_l70.npy'.format(post_fix))
#Rm_multi = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}_l75.npy'.format(post_fix))
#Rm_multi = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}_l72.npy'.format(post_fix))

Rm_org = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Rm_{}.npy'.format(post_fix))
Rm_gmag = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_gmag/Rm_{}.npy'.format(post_fix))
Rm_multi = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_gmag_multi/Rm_{}.npy'.format(post_fix))
#Rm_multi = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_gmag_multi/Rm_{}_c1.npy'.format(post_fix))

#%% Calculate Hoyer sparseness metric

#H_org = P1.calc_Hoyer(Rm_org, 'Rm_org')
#H_gmag = P1.calc_Hoyer(Rm_gmag, 'Rm_org', postfix='_l70')
#H_multi = P1.calc_Hoyer(Rm_multi, 'Rm_org', postfix='_l75')
#H_multi = P1.calc_Hoyer(Rm_multi, 'Rm_org', postfix='_l72')

H_org = P1.calc_Hoyer(Rm_org, 'Rm_org')
H_gmag = P1.calc_Hoyer(Rm_gmag, 'Rm_gmag')
H_multi = P1.calc_Hoyer(Rm_multi, 'Rm_gmag_multi')

#%% Combining lambda - Get most shitty model parameter

###
'''
# ORG
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
projection, grid = P1.get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
row_org, col_org = P1.find_param_with_lowest_Hoyer(H_org, obs, data, rs, tm, RI, LRES, WRES, wshift)
# ORG
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
projection, grid = P1.get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
row_gmag, col_gmag = P1.find_param_with_lowest_Hoyer(H_gmag, obs, data, rs, tm, RI, LRES, WRES, wshift)
# ORG
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
projection, grid = P1.get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
row_multi, col_multi = P1.find_param_with_lowest_Hoyer(H_multi, obs, data, rs, tm, RI, LRES, WRES, wshift)
'''
###

# ORG
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
#projection, grid = P1.get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
projection, grid = P1.get_projection(data, rs, tm, RI, LRES, WRES, wshift)
row_org, col_org = P1.find_param_with_lowest_Hoyer(H_org, obs, data, rs, tm, RI, LRES, WRES, wshift)

# G-MAG
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

row_st = 54 #58
col_st = 27 #17

lat_st = grid.lat_mesh[row_st, col_st]
lon_st = grid.lon_mesh[row_st, col_st]

obs['lat'].append(lat_st)
obs['lon'].append(lon_st)

row_gmag, col_gmag = P1.find_param_with_lowest_Hoyer(H_gmag, obs, data, rs, tm, RI, LRES, WRES, wshift)

# MULTI
data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
n_st = 10

np.random.seed(1337)
row_st = np.ones(n_st).astype(int)
while len(np.unique(row_st)) != n_st:
    row_st = np.random.uniform(10, 98, n_st).astype(int)
col_st = np.ones(n_st).astype(int)
while len(np.unique(col_st)) != n_st:    
    col_st = np.random.uniform(12, 37, n_st).astype(int)

xi_st_multi = grid.xi_mesh[row_st, col_st]
eta_st_multi = grid.eta_mesh[row_st, col_st]

lat_st = grid.lat_mesh[row_st, col_st]
lon_st = grid.lon_mesh[row_st, col_st]

obs['lat'].extend(lat_st)
obs['lon'].extend(lon_st)

row_multi, col_multi = P1.find_param_with_lowest_Hoyer(H_multi, obs, data, rs, tm, RI, LRES, WRES, wshift)

# Patch into existing code for testing
row_org_list = np.ones(100).astype(int) * row_org
col_org_list = np.ones(100).astype(int) * col_org
row_gmag_list = np.ones(100).astype(int) * row_gmag
col_gmag_list = np.ones(100).astype(int) * col_gmag
row_multi_list = np.ones(100).astype(int) * row_multi
col_multi_list = np.ones(100).astype(int) * col_multi

#%%
import os
base = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_multi_c1'

plt.ioff()
for i in range(0, 100, 5):
    print(i)
    try:
        os.mkdir('{}/l1_{}'.format(base, i))
    except:
        print('')
    for j in range(0, 200, 5):
        if i == 0:
            try:
                os.mkdir('{}/l2_{}'.format(base, j))
            except:
                print('')        
        
        mlt_shift = 3
        data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)
        projection, grid = P1.get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
        
        fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(Rm_multi[i, j, :, :], grid, data, t0, t1, RI, cmap='Reds')
        
        ax.plot(xi_st_multi, eta_st_multi, '*', markersize=20, color='k', alpha=0.5)
        ax.plot(grid.xi[row_multi_list[i], col_multi_list[i]], grid.eta[row_multi_list[i], col_multi_list[i]], 'o', markersize=10, color='tab:blue', alpha=0.5)
        
        plt.savefig('{}/l1_{}/{}.png'.format(base, i, j), bbox_inches='tight')
        plt.savefig('{}/l2_{}/{}.png'.format(base, j, i), bbox_inches='tight')
        plt.close('all')
plt.ion()

#%% Combining lambda - Get optimal l2

l1_list = np.linspace(-3, 3, 100)
l2_list = np.linspace(-5, 10, 200)
s = 12000
xx = np.tile(l1_list, (10, 1)).T.flatten()

l2_opt_org_id = P1.get_opt_l2(H_org, row_org_list, col_org_list)
l2_opt_org = l2_list[l2_opt_org_id]
yy = l2_list[l2_opt_org_id].flatten()
l2_opt_org_fit, tck_org = P1.make_spline_fit(xx, yy, s=65)

l2_opt_gmag_id = P1.get_opt_l2(H_gmag, row_gmag_list, col_gmag_list)
l2_opt_gmag = l2_list[l2_opt_gmag_id]
yy = l2_list[l2_opt_gmag_id].flatten()
l2_opt_gmag_fit, tck_gmag = P1.make_spline_fit(xx, yy, s=50)

l2_opt_multi_id = P1.get_opt_l2(H_multi, row_multi_list, col_multi_list)
l2_opt_multi = l2_list[l2_opt_multi_id]
yy = l2_list[l2_opt_multi_id].flatten()
l2_opt_multi_fit, tck_multi = P1.make_spline_fit(xx, yy, s=65)


plt.figure(figsize=(12, 10))
plt.plot(l1_list, l2_opt_org_fit)
plt.plot(l1_list, l2_opt_gmag_fit)
plt.plot(l1_list, l2_opt_multi_fit)

#%% Combining lambda - plot relation

mrkr1 = '.'
mrkr2 = '^'
mrkr3 = '*'
alpha = 0.3

plt.ioff()
plt.figure(figsize=(15, 10))

for i in range(l2_opt_org.shape[1]):
    if i == 0:        
        plt.plot(l1_list, l2_opt_org[:, i], mrkr1, color='tab:blue', alpha=alpha, label='Without ground magnetometer', zorder=0)
        #plt.plot(l1_list, l2_opt_org[:, i], mrkr1, color='tab:blue', alpha=alpha, label='Case 2', zorder=0)
    else:
        plt.plot(l1_list, l2_opt_org[:, i], mrkr1, color='tab:blue', alpha=alpha, zorder=0)
        
for i in range(l2_opt_org.shape[1]):
    if i == 0:        
        plt.plot(l1_list, l2_opt_gmag[:, i], mrkr2, color='tab:orange', alpha=alpha, label='With ground magnetometer', zorder=0)
        #plt.plot(l1_list, l2_opt_gmag[:, i], mrkr2, color='tab:orange', alpha=alpha, label='Case 1', zorder=0)
    else:
        plt.plot(l1_list, l2_opt_gmag[:, i], mrkr2, color='tab:orange', alpha=alpha, zorder=0)

plt.plot(np.linspace(-3, 3, 100), l2_opt_org_fit, color='k', linewidth=2, zorder=1)
plt.plot(np.linspace(-3, 3, 100), l2_opt_gmag_fit, color='k', linewidth=2, zorder=1)

plt.plot(np.linspace(-3, 3, 100), l2_opt_org_fit, '--', color='tab:blue', linewidth=5, zorder=2)
plt.plot(np.linspace(-3, 3, 100), l2_opt_gmag_fit, '--', color='tab:orange', linewidth=5, zorder=2)

plt.xlabel('log$_{10}(\lambda_1)$', fontsize=28)
plt.ylabel('log$_{10}(\lambda_2)$', fontsize=28)
#plt.legend()
lgnd = plt.legend(fontsize=25)
#change the marker size manually for both lines
lgnd.legendHandles[0].set_markersize(15)
lgnd.legendHandles[1].set_markersize(15)
lgnd.legendHandles[0].set_alpha(1)
lgnd.legendHandles[1].set_alpha(1)
plt.xlim(-3, 3)
plt.ylim(0, 5.2)
plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.tight_layout()
plt.grid(linewidth=0.5, which='both')

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/lambda_relation.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/lambda_relation.pdf', format='pdf', bbox_inches='tight')

for i in range(l2_opt_org.shape[1]):
    if i == 0:        
        plt.plot(l1_list, l2_opt_multi[:, i], mrkr3, color='tab:green', alpha=alpha, label='With 10 G-mags', zorder=0)
        #plt.plot(l1_list, l2_opt_multi[:, i], mrkr3, color='tab:green', alpha=alpha, label='Case 3', zorder=0)
    else:
        plt.plot(l1_list, l2_opt_multi[:, i], mrkr3, color='tab:green', alpha=alpha, zorder=0)

plt.plot(np.linspace(-3, 3, 100), l2_opt_multi_fit, color='k', linewidth=1.5, zorder=1)
        
plt.plot(np.linspace(-3, 3, 100), l2_opt_multi_fit, '--', color='tab:green', linewidth=3, zorder=2)

plt.legend()
plt.tight_layout()
plt.xlim(-3, 3)
plt.ylim(-0.5, 5.2)
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/lambda_relation_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/lambda_relation_multi.pdf', format='pdf', bbox_inches='tight')

#plt.close('all')
plt.ion()

#%% Combining lambda - Diagnostics - Plot surface

plt.ioff()
# Org
ax = P1.do_surface_diagnostic_plot(H_org, l1_list, l2_list, l2_opt_org_fit, 
                                   l2_opt_org, l2_opt_org_id, row_org_list, 
                                   col_org_list, title='Hoyer - EZIE')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org.pdf', format='pdf', bbox_inches='tight')
plt.close('all')

# Gmag
ax = P1.do_surface_diagnostic_plot(H_gmag, l1_list, l2_list, l2_opt_gmag_fit, 
                                   l2_opt_gmag, l2_opt_gmag_id, row_gmag_list, 
                                   col_gmag_list, title='Hoyer - EZIE + 1 G-mag')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_gmag.png', bbox_inches='tight')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_gmag.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org_l70.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org_l70.pdf', format='pdf', bbox_inches='tight')
plt.close('all')

# Multi
ax = P1.do_surface_diagnostic_plot(H_multi, l1_list, l2_list, l2_opt_multi_fit, 
                                   l2_opt_multi, l2_opt_multi_id, row_multi_list, 
                                   col_multi_list, title='Hoyer - EZIE + 10 G-mags')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_multi.png', bbox_inches='tight')
#plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_multi.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org_l72.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/surface_diagnostic_org_l72.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()


#%% Combining lambda - plot method

l1_id = 10
l2_id = 40

row_org = row_org_list[l1_id]
col_org = col_org_list[l1_id]

m_slice_org = H_org[:, :, row_org, :]

plt.ioff()
fig, axs = plt.subplots(1, 3, sharex=False, sharey=False, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(30, 10))

var = Rm_org[l1_id, l2_id, :, :].flatten()
vmin = np.min(var)
vmax = np.max(var)

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(0, vmax, 20), cbar=False)

# Maps
_, _, cc = P1.plot_map_of_matrix(var, ax=axs[0], gridcol='grey', **args)

xlim = axs[0].get_xlim()
axs[0].plot(xlim, [grid.eta[row_org, 0]]*2, color='k', linewidth=3)

cax = fig.add_axes([0.14, 0.9, 0.21, 0.03])
cbar = fig.colorbar(cc, cax=cax, orientation='horizontal')
cax.set_xticks([])
cax.set_xticklabels([])
#cax.text(0.5, 1.1, 'Information $\longrightarrow$', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0.5, 1.1, 'Information', fontsize=30, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0, 1.1, 'Less', fontsize=30, va='bottom', ha='left', transform=cax.transAxes)
cax.text(1, 1.1, 'More', fontsize=30, va='bottom', ha='right', transform=cax.transAxes)

# Movies
###
'''
movie = m_slice_org[l1_id, :, :]
#movie = np.gradient(movie, axis=0)
xx, yy = np.meshgrid(np.arange(1, 49), l2_list)
axs[1].contourf(xx, yy, np.log10(movie), levels=15)
#vmax = np.max(abs(movie))
#axs[1].contourf(xx, yy, movie, cmap='bwr', levels=np.linspace(-vmax, vmax, 40))
'''
###


#np.arange(1, 49)
s_org = 0.012
for i in np.arange(0, 200, 2):
    if np.isin(i, [0, 66, 134, 198]):
        axs[1].plot(np.arange(1, 49), m_slice_org[l1_id, i, :] + i*s_org, color='tab:red')
    elif i == l2_id:
        axs[1].plot(np.arange(1, 49), m_slice_org[l1_id, i, :] + i*s_org, color='k')
    else:
        axs[1].plot(np.arange(1, 49), m_slice_org[l1_id, i, :] + i*s_org, color='grey', linewidth=0.5)

axs[1].plot([col_org+1]*100, m_slice_org[l1_id, ::2, col_org] + np.arange(0, 200, 2)*s_org, '.-', color='tab:blue')


axs[1].spines.right.set_visible(False)
axs[1].spines.top.set_visible(False)
axs[1].set_yticks([])
axs[1].set_xlim([1, 48])
axs[1].set_xticks([12.5, 21.5, 26, 36])
#axs[1].set_ylabel('$\lambda_2 \longrightarrow$')
axs[2].text(-0.095, 0.06, '$\lambda_2$=10$^{-5}$', va='center', ha='left', fontsize=20, transform=axs[2].transAxes, color='tab:red')
axs[2].text(-0.095, 0.39, '$\lambda_2$=10$^{0}$', va='center', ha='left', fontsize=20, transform=axs[2].transAxes, color='tab:red')
#axs[2].text(-0.095, 0.73, '$\lambda_2$=10$^{5}$', va='center', ha='left', fontsize=20, transform=axs[2].transAxes, color='tab:red')
axs[2].text(-0.095, 0.94, '$\lambda_2$=10$^{10}$', va='center', ha='left', fontsize=20, transform=axs[2].transAxes, color='tab:red')

axs[1].set_xticklabels(['1', '2', '3', '4'], fontsize=25)
axs[1].set_xlabel('Beam #', fontsize=30)

# Lines
axs[2].plot(l2_list, m_slice_org[l1_id, :, col_org], color='tab:blue', linewidth=2)

ylim = axs[2].get_ylim()

l2_opt_id_org = l2_opt_org_id[l1_id]

id_min = np.argmin(m_slice_org[l1_id, :, col_org])
id_max = np.argmax(m_slice_org[l1_id, :, col_org])
axs[2].plot([l2_list[id_max]]*2, [ylim[0], m_slice_org[l1_id, id_max, col_org]], '--', color='tab:blue', linewidth=0.7)

axs[2].set_ylim(ylim)

axs[2].set_xlim([-5, 10])
axs[2].set_yticks([])
axs[2].text(1.03, 0.5, 'Information', va='center', ha='left', fontsize=30, transform=axs[2].transAxes, rotation='vertical')
axs[2].set_xlabel('log$_{10}(\lambda_2)$', fontsize=30)
axs[2].set_xticks(axs[2].get_xticks(), fontsize=25)
axs[2].spines.left.set_visible(False)
axs[2].spines.top.set_visible(False)

axs[0].plot(grid.xi[row_org, col_org], grid.eta[row_org, col_org], '.', color='white', markersize='22')
axs[0].plot(grid.xi[row_org, col_org], grid.eta[row_org, col_org], '.', color='tab:blue', markersize='20')

axs[1].plot(col_org+1, m_slice_org[l1_id, l2_id, col_org] + l2_id*s_org, '*', markersize=17, color='white')
axs[2].plot(l2_list[l2_id], m_slice_org[l1_id, l2_id, col_org], '*', markersize=17, color='white')
axs[1].plot(col_org+1, m_slice_org[l1_id, l2_id, col_org] + l2_id*s_org, '*', markersize=15, color='tab:red')
axs[2].plot(l2_list[l2_id], m_slice_org[l1_id, l2_id, col_org], '*', markersize=15, color='tab:red')

axs[1].plot(col_org+1, m_slice_org[l1_id, id_max, col_org] + id_max*s_org, 's', markersize=12, color='white')
axs[2].plot(l2_list[id_max], m_slice_org[l1_id, id_max, col_org], 's', markersize=12, color='white')
axs[1].plot(col_org+1, m_slice_org[l1_id, id_max, col_org] + id_max*s_org, 's', markersize=10, color='tab:red')
axs[2].plot(l2_list[id_max], m_slice_org[l1_id, id_max, col_org], 's', markersize=10, color='tab:red')

axs[1].plot(col_org+1, m_slice_org[l1_id, id_min, col_org] + id_min*s_org, '^', markersize=12, color='white')
axs[2].plot(l2_list[id_min], m_slice_org[l1_id, id_min, col_org], '^', markersize=12, color='white')
axs[1].plot(col_org+1, m_slice_org[l1_id, id_min, col_org] + id_min*s_org, '^', markersize=10, color='tab:red')
axs[2].plot(l2_list[id_min], m_slice_org[l1_id, id_min, col_org], '^', markersize=10, color='tab:red')

# Other stuff
axs[2].text(0.25, 0.5, 'Optimal $\lambda_2$', fontsize=24, va='center', ha='center', transform=axs[2].transAxes)

# Patches
con1 = ConnectionPatch(xyA = (0.11, 0.04), coordsA = axs[0].transData,
                      xyB = (0, 1), coordsB = axs[1].transData,
                      arrowstyle = 'Simple, tail_width=0.1, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=.2')

con2 = ConnectionPatch(xyA = (31.5, 1.7), coordsA = axs[1].transData,
                      xyB = (-5.2, 0.69), coordsB = axs[2].transData,
                      arrowstyle = 'Simple, tail_width=0.1, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=.1')

con3 = ConnectionPatch(xyA = (-1, 0.61), coordsA = axs[2].transData,
                      xyB = (2, 0.55), coordsB = axs[2].transData,
                      arrowstyle = 'Simple, tail_width=0.1, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=.4')


axs[0].arrow(0.3, -0.02, 0.4, 0, width=0.005, transform=axs[0].transAxes, clip_on=False, color='k')
axs[0].text(0.5, -0.06, 'Cross-track', va='center', ha='center', fontsize=24, transform=axs[0].transAxes)

axs[0].arrow(-0.02, 0.3, 0, 0.4, width=0.005, transform=axs[0].transAxes, clip_on=False, color='k')
axs[0].text(-0.07, 0.5, 'Along-track', va='center', ha='center', fontsize=24, transform=axs[0].transAxes, rotation=90)

axs[0].text(0.95, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=18)
axs[0].text(0.95, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=18)
axs[0].text(0.95, 0.35-0.02, '60$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=18)
axs[0].text(0.95, 0.08-0.02, '50$^{\circ}$', va='center', ha='center', transform=axs[0].transAxes, fontsize=18)

axs[0].add_artist(con1)
axs[1].add_artist(con2)
axs[2].add_artist(con3)

for i, j in enumerate(['A', 'B', 'C']):
    axs[i].text(0.05, 0.025, j, va='center', ha='center', transform=axs[i].transAxes, fontsize=30)

# Save
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/combining_lambda.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/combining_lambda.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Get L-curve and GCV

'''
l1_gcv = np.linspace(np.min(l1_list), np.max(l1_list), 1000)
l2_fit_org = BSpline(*tck_org)(l1_gcv)
np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_lis/l1.npy', l1_gcv)
np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/EZIE_lis/l2.npy', l2_fit_org)
'''

l1_gcv = np.linspace(np.min(l1_list), np.max(l1_list), 100)

l2_fit_org = BSpline(*tck_org)(l1_gcv)
l2_fit_gmag = BSpline(*tck_gmag)(l1_gcv)
l2_fit_multi = BSpline(*tck_multi)(l1_gcv)

gcv_org, rnorm_org, mnorm_org = P1.L_curve_and_GCV(info, LRES, WRES, RI, wshift, OBSHEIGHT, l1_gcv, l2_fit_org)
gcv_gmag, rnorm_gmag, mnorm_gmag = P1.L_curve_and_GCV(info, LRES, WRES, RI, wshift, OBSHEIGHT, l1_gcv, l2_fit_gmag, gmag=True, ground_file=ground_file)
gcv_multi, rnorm_multi, mnorm_multi = P1.L_curve_and_GCV(info, LRES, WRES, RI, wshift, OBSHEIGHT, l1_gcv, l2_fit_multi, gmag_multi=True, ground_file=ground_file)

'''
rnorm_c1 = copy.deepcopy(rnorm_org)
mnorm_c1 = copy.deepcopy(mnorm_org)

rnorm_c2 = copy.deepcopy(rnorm_org)
mnorm_c2 = copy.deepcopy(mnorm_org)

rnorm_c3 = copy.deepcopy(rnorm_org)
mnorm_c3 = copy.deepcopy(mnorm_org)
'''

#%% Plot L-curve
'''
rnorm_org = copy.deepcopy(rnorm_c2)
mnorm_org = copy.deepcopy(mnorm_c2)

rnorm_gmag = copy.deepcopy(rnorm_c1)
mnorm_gmag = copy.deepcopy(mnorm_c1)

rnorm_multi = copy.deepcopy(rnorm_c3)
mnorm_multi = copy.deepcopy(mnorm_c3)

l2_fit_gmag = copy.deepcopy(l2_fit_org)
l2_fit_multi = copy.deepcopy(l2_fit_org)
'''

kn_org_id, skip_org = P1.robust_Kneedle(rnorm_org, mnorm_org)
kn_gmag_id, skip_gmag = P1.robust_Kneedle(rnorm_gmag, mnorm_gmag)
kn_multi_id, skip_multi = P1.robust_Kneedle(rnorm_multi, mnorm_multi)

plt.ioff()
plt.figure(figsize=(15, 10))

#plt.loglog(rnorm_org, mnorm_org, color='tab:blue', label='Case 2')
plt.loglog(rnorm_org, mnorm_org, color='tab:blue', label='Without ground magnetometer', linewidth=3)
plt.loglog(rnorm_org[kn_org_id], mnorm_org[kn_org_id], '*', color='k', alpha=0.5, markersize=15)

#plt.loglog(rnorm_gmag, mnorm_gmag, color='tab:orange', label='Case 1')
plt.loglog(rnorm_gmag, mnorm_gmag, color='tab:orange', label='With ground magnetometer', linewidth=3)
plt.loglog(rnorm_gmag[kn_gmag_id], mnorm_gmag[kn_gmag_id], '*', color='k', alpha=0.5, markersize=15)

plt.grid(linewidth=0.5, which='both')
plt.gca().spines.top.set_visible(False)
plt.gca().spines.right.set_visible(False)

plt.xlabel('Misfit norm: $\sqrt{(Gm-d)^TC_d^{-1}(Gm-d)}$', fontsize=25)
plt.ylabel('Model norm: $\sqrt{m^T(I+L^TL)m}$', fontsize=25)
plt.legend(fontsize=25)

plt.tight_layout()

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/L_curve.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/L_curve.pdf', format='pdf', bbox_inches='tight')

#plt.loglog(rnorm_multi, mnorm_multi, color='tab:green', label='Case 3')
plt.loglog(rnorm_multi, mnorm_multi, color='tab:green', label='EZIE + 10 G-mag')
plt.loglog(rnorm_multi[kn_multi_id], mnorm_multi[kn_multi_id], '*', color='k', alpha=0.5)

plt.legend()
plt.tight_layout()
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/L_curve_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/L_curve_multi.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

print('')
print('Org kn:')
print(l1_gcv[kn_org_id])
print(l2_fit_org[kn_org_id])

print('')
print('Gmag kn:')
print(l1_gcv[kn_gmag_id])
print(l2_fit_gmag[kn_gmag_id])

print('')
print('Gmag multi kn:')
print(l1_gcv[kn_multi_id])
print(l2_fit_multi[kn_multi_id])


