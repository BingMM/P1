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

sys.path.append('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lib')
import P1_lib as P1

#%% Figure settings

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)

#%% Define case

reload(cases) 

info = cases.cases_all['case_2_K0582_mlt03']
#info = cases.cases_all['case_2_K0400_mlt03']
info['clevels'] = np.linspace(-1000, 1000, 21)
info['central_lat'] = 68
info['segment'] = 2

#info = cases.cases_all['case_1_K0582_mlt02']
#info['clevels'] = np.linspace(-1000, 1000, 21)

#info = cases.cases_all['case_3_K0582_mlt23']
#info['clevels'] = np.linspace(-600, 600, 21)

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

ground_file = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Gamera_data/ground/case_2/gamera_dBs_00km_2016-08-09T09:18:00.txt'
#ground_file = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Gamera_data/ground/case_1/gamera_dBs_00km_2016-08-09T08:50:00.txt'
#ground_file = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Gamera_data/ground/case_3/gamera_dBs_00km_2016-08-09T09:24:00.txt'

info['clevels'] = np.linspace(np.min(info['clevels']), np.max(info['clevels']), 40)

#%%
#P1.combine_lambda_analysis_SSH(info, LRES, WRES, RI, wshift, OBSHEIGHT, case=3)
#P1.combine_lambda_analysis_SSH(info, LRES, WRES, RI, wshift, OBSHEIGHT, lat_shift=info['central_lat'])


#%% Show MHD

#mlt_shift = 0
mlt_shift = 3

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

#projection, grid = P1.get_projection(data, 800, 1400, rs, tm, RI, LRES, WRES, wshift)
projection, grid = P1.get_projection(data, rs, tm, RI, LRES, WRES, wshift)

lat_mhd, lon_mhd = np.meshgrid(np.linspace(50, 90, 500), np.linspace(0, 360, 2000))

mhdBu =  info['mhdfunc'](lat_mhd.flatten(), lon_mhd.flatten() + info['mapshift'], fn = info['mhd_B_fn'])

plt.ioff()
fig = plt.figure(figsize=(15, 15))
ax = plt.gca()
pax = Polarsubplot(ax, minlat = 50, linestyle = '--', linewidth = .5, color = 'grey')

cc = pax.contourf(lat_mhd, (lon_mhd/15+mlt_shift)%24, mhdBu.reshape(lat_mhd.shape), cmap='bwr', levels=np.linspace(-np.max(abs(mhdBu)), np.max(abs(mhdBu)), 41))

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

ax.text(0.5+0.08, 0.5-0.08, '80', **args)
ax.text(0.5+0.16, 0.5-0.16, '70', **args)
ax.text(0.5+0.24, 0.5-0.24, '60', **args)
ax.text(0.5+0.32, 0.5-0.32, '50', **args)

cax = fig.add_axes([0.2, 0.06, 0.61, 0.015])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([-1000, -500, 0, 500, 1000])
#cax.set_xticklabels([])
cax.text(0.5, -2, '$\Delta$B$_r$ [nT]', fontsize=24, va='top', ha='center', transform=cax.transAxes)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/overview.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/overview.pdf', format='pdf', bbox_inches='tight')
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

# Hoyer test Kneedle - case 1
#l1 = 10**0.8181818181818183
#l2 = 10**3.2190284649528484

# Hoyer test Kneedle - case 3
#l1 = 10**0.5151515151515151
#l2 = 10**3.097689007218963

m_org, scale = P1.do_inversion(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                    obs['lat'], obs['lon'], 
                    obs['Bu'], obs['Bn'], obs['Be'],
                    grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

R_org =     P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)
R_org_low = P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=10**-5, l2=0)

Cmp_org = P1.get_posterior_model_covariance(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

q = P1.error_covariance_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

#%% Solve inversion for EZIE data alone - higher uncertainty in one area

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

fac = 2
for i in ['cov_uu', 'cov_ee', 'cov_nn', 'cov_en', 'cov_eu', 'cov_nu']:
    qq = np.array(copy.deepcopy(obs[i]))
    qq[20:60] *= fac
    obs[i] = qq.tolist()

Q = P1.get_covariance_matrix(obs)

# Hoyer test Kneedle
l1 = 10**0.8787878787878789
l2 = 10**3.245490870668853

#l1 = 0
#l2 = 0

R_gmag = P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=l2)

#%%

# get maps of MHD magnetic fields:
mhdBu_j =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
mhdBe_j =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn_j = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

dxi = np.diff(grid.xi_mesh, axis=1)[0, 0]
deta = np.diff(grid.eta_mesh, axis=0)[0, 0]

xi_pad_1 = grid.xi_mesh + dxi/2
xi_pad_2 = grid.xi_mesh
eta_pad_1 = grid.eta_mesh
eta_pad_2 = grid.eta_mesh + deta/2

xi_pad = np.hstack((grid.xi_mesh.flatten(), xi_pad_1.flatten(), xi_pad_2.flatten()))
eta_pad = np.hstack((grid.eta_mesh.flatten(), eta_pad_1.flatten(), eta_pad_2.flatten()))

lon_pad, lat_pad = grid.projection.cube2geo(xi_pad, eta_pad)

mhdBu =  info['mhdfunc'](lat_pad, lon_pad + info['mapshift'], fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](lat_pad, lon_pad + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](lat_pad, lon_pad + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

Ge, Gn, Gu = get_SECS_B_G_matrices(lat_pad, lon_pad, RE + OBSHEIGHT * 1e3, grid.lat, grid.lon, RI = RI)

G = np.vstack((Ge, Gn, Gu))
GTG = G.T@G
scale = np.median(np.diag(GTG))
I = np.eye(G.shape[1])
d = np.hstack((mhdBe, mhdBn, mhdBu))
GTd = G.T@d

alpha = 10**np.linspace(-11, 3, 100)
rnorm = np.zeros(len(alpha))
mnorm = np.zeros(len(alpha))
for i, alpha_i in enumerate(alpha):
    print(i, '/', len(alpha))
    m_i = np.linalg.solve(GTG + (alpha_i*scale)*I, GTd)
    mnorm[i] = np.linalg.norm(m_i)
    rnorm[i] = np.linalg.norm(d - G@m_i)

plt.figure()
plt.loglog(rnorm, mnorm)
plt.xlabel('rnorm')
plt.ylabel('mnorm')
plt.tight_layout()
id_opt = np.argmin(abs(rnorm - 8338.48))
#id_opt = np.argmin(abs(rnorm - 9166))
#id_opt = np.argmin(abs(rnorm - 20154)) # for grid2
plt.loglog(rnorm[id_opt], mnorm[id_opt], '*')
alpha_opt = alpha[id_opt]

m_t = np.linalg.solve(GTG + (alpha_opt*scale)*I, GTd)

#m_t = np.linalg.lstsq(np.vstack((Ge, Gn, Gu)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 10**-1.5)[0]
#m_t = np.linalg.lstsq(np.vstack((Ge, Gn, Gu)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 10**-1.5)[0]

'''
Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE + OBSHEIGHT * 1e3, grid.lat, grid.lon, RI = RI)
#m_t = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]
m_t = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 10**-2)[0]
'''

m_f = R_org@m_t

Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE + OBSHEIGHT * 1e3, grid.lat, grid.lon, RI = RI)

dBr_t = Gu_Bj@m_t
dBr_f = Gu_Bj@m_f

vmax = np.max(abs(np.vstack((dBr_t, dBr_f, mhdBu_j))))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(dBr_t.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('True dBr')
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(dBr_f.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('Filtered dBr')
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(mhdBu_j.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('MHD dBr')

vmax = np.max(abs(np.vstack((m_t, m_f))))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(m_t.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True)
ax.set_title('True model')
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(m_f.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True)
ax.set_title('Filtered model')

#%%

###
'''
from secsy import get_SECS_J_G_matrices
Ge, Gn = get_SECS_J_G_matrices(grid.lat.flatten()+0.1, grid.lon.flatten(), 
                               grid.lat.flatten(), grid.lon.flatten(), 
                               current_type = 'divergence_free', RI = RI)
'''
Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat.flatten()+0.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)
row = 60
col = 20
idd = row*48+col

#Rd = Gn@R_org@Gn
#PSFd = Rd[:, idd]

qq = Gu@R_org
PSFd = qq[:, idd]

PSFd = R_org[:, idd]

vmax = np.max(abs(PSFd))
#fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(PSFd.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(PSFd.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True)
ax.plot(grid.xi[row, col], grid.eta[row, col], '*', markersize=20, color='k')

plt.figure()
plt.plot(np.sum(PSFd.reshape(grid.shape), axis=0))
plt.plot(np.sum(PSFd.reshape(grid.shape), axis=1))
###

Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat.flatten()+0.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.lon_mesh.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

dBr = np.zeros(Gdu.shape[0])
rows = (np.arange(Gdu.shape[0])/49).astype(int)
cols = np.arange(Gdu.shape[0])%49
#f = ((rows > 65) & (rows <= 70)) & ((cols > 15) & (cols <= 18))
#f = ((rows > 65) & (rows <= 70)) & ((cols > 20) & (cols <= 23))
f = (rows == 65) & (cols == 18)
dBr[f.flatten()] = 10
f = (rows == 60) & (cols == 18)
dBr[f.flatten()] = -10


vmax = 10
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(dBr.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('True dBr')

scale = np.median(np.diag(Gdu.T@Gdu))
m = np.linalg.solve(Gdu.T@Gdu + scale*10**-3*np.eye(Gdu.shape[1]), Gdu.T@dBr)
#m = np.linalg.solve(Gdu.T@Gdu, Gdu.T@dBr)
m[:] = 0
#m[2800]= 1
m[75*48+16]= 1
m[60*48+16]= -1
m[45*48+16]= 1
m[30*48+16]= -1

m[75*48+27]= -1
m[60*48+27]= 1
m[45*48+27]= -1
m[30*48+27]= 1
vmax = np.max(abs(m))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(m.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True)
ax.set_title('Model of true dBr')

dBr_p = Gdu@m
vmax = np.max(abs(dBr_p))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(dBr_p.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('dBr from model')

mf = R_org@m
vmax = np.max(abs(mf))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(mf.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True)
ax.set_title('Filtered model')

dBr_f = Gdu@mf
vmax = np.max(abs(dBr_f))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(dBr_f.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
ax.set_title('Filtered dBr')

plt.figure()
plt.plot(np.sum(abs(mf.reshape(grid.shape)), axis=0))

plt.figure()
plt.plot(np.sum(abs(dBr_p.reshape(grid.xi_mesh.shape)), axis=0))

plt.figure()
plt.plot(np.sum(abs(dBr_f.reshape(grid.xi_mesh.shape)), axis=0))


q = Gdu@R_org@Gdu.T
q = Gdu@Gdu.T
qq = q[:, 2820]
vmax = np.max(abs(qq))
fig, ax, ax_cbar, cntrs = P1.plot_map_of_matrix(qq.reshape(grid.xi_mesh.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 50), cbar=True, mesh=True)
#ax.set_title('Filtered dBr')

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

#%% Plot initial example

i = 2324
PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
vmax = np.max(abs(PSF))

### Ellipse fit
result = P1.do_nonlinear_PSF_fit_2(grid, PSF)

gauss_2d = result.eval(x=grid.xi.flatten(), y=grid.eta.flatten()).reshape(grid.shape)

e1 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     1.1775*result.best_values['sigmax'], 1.1775*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1)

e2 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     3*result.best_values['sigmax'], 3*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1)

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
#fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 40), cbar=False)
#fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'Reds', clevels=np.linspace(0, vmax, 40), cbar=False)

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('bwr')
cmap = truncate_colormap(cmap, minval=0.5, maxval=1, n=1000)
fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = cmap, clevels=np.linspace(0, vmax, 40), cbar=False)
'''
for i in [0, 9, 19, 29, 39]:
    ax.plot(grid.xi[:, i], grid.eta[:, i], color='k')

for i in [0, 39, 59, 79, 99]:
    ax.plot(grid.xi[i, :], grid.eta[i, :], color='k')
'''

ax.add_patch(e1)
ax.add_patch(e2)

ax.arrow(0.05, 0.5, 0, 0.1, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k')
ax.arrow(0.05, 0.5, 0.1, 0, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k')
ax.text(0.15, 0.48, 'Cross-track', va='center', ha='center', transform=ax.transAxes, fontsize=16)
ax.text(0.08, 0.63, 'Along-track', va='center', ha='center', transform=ax.transAxes, fontsize=16)

    
cax = fig.add_axes([0.19, 0.16, 0.61, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([])
cax.set_xticklabels([])
#cax.text(0.486, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0, 1.1, '0', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(1, 1.1, 'max', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
ax_b = fig.add_axes([0.01, 0.04, 0.98, 0.1])
#ax_b = fig.add_axes([0.01, 0.04, 1.01, 0.1])

ax_r = fig.add_axes([1, 0.118, 0.2, 0.83])
ax_r_2 = ax_r.twinx()
    
PSF_xi = np.sum(PSF, axis=0)
PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
PSF_xi_g = np.sum(gauss_2d, axis=0)
PSF_eta_g = -np.sum(gauss_2d, axis=1)

ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='tab:blue')

ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='tab:blue')
    
ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
i_left, i_right, _ = P1.left_right(PSF_xi)
ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
i_left, i_right, _ = P1.left_right(-PSF_eta)
ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    


ax_b.plot([xi_mu]*2, [0, np.max(PSF_xi)], color='tab:green')
ax_b.fill_between([xi_mu-1.1775*xi_sig, xi_mu+1.1775*xi_sig], [0, 0], [np.max(PSF_xi)]*2, color='tab:green', alpha=0.4, zorder=-1)

ax_r_2.plot([0, np.min(PSF_eta)], [eta_mu]*2, color='tab:green')
ax_r_2.fill_between([np.min(PSF_eta), 0], [eta_mu-1.1775*eta_sig]*2, [eta_mu+1.1775*eta_sig]*2, color='tab:green', alpha=0.4, zorder=-1)

ax_b.plot([xi_mu_e]*2, [0, np.max(PSF_xi)], '--', color='tab:orange')
ax_b.fill_between([xi_mu_e-1.1775*xi_sig_e, xi_mu_e+1.1775*xi_sig_e], [0, 0], [np.max(PSF_xi)]*2, color='tab:orange', alpha=0.4, zorder=-1)

ax_r_2.plot([0, np.min(PSF_eta)], [eta_mu_e]*2, '--', color='tab:orange')
ax_r_2.fill_between([np.min(PSF_eta), 0], [eta_mu_e-1.1775*eta_sig_e]*2, [eta_mu_e+1.1775*eta_sig_e]*2, color='tab:orange', alpha=0.4, zorder=-1)

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
    
#ax.scatter(grid.xi[col, i], grid.eta[col, i], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
row = int(i/48)
col = i%48    
ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)

ax_mhd  = fig.add_axes([0.007, 0.97, 0.478, 0.44])
ax_secs = fig.add_axes([0.51, 0.97, 0.478, 0.44])
ax_cbar = fig.add_axes([0.19, 1.45, 0.61, 0.02])

import matplotlib.patheffects as mpe
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection

# plot the data tracks:
pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
for i in range(4):
    lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
    xi, eta = grid.projection.geo2cube(lon, lat)
    for ax in [ax_mhd, ax_secs]:
        ax.plot(xi, eta, color = 'C' + str(i), linewidth = 5, path_effects=pe1)

ximin = np.min(grid.xi)
ximax = np.max(grid.xi)
etamin = np.min(grid.eta)
etamax = np.max(grid.eta)

# set up G matrices for the magnetic field evaluated on a grid - for plotting maps
Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

# get maps of MHD magnetic fields:
mhdBu =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
mhdBe =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
mhdBn = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

# plot magnetic field in upward direction (MHD and retrieved)    
cntrs = ax_secs.contourf(grid.xi, grid.eta, Gdu.dot(m_org).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
ax_mhd.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

# plot colorbar:
ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
ax_cbar.text(1.12, -1, 'nT', va='center', ha='center', transform=ax_cbar.transAxes, fontsize=20)
#ax_cbar.set_xlabel('nT')
ax_cbar.set_yticks([])

scale_j = 1e10
# calculate the equivalent current of retrieved magnetic field:
jlat = grid.lat_mesh[::2, ::2].flatten()
jlon = grid.lon_mesh[::2, ::2].flatten()    
Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
je, jn = Gje.dot(m_org).flatten(), Gjn.dot(m_org).flatten()
xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

# plot the equivalent current in the SECS panels:
ax_secs.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)    

# calcualte the equivalent current corresponding to MHD output with perfect coverage:
Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE + OBSHEIGHT * 1e3, grid.lat[::2, ::2], grid.lon[::2, ::2], RI = RI)
mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 1e-2)[0]

Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid.lat[::2, ::2], grid.lon[::2, ::2], current_type = 'divergence_free', RI = RI)
mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
#mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

# plot the MHD equivalent current in eaach panel
for ax in [ax_mhd, ax_secs]:
    ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
# plot coordinate grids, fix aspect ratio and axes in each panel
for ax in [ax_mhd, ax_secs]:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for l in np.r_[60:90:5]:
        xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    for l in np.r_[0:360:15]:
        xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
        ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

    ax.axis('off')

# Write labels:
for ax, label in zip([ax_mhd, ax_secs], ['Br MHD', 'Br SECS']):        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

# set plot limits:
for ax in [ax_mhd, ax_secs]:
    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)
    ax.set_adjustable('datalim') 
    ax.set_aspect('equal')

# remove whitespace
plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/ecample_init.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_init.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot initial example 2

i = 2324
PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
vmax = np.max(abs(PSF))

### Ellipse fit
result = P1.do_nonlinear_PSF_fit_2(grid, PSF)

gauss_2d = result.eval(x=grid.xi.flatten(), y=grid.eta.flatten()).reshape(grid.shape)

e1 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     1.1775*result.best_values['sigmax'], 1.1775*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1)

e2 = patches.Ellipse((result.best_values['centerx'], result.best_values['centery']), 
                     3*result.best_values['sigmax'], 3*result.best_values['sigmay'],
                     angle=360-result.best_values['rotation']/np.pi*180, linewidth=2, fill=False, zorder=1)

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
#fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 40), cbar=False)
#fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'Reds', clevels=np.linspace(0, vmax, 40), cbar=False)

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('bwr')
cmap = truncate_colormap(cmap, minval=0.5, maxval=1, n=1000)
fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = cmap, clevels=np.linspace(0, vmax, 40), cbar=False)
'''
for i in [0, 9, 19, 29, 39]:
    ax.plot(grid.xi[:, i], grid.eta[:, i], color='k')

for i in [0, 39, 59, 79, 99]:
    ax.plot(grid.xi[i, :], grid.eta[i, :], color='k')
'''

ax.add_patch(e1)
ax.add_patch(e2)

ax.arrow(0.05, 0.5, 0, 0.1, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k')
ax.arrow(0.05, 0.5, 0.1, 0, transform=ax.transAxes, head_width=0.01, clip_on=True, color='k')
ax.text(0.15, 0.48, 'Cross-track', va='center', ha='center', transform=ax.transAxes, fontsize=16)
ax.text(0.08, 0.63, 'Along-track', va='center', ha='center', transform=ax.transAxes, fontsize=16)

    
cax = fig.add_axes([0.19, 0.2, 0.61, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([])
cax.set_xticklabels([])
#cax.text(0.486, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0, 1.1, '0', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(0.5, 1.1, '|PSF|', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
cax.text(1, 1.1, 'max', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
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
ax.text(0.95, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.35-0.02, '60$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12)
ax.text(0.95, 0.08-0.02, '50$^{\circ}$', va='center', ha='center', transform=ax.transAxes, fontsize=12)

con1 = ConnectionPatch(xyA = (0.1, 0.039), coordsA = ax.transData,
                      xyB = (-0.04, 0.14), coordsB = ax.transData,
                      arrowstyle = 'Simple, tail_width=0.2, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=-.18')

con2 = ConnectionPatch(xyA = (-0.04, 0.14), coordsA = ax.transData,
                      xyB = (0.1, 0.039), coordsB = ax.transData,
                      arrowstyle = 'Simple, tail_width=0.1, head_width=1, head_length=1',
                      color='k', connectionstyle='arc3,rad=.18')

ax.add_artist(con1)
ax.add_artist(con2)

ax.text(0.84, 0.7, 'East/West\ndirection', va='center', ha='center', transform=ax.transAxes, fontsize=16)

## remove whitespace
#plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/ecample_init_small.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_init_small.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot both solutions

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare_2(m_org, m_gmag, data, grid, grid2, 2, t0, t1, info, OBSHEIGHT, RI, RE)
for ax in axs[0:3]:
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
for ax in axs[6:]:
    ax.scatter(xi_st, eta_st, 300, marker = '*', color='cyan', edgecolor='k', zorder=100)

#pe1 = [mpe.Stroke(linewidth=4, foreground='white',alpha=1), mpe.Normal()]
#for ax in axs:
#    ax.plot([xi_min-0.003, xi_max+0.0015, xi_max+0.0015, xi_min-0.003, xi_min-0.003], 
#            [eta_mid-eta_window/2, eta_mid-eta_window/2, eta_mid+eta_window/2, eta_mid+eta_window/2, eta_mid-eta_window/2], 
#            color='k', path_effects=pe1, linewidth=3)

axs[3].text(0.5, 1.05, 'EZIE', va='center', ha='center', transform=axs[3].transAxes, fontsize=20)
axs[6].text(0.5, 1.05, 'EZIE + ground magnetometer', va='center', ha='center', transform=axs[6].transAxes, fontsize=20)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot both solutions - Bigger arrows

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare_2(m_org, m_gmag, data, grid, grid2, 4, t0, t1, info, OBSHEIGHT, RI, RE)
for ax in axs[0:3]:
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
for ax in axs[6:]:
    ax.scatter(xi_st, eta_st, 300, marker = '*', color='cyan', edgecolor='k', zorder=100)

axs[5].quiver(0.004, -0.192, 1e3, 0, linewidth = 2, scale=1e4, zorder=40, color='k', clip_on=False)
axs[5].text(0, -0.185, '1 [A/m]', va = 'top', ha = 'right', zorder = 101, size = 22)

axs[0].text(0.5, 1.05, 'Gamera (MHD)', va='center', ha='center', transform=axs[0].transAxes, fontsize=25)
axs[3].text(0.5, 1.05, 'EZIE', va='center', ha='center', transform=axs[3].transAxes, fontsize=25)
axs[6].text(0.5, 1.05, 'EZIE + ground magnetometer', va='center', ha='center', transform=axs[6].transAxes, fontsize=25)

axs[2].text(0.95+0.01, 0.88-0.02, '80$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.95+0.01, 0.61-0.02, '70$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.95+0.01, 0.35-0.02, '60$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12)
axs[2].text(0.95+0.01, 0.08-0.02, '50$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=12)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot both solutions - Bigger arrows - with diff

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare_2_wdiff(m_org, m_gmag, data, grid, grid2, 4, t0, t1, info, OBSHEIGHT, RI, RE)
for ax in axs[0:3]:
    ax.scatter(xi_st, eta_st, 700,  marker = '*', color='cyan', edgecolor='k', zorder=100)
for ax in axs[6:]:
    ax.scatter(xi_st, eta_st, 700, marker = '*', color='cyan', edgecolor='k', zorder=100)

axs[5].quiver(0.004, -0.192, 1e3, 0, linewidth = 2, scale=1e4, zorder=40, color='k', clip_on=False)
axs[5].text(0, -0.185, '1 [A/m]', va = 'top', ha = 'right', zorder = 101, fontsize = 35)

axs[0].text(0.5, 1.02, 'Gamera (MHD)', va='bottom', ha='center', transform=axs[0].transAxes, fontsize=40)
axs[3].text(0.5, 1.02, 'EZIE', va='bottom', ha='center', transform=axs[3].transAxes, fontsize=40)
axs[6].text(0.5, 1.02, 'EZIE +\nground magnetometer', va='bottom', ha='center', transform=axs[6].transAxes, fontsize=40)
axs[9].text(0.5, 1.02, 'Difference', va='bottom', ha='center', transform=axs[9].transAxes, fontsize=40)

axs[2].text(0.95+0.01, 0.88-0.05, '80$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=26)
axs[2].text(0.95+0.01, 0.61-0.03, '70$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=26)
axs[2].text(0.95+0.01, 0.35-0.03, '60$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=26)
axs[2].text(0.95+0.01, 0.08-0.01, '50$^{\circ}$', va='center', ha='center', transform=axs[2].transAxes, fontsize=26)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger_wdiff.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_bigger_wdiff.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot both solutions - multi

data, obs, rs, tm, t0, t1 = P1.get_data_from_case(info)

plt.ioff()
axs = P1.plot_function_compare(m_org, m_multi, data, grid, t0, t1, info, OBSHEIGHT, RI, RE)
for ax in axs[0:3]:
    ax.scatter(xi_st_multi, eta_st_multi, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
for ax in axs[6:]:
    ax.scatter(xi_st_multi, eta_st_multi, 300, marker = '*', color='cyan', edgecolor='k', zorder=100)

pe1 = [mpe.Stroke(linewidth=4, foreground='white',alpha=1), mpe.Normal()]
for ax in axs:
    ax.plot([xi_min-0.003, xi_max+0.0015, xi_max+0.0015, xi_min-0.003, xi_min-0.003], 
            [eta_mid-eta_window/2, eta_mid-eta_window/2, eta_mid+eta_window/2, eta_mid+eta_window/2, eta_mid-eta_window/2], 
            color='k', path_effects=pe1, linewidth=3)

axs[3].text(0.5, 1.05, 'EZIE', va='center', ha='center', transform=axs[3].transAxes, fontsize=20)
axs[6].text(0.5, 1.05, 'EZIE + 10 G-mag', va='center', ha='center', transform=axs[6].transAxes, fontsize=20)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/example_multi.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot a PSF

###
'''
q = copy.deepcopy(PSF.flatten())
qq = np.flip(np.sort(q))

#plt.figure()
#plt.plot(qq)
#ylim = plt.gca().get_ylim()
f = (np.cumsum(qq) / np.sum(qq) * 100) < 60
#plt.plot([np.argmin(f)]*2, ylim, '--')

q[q <= np.max(qq[~f])] = 0
fig, ax, cc = P1.plot_map_of_matrix(q.reshape(grid.shape), grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
'''
###

col = 44

plt.ioff()
#for i in range(48):
#    PSF = R_org[:, col*48+i].reshape(grid.shape)
for i in range(R_org.shape[0]):
    if (i < 4000) or (i >4200):
        continue    
    print('org', i, '/', R_org.shape[0])
    PSF = abs(copy.deepcopy(R_org[:, i].reshape(grid.shape)))
    vmax = np.max(abs(PSF))
    fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
    
    cax = fig.add_axes([0.2, 0.85, 0.61, 0.015])
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    cax.set_xticklabels([])
    cax.text(0.5, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
    ax_b = fig.add_axes([0.13, 0.07, 0.77, 0.1])
    ax_r = fig.add_axes([0.91, 0.18, 0.2, 0.66])
    ax_r_2 = ax_r.twinx()
    
    PSF_xi = np.sum(PSF, axis=0)
    PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
    ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='k')
    ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='k')
    
    ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
    ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
    i_left, i_right, _ = P1.left_right(PSF_xi)
    ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
    i_left, i_right, _ = P1.left_right(-PSF_eta)
    ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    
        
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
    
    #ax.scatter(grid.xi[col, i], grid.eta[col, i], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    row = int(i/48)
    col = i%48    
    ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_org_Hoyer/PSF_{}.png'.format(i), bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_org_Hoyer/PSF_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
    plt.close('all')
plt.ion()

#%% Plot a AF

col = 44

plt.ioff()
#for i in range(48):
#    PSF = R_org[:, col*48+i].reshape(grid.shape)
for i in range(R_org.shape[0]):
    if i < 2000:
        continue
    if i > 2500:
        break
    print('org', i, '/', R_org.shape[0])
    PSF = copy.deepcopy(R_org.T[:, i].reshape(grid.shape)) # Actually AF
    vmax = np.max(abs(PSF))
    fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
    
    cax = fig.add_axes([0.2, 0.85, 0.61, 0.015])
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    cax.set_xticklabels([])
    cax.text(0.5, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
    ax_b = fig.add_axes([0.13, 0.07, 0.77, 0.1])
    ax_r = fig.add_axes([0.91, 0.18, 0.2, 0.66])
    ax_r_2 = ax_r.twinx()
    
    PSF_xi = np.sum(PSF, axis=0)
    PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
    ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='k')
    ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='k')
    
    ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
    ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
    i_left, i_right, _ = P1.left_right(PSF_xi)
    ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
    i_left, i_right, _ = P1.left_right(-PSF_eta)
    ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    
        
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
    
    #ax.scatter(grid.xi[col, i], grid.eta[col, i], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    row = int(i/48)
    col = i%48    
    ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/AF_org/PSF_{}.png'.format(i), bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/AF_org/PSF_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
    plt.close('all')
plt.ion()


cols = np.arange(R_org.shape[0]).astype(int)%48
R_test = copy.deepcopy(R_org)
#R_test[:, cols<10] = 0
#R_test[:, cols>36] = 0
#Rm_AF = np.sum(R_test, axis=1).reshape(grid.shape)
Rm_AF = np.sum(abs(R_test), axis=1).reshape(grid.shape)
P1.plot_map_of_matrix(Rm_AF, grid, data, t0, t1, RI, cmap='Reds')


plt.figure()
plt.plot(np.cumsum(np.sum(R_org[2042, :].reshape(grid.shape), axis=0)), label='Case 1')
plt.plot(np.cumsum(np.sum(R_org[2049, :].reshape(grid.shape), axis=0)), label='Case 2')
plt.legend()

plt.figure()
plt.plot(np.cumsum(np.sum(abs(R_org[2042, :]).reshape(grid.shape), axis=0)))
plt.plot(np.cumsum(np.sum(abs(R_org[2049, :]).reshape(grid.shape), axis=0)))

#%% Plot a AF - l2

plt.ioff()
for i, l2 in enumerate(np.linspace(-5, 10, 100)):
    print(i)
    R_i = P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=10**l2)
    PSF = R_i[3342, :].reshape(grid.shape) # Actually AF

    vmax = np.max(abs(PSF))
    fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
    
    cax = fig.add_axes([0.2, 0.85, 0.61, 0.015])
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    cax.set_xticklabels([])
    cax.text(0.5, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
    ax_b = fig.add_axes([0.13, 0.07, 0.77, 0.1])
    ax_r = fig.add_axes([0.91, 0.18, 0.2, 0.66])
    ax_r_2 = ax_r.twinx()
    
    PSF_xi = np.sum(PSF, axis=0)
    PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
    ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='k')
    ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='k')
    
    ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
    ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
    i_left, i_right, _ = P1.left_right(PSF_xi)
    ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
    i_left, i_right, _ = P1.left_right(-PSF_eta)
    ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    
        
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
    
    #ax.scatter(grid.xi[col, i], grid.eta[col, i], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    row = 69
    col = 30
    ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/AF_l2/PSF_{}.png'.format(i), bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/AF_l2/PSF_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
    plt.close('all')
plt.ion()



l2_list = np.linspace(-5, 10, 200)
#l1 = 10**-3
l1 = 10**0.8
l1s = np.zeros(200)
l2s = np.zeros(200)
gini = np.zeros(200)
dl = np.zeros(200)
for i, l2 in enumerate(l2_list):
    print(i)
    if i != 124:
        continue
    R_i = P1.get_resolution_matrix(np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, obs['lat'], obs['lon'], grid.lat.flatten(), grid.lon.flatten(), LL, Q, RI, l1=l1, l2=10**l2)
    continue
    id_max = np.argmax(abs(R_i[:, 3342]))
    row_max = int(id_max/48)
    col_max = id_max%48
    drow = (69 - row_max) * 20
    dcol = (30 - col_max) * 40
    dl[i] = np.sqrt(np.sum(drow**2 + dcol**2))    
    
    AF = R_i[3342, :]
    l1s[i] = np.sum(abs(AF))
    l2s[i] = np.sqrt(np.sum(AF**2))
    
    # Gini
    N = R_i.shape[0]
    k = np.arange(N)+1
    c = np.sort(abs(AF))
    gini[i] = 1 - 2*np.sum(c / l1s[i] * ((N-k+0.5) / N))

H = (np.sqrt(N) - (l1s/l2s)) / (np.sqrt(N)-1)

plt.figure(figsize=(12, 9))
plt.plot(gini, label='Gini')
plt.plot(H, label='Hoyer')
#ylim = plt.gca().get_ylim()
#plt.plot([126]*2, ylim)
#plt.ylim(ylim)
plt.legend()
plt.xlabel('index of $\lambda_2$')
plt.ylabel('Information')
plt.tight_layout()

plt.figure(figsize=(12, 9))
plt.plot(dl)
plt.title('Euclidean distance between impulse and PSF max')
plt.xlabel('index of $\lambda_2$')
plt.ylabel('Distance [km]')
plt.tight_layout()

PSF = abs(R_i[:, 3342].reshape(grid.shape))
vmax = np.max(PSF)
vmin = np.min(PSF)
P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'Reds', clevels=np.linspace(vmin, vmax, 50), cbar=True)

AF = abs(R_i[3342, :].reshape(grid.shape))
vmax = np.max(AF)
vmin = np.min(AF)
P1.plot_map_of_matrix(AF, grid, data, t0, t1, RI, cmap = 'Blues', clevels=np.linspace(vmin, vmax, 50), cbar=True)


plt.figure()
plt.plot(gini/dl)
plt.plot(H/dl)

plt.figure()
plt.plot(dl)


plt.figure()
plt.plot(np.linspace(-5, 10, 100), l2s/l1s)

#%%

def get_cut_off(L):
    
    cut_off = np.zeros(100).astype(int)    
    for i in range(100):
        q = np.zeros(200)
        q[1:] = np.diff(L[i, :])
        q[0] = q[1]
        
        med = np.median(q)
        std = np.std(q)
        
        f = abs(q - med) > 5*std
        if np.any(f):
            cut_off[i] = np.argmax(f)
    
    return cut_off

def get_cut_off_2(L):
    
    cut_off = np.zeros(100).astype(int)    
    for i in range(100):
        q = copy.deepcopy(L[i, :])
        q[:np.argmin(q)] = q[0]
        cut_off[i] = np.argmax(q > q[0])
        if cut_off[i] == 0:
            cut_off[i] = 199
    
    return cut_off

def get_cut_off_2_1(L, threshold=0.05):
    
    cut_off = np.zeros(100).astype(int)    
    for i in range(100):
        q = copy.deepcopy(L[i, :])
        q[:np.argmin(q)] = q[0]
        cut_off[i] = np.argmax(q > q[0]*(1+threshold))
        if cut_off[i] == 0:
            cut_off[i] = 199
    
    return cut_off

def get_cut_off_3(L, threshold=40):
    
    cut_off = np.zeros(100).astype(int)
    for i in range(100):
        q = copy.deepcopy(L[i, :])
        if threshold >= 1:
            cut_off[i] = np.argmax(q > (q[0]+threshold))
        else:
            cut_off[i] = np.argmax(q > ((1+threshold)*q[0]))
            
        if cut_off[i] == 0:
            cut_off[i] = len(q)-1
        
    return cut_off

def get_line(cut_off, metric):
    y = []
    z = []
    for i in range(100):
        y.append(np.linspace(-5, 10, 200)[cut_off[i]])
        z.append(metric[i, :][cut_off[i]])
    return np.array(y), np.array(z)

def get_id_max(metric, cut_off):
    id_max = np.zeros(100).astype(int)
    for i in range(100):        
        id_max[i] = np.argmax(metric[i, :cut_off[i]])
    return id_max

Hoyer_68 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Hoyer.npy')[:, :, 69*48+30]
Hoyer_72 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Hoyer_l72.npy')[:, :, 69*48+30]

Gini_68 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Gini.npy')[:, :, 69*48+30]
Gini_72 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Gini_l72.npy')[:, :, 69*48+30]

L_68 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/L.npy')[:, :, 69*48+30]
L_72 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/L_l72.npy')[:, :, 69*48+30]

        
cut_off_68 = get_cut_off(L_68)
cut_off_72 = get_cut_off(L_72)


y_68, z_68 = get_line(cut_off_68, Gini_68)
y_72, z_72 = get_line(cut_off_72, Gini_72)

id_max_68 = get_id_max(Gini_68, cut_off_68)
id_max_72 = get_id_max(Gini_72, cut_off_72)

y_max_68, z_max_68 = get_line(id_max_68, Gini_68)
y_max_72, z_max_72 = get_line(id_max_72, Gini_72)
        
ax = P1.surface_plot(Gini_68, 'Gini : lat 68')
ax.plot(np.linspace(-3, 3, 100), y_68, z_68, zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max_68, z_max_68, zorder=10)

ax = P1.surface_plot(Gini_72, 'Gini : lat 72')
ax.plot(np.linspace(-3, 3, 100), y_72, z_72, zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max_72, z_max_72, zorder=10)


y_68, z_68 = get_line(cut_off_68, Hoyer_68)
y_72, z_72 = get_line(cut_off_72, Hoyer_72)

id_max_68 = get_id_max(Hoyer_68, cut_off_68)
id_max_72 = get_id_max(Hoyer_72, cut_off_72)

y_max_68, z_max_68 = get_line(id_max_68, Hoyer_68)
y_max_72, z_max_72 = get_line(id_max_72, Hoyer_72)
        
ax = P1.surface_plot(Hoyer_68, 'Hoyer : lat 68')
ax.plot(np.linspace(-3, 3, 100), y_68, z_68, zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max_68, z_max_68, zorder=10)

ax = P1.surface_plot(Hoyer_72, 'Hoyer : lat 72')
ax.plot(np.linspace(-3, 3, 100), y_72, z_72, zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max_72, z_max_72, zorder=10)


y_68, z_68 = get_line(cut_off_68, np.log10(L_68))
y_72, z_72 = get_line(cut_off_72, np.log10(L_72))

ax = P1.surface_plot(np.log10(L_68), 'Log of localization error : lat 68')
ax.plot(np.linspace(-3, 3, 100), y_68, z_68 , '*-', color='k', zorder=10)

ax = P1.surface_plot(np.log10(L_72), 'Log of localization error : lat 72')
ax.plot(np.linspace(-3, 3, 100), y_72, z_72, '*-', color='k', zorder=10)


id_max_68 = get_id_max(Gini_68 / np.log10(L_68), np.ones(100).astype(int)*199)
y_68, z_68 = get_line(id_max_68, Gini_68 / np.log10(L_68))
ax = P1.surface_plot(Gini_68 / np.log10(L_68), 'Local * Gini : lat 68')
ax.plot(np.linspace(-3, 3, 100), y_68, z_68 , '*-', color='k', zorder=10)

id_max_68 = get_id_max(Hoyer_68 / np.log10(L_68), np.ones(100).astype(int)*199)
y_68, z_68 = get_line(id_max_68, Hoyer_68 / np.log10(L_68))
ax = P1.surface_plot(Hoyer_68 / np.log10(L_68), 'Local * Hoyer : lat 68')
ax.plot(np.linspace(-3, 3, 100), y_68, z_68 , '*-', color='k', zorder=10)

Lw = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/Lw.npy')[:, :, 69*48+30]
ax = P1.surface_plot(Lw, 'Weighted localization error : lat 68')

q = Gini_68 / Lw
id_max = get_id_max(q, np.ones(100).astype(int)*199)
y, z = get_line(id_max, q)
ax = P1.surface_plot(q, 'Weighted localization error : lat 68')
ax.plot(np.linspace(-3, 3, 100), y, z , '*-', color='k', zorder=10)


ax = P1.surface_plot(Lw, 'Weighted localization error : lat 68', cmap='Reds')
for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cut_off = get_cut_off_3(Lw, threshold=i)
    y_cut, z_cut = get_line(cut_off, Lw)
    ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, zorder=10, label=i)
plt.legend()

ax = P1.surface_plot(Lw, 'Weighted localization error : lat 68')
cut_off = get_cut_off_2_1(Lw)
y_cut, z_cut = get_line(cut_off, Lw)
ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, zorder=10)

cut_off = get_cut_off_2_1(Lw, threshold=0.1)
y_cut, z_cut = get_line(cut_off, Gini_68)
id_max = get_id_max(Gini_68, cut_off)
y_max, z_max = get_line(id_max, Gini_68)
ax = P1.surface_plot(Gini_68, 'Gini : lat 68', cmap='Reds')
ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, '*-', color='k', zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max, z_max, '*-', color='magenta', zorder=10)


for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    cut_off = get_cut_off_3(Lw, threshold=i)
    y_cut, z_cut = get_line(cut_off, Lw)
    ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, zorder=10, label=i)
plt.legend()
    

cut_off = get_cut_off_3(Lw, threshold=0.1)
y_cut, z_cut = get_line(cut_off, Gini_68)
id_max = get_id_max(Gini_68, cut_off)
y_max, z_max = get_line(id_max, Gini_68)
ax = P1.surface_plot(Gini_68, 'Gini : lat 68', cmap='Reds')
ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, '*-', color='k', zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max, z_max, '*-', color='magenta', zorder=10)

cut_off = get_cut_off_3(Lw, threshold=0.5)
y_cut, z_cut = get_line(cut_off, Hoyer_68)
id_max = get_id_max(Hoyer_68, cut_off)
y_max, z_max = get_line(id_max, Hoyer_68)
ax = P1.surface_plot(Hoyer_68, 'Hoyer : lat 68', cmap='Reds')
ax.plot(np.linspace(-3, 3, 100), y_cut, z_cut, '*-', color='k', zorder=10)
ax.plot(np.linspace(-3, 3, 100), y_max, z_max, '*-', color='magenta', zorder=10)

cut_off = get_cut_off_3(Lw, threshold=0.5)
y_cut, z_cut = get_line(cut_off, Hoyer_68)
id_max = get_id_max(Hoyer_68, cut_off)
y_max, z_max = get_line(id_max, Hoyer_68)

ax = P1.surface_plot(Gini_68 / Lw, 'Gini / Lw : lat 68')
ax = P1.surface_plot(Hoyer_68 / Lw, 'Hoyer / Lw : lat 68')


q = Gini_68 / Lw
plt.figure()
plt.plot(q[0, :])
plt.plot(q[25, :])
plt.plot(q[50, :])
plt.plot(q[99, :])



L10 = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/L10.npy')[:, :, 69*48+30, :]
L10_mean = np.max(L10, axis=2)
ax = P1.surface_plot(L10_mean, 'avg L 10 : lat 68', cmap='Reds')

plt.figure()
plt.plot(L10_mean[0, :])
plt.plot(L10_mean[25, :])
plt.plot(L10_mean[50, :])
plt.plot(L10_mean[75, :])
plt.plot(L10_mean[99, :])

ax = P1.surface_plot(L_68, 'L : lat 68', cmap='Reds')
ax = P1.surface_plot(L_72, 'L : lat 72', cmap='Reds')


L_wide = np.load('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_org/L_wide.npy')[:, :, 69*48+30]
ax = P1.surface_plot(L_wide, 'L wide : lat 68')

ax = P1.surface_plot(Lw, 'Lw : lat 68')

#%% Plot a PSF - Gmag

plt.ioff()
for i in range(R_org.shape[0]):
    if (i < 4000) or (i >4200):
        continue
    print('gmag', i, '/', R_org.shape[0])
    PSF = abs(copy.deepcopy(R_gmag[:, i].reshape(grid.shape)))
    vmax = np.max(abs(PSF))            
    fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
    
    
    cax = fig.add_axes([0.2, 0.85, 0.61, 0.015])
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    cax.set_xticklabels([])
    cax.text(0.5, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
    ax_b = fig.add_axes([0.13, 0.07, 0.77, 0.1])
    ax_r = fig.add_axes([0.91, 0.18, 0.2, 0.66])
    ax_r_2 = ax_r.twinx()
    
    PSF_xi = np.sum(PSF, axis=0)
    PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
    ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='k')
    ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='k')
    
    ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
    ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
    i_left, i_right, _ = P1.left_right(PSF_xi)
    ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
    i_left, i_right, _ = P1.left_right(-PSF_eta)
    ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    
        
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
    
    row = int(i/48)
    col = i%48
    
    ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)
    ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_gmag_Hoyer/PSF_{}.png'.format(i), bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_gmag_Hoyer/PSF_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
    plt.close('all')
plt.ion()

#%% Plot a PSF - multi

plt.ioff()
for i in range(R_org.shape[0]):
    print('multi', i, '/', R_org.shape[0])
    PSF = copy.deepcopy(R_multi[:, i].reshape(grid.shape))
    vmax = np.max(abs(PSF))            
    fig, ax, cc = P1.plot_map_of_matrix(PSF, grid, data, t0, t1, RI, cmap = 'bwr', clevels=np.linspace(-vmax, vmax, 41), cbar=False)
    
    
    cax = fig.add_axes([0.2, 0.85, 0.61, 0.015])
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticks([])
    cax.set_xticklabels([])
    cax.text(0.5, 1.1, 'Negative $\longleftarrow$|$\longrightarrow$ Positive', fontsize=18, va='bottom', ha='center', transform=cax.transAxes)
    
    ax_b = fig.add_axes([0.13, 0.07, 0.77, 0.1])
    ax_r = fig.add_axes([0.91, 0.18, 0.2, 0.66])
    ax_r_2 = ax_r.twinx()
    
    PSF_xi = np.sum(PSF, axis=0)
    PSF_eta = -np.sum(PSF, axis=1) # negative for plotting reasons
    
    ax_b.plot(np.arange(1, grid.shape[1]+1), PSF_xi, color='k')
    ax_r_2.plot(PSF_eta, np.arange(1, grid.shape[0]+1), color='k')
    
    ax_b.plot([1, grid.shape[1]], [0, 0], '--', color='k', linewidth=0.5)
    ax_r_2.plot([0, 0], [1, grid.shape[0]], '--', color='k', linewidth=0.5)
    
    i_left, i_right, _ = P1.left_right(PSF_xi)
    ax_b.plot([i_left+1, i_right+1], [0.5*np.max(PSF_xi)]*2, '|-', color='tab:red', markersize=15)
    
    i_left, i_right, _ = P1.left_right(-PSF_eta)
    ax_r_2.plot([0.5*np.min(PSF_eta)]*2, [i_left+1, i_right+1], '_-', color='tab:red', markersize=15)    
        
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
    
    row = int(i/48)
    col = i%48
    
    ax.scatter(grid.xi[row, col], grid.eta[row, col], 100,  marker = 'o', color='cyan', edgecolor='k', zorder=100)
    ax.scatter(xi_st_multi, eta_st_multi, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)    
    ax.text(0.95, 0.88, '80', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.61, '70', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.35, '60', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.08, '50', va='center', ha='center', transform=ax.transAxes, fontsize=12)
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_multi/PSF_{}.png'.format(i), bbox_inches='tight')
    plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/PSF_multi/PSF_{}.pdf'.format(i), format='pdf', bbox_inches='tight')
    plt.close('all')
plt.ion()

#%% Plot spatial resolution

###
#R_org = Gu@R_org
#R_gmag = Gu
###

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
#args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 1200, 50), cbar=False)
#args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(0, 600, 15), cbar=False)
#args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 1500, 50), cbar=False)
#args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-900, 900, 30), cbar=False)

_, _, cc =      P1.plot_map_of_matrix(40* xi_sig_org, mask=xi_sig_org_f, ax=axs[0, 0], **args)
_, _, _ =       P1.plot_map_of_matrix(40* xi_sig_gmag, mask=xi_sig_gmag_f, ax=axs[0, 1], **args)
_, _, cc_diff = P1.plot_map_of_matrix(40*(xi_sig_org - xi_sig_gmag), mask=(xi_sig_org_f+xi_sig_gmag_f)/2, ax=axs[0, 2], **args_diff)

_, _, _ = P1.plot_map_of_matrix(20* eta_sig_org, mask=eta_sig_org_f, ax=axs[1, 0], **args)
_, _, _ = P1.plot_map_of_matrix(20* eta_sig_gmag, mask=eta_sig_gmag_f, ax=axs[1, 1], **args)
_, _, _ = P1.plot_map_of_matrix(20*(eta_sig_org - eta_sig_gmag), mask=(eta_sig_org_f+eta_sig_gmag_f)/2, ax=axs[1, 2], **args_diff)

for ax in axs[:, 1:].flatten():
    ax.scatter(xi_st, eta_st, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

#pe1 = [mpe.Stroke(linewidth=3, foreground='white',alpha=1), mpe.Normal()]
#for ax in axs.flatten():
#    ax.plot([xi_min-0.003, xi_max+0.0015, xi_max+0.0015, xi_min-0.003, xi_min-0.003], 
#            [eta_mid-eta_window/2, eta_mid-eta_window/2, eta_mid+eta_window/2, eta_mid+eta_window/2, eta_mid-eta_window/2], 
#            color='k', path_effects=pe1, linewidth=2)

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

#axs[0,0].text(0.5, 1.05, 'Gr@R', transform=axs[0,0].transAxes, **args_text)
#axs[0,1].text(0.5, 1.05, 'Gr', transform=axs[0,1].transAxes, **args_text)
#axs[0,2].text(0.5, 1.05, 'Difference', transform=axs[0,2].transAxes, **args_text)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Plot spatial resolution - multi

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
    #PSF = copy.deepcopy(R_multi[:, i].reshape(grid.shape))
    PSF = abs(copy.deepcopy(R_multi[:, i].reshape(grid.shape)))
    
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

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.arange(0, 1200, 50), cbar=False, extend='max')
args_diff = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='bwr', clevels=np.linspace(-300, 300, 16), cbar=False, extend='both')

_, _, cc =      P1.plot_map_of_matrix(40* xi_sig_org, mask=xi_sig_org_f, ax=axs[0, 0], **args)
_, _, _ =       P1.plot_map_of_matrix(40* xi_sig_gmag, mask=xi_sig_gmag_f, ax=axs[0, 1], **args)
_, _, cc_diff = P1.plot_map_of_matrix(40*(xi_sig_org - xi_sig_gmag), mask=(xi_sig_org_f+xi_sig_gmag_f)/2, ax=axs[0, 2], **args_diff)

_, _, _ = P1.plot_map_of_matrix(20* eta_sig_org, mask=eta_sig_org_f, ax=axs[1, 0], **args)
_, _, _ = P1.plot_map_of_matrix(20* eta_sig_gmag, mask=eta_sig_gmag_f, ax=axs[1, 1], **args)
_, _, _ = P1.plot_map_of_matrix(20*(eta_sig_org - eta_sig_gmag), mask=(eta_sig_org_f+eta_sig_gmag_f)/2, ax=axs[1, 2], **args_diff)

#pe1 = [mpe.Stroke(linewidth=3, foreground='white',alpha=1), mpe.Normal()]
#for ax in axs.flatten():
#    ax.plot([xi_min-0.003, xi_max+0.0015, xi_max+0.0015, xi_min-0.003, xi_min-0.003], 
#            [eta_mid-eta_window/2, eta_mid-eta_window/2, eta_mid+eta_window/2, eta_mid+eta_window/2, eta_mid-eta_window/2], 
#            color='k', path_effects=pe1, linewidth=2)

for ax in axs[:, 1:].flatten():
    ax.scatter(xi_st_multi, eta_st_multi, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

cax = fig.add_axes([0.15, 0.07, 0.46, 0.02])
cax_diff = fig.add_axes([0.68, 0.07, 0.19, 0.02])

args_text = dict(va='center', ha='center', fontsize=20)

cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([0, 200, 400, 600, 800, 1000])
cax.text(0.5, -2.5, 'km', transform=cax.transAxes, **args_text)

cbar_diff = fig.colorbar(cc_diff, cax=cax_diff, orientation="horizontal")
cax_diff.set_xticks([-200, 0, 200])
cax_diff.text(0.5, -2.5, 'km', transform=cax_diff.transAxes, **args_text)

axs[0,0].text(-0.1, 0.5, 'Cross-track', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
axs[1,0].text(-0.1, 0.5, 'Along-track', transform=axs[1,0].transAxes, rotation='vertical', **args_text)

axs[0,0].text(0.5, 1.09, 'Without\nground magnetometer', transform=axs[0,0].transAxes, **args_text)
axs[0,1].text(0.5, 1.09, 'With\n10 ground magnetometer', transform=axs[0,1].transAxes, **args_text)
#axs[0,0].text(0.5, 1.05, 'Without G-mag', transform=axs[0,0].transAxes, **args_text)
#axs[0,1].text(0.5, 1.05, 'With 10 G-mag', transform=axs[0,1].transAxes, **args_text)
axs[0,2].text(0.5, 1.05, 'Difference', transform=axs[0,2].transAxes, **args_text)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/resolution_comparison_multi.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Model variance

m_sig_org = np.sqrt(np.diag(Cmp_org).reshape(grid.shape))
m_sig_gmag = np.sqrt(np.diag(Cmp_gmag).reshape(grid.shape))

plt.ioff()
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(2.5e12, 1e13, 40), cbar=False)

_, _, cc = P1.plot_map_of_matrix(m_sig_org.flatten(), ax=axs[0], **args)
_, _, cc = P1.plot_map_of_matrix(m_sig_gmag.flatten(), ax=axs[1], **args)

cax = fig.add_axes([0.15, 0.18, 0.73, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([2.5e12, 5e12, 7.5e12, 1e13])
cax.set_xticklabels(['2.5e12', '5e12', '7.5e12', '1e13'])

args_text = dict(va='center', ha='center', fontsize=20)

cax.text(0.5, -2.5, '[A/m$^2$]', transform=cax.transAxes, **args_text)

axs[0].text(0.5, 1.05, 'Without G-mag', transform=axs[0].transAxes, **args_text)
axs[1].text(0.5, 1.05, 'With G-mag', transform=axs[1].transAxes, **args_text)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Model variance - multi

m_sig_org = np.sqrt(np.diag(Cmp_org).reshape(grid.shape))
m_sig_gmag = np.sqrt(np.diag(Cmp_multi).reshape(grid.shape))

plt.ioff()
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))

args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(2.5e12, 1e13, 40), cbar=False)

_, _, cc = P1.plot_map_of_matrix(m_sig_org.flatten(), ax=axs[0], **args)
_, _, cc = P1.plot_map_of_matrix(m_sig_gmag.flatten(), ax=axs[1], **args)

cax = fig.add_axes([0.15, 0.18, 0.73, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([2.5e12, 5e12, 7.5e12, 1e13])
cax.set_xticklabels(['2.5e12', '5e12', '7.5e12', '1e13'])

args_text = dict(va='center', ha='center', fontsize=20)

cax.text(0.5, -2.5, '[A/m$^2$]', transform=cax.transAxes, **args_text)

axs[0].text(0.5, 1.05, 'Without G-mag', transform=axs[0].transAxes, **args_text)
axs[1].text(0.5, 1.05, 'With 10 G-mag', transform=axs[1].transAxes, **args_text)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_multi.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

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

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d.pdf', format='pdf', bbox_inches='tight')
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

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_diff.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_diff.pdf', format='pdf', bbox_inches='tight')
plt.close('all')
plt.ion()

#%% Model variance projection into d - multi

Ge, Gn, Gu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), np.ones(grid.lon_mesh.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                      grid.lat.flatten(), grid.lon.flatten(), 
                                      current_type = 'divergence_free', RI = RI)

dCmp_org_e = Ge@Cmp_org@Ge.T
dCmp_org_n = Gn@Cmp_org@Gn.T
dCmp_org_u = Gu@Cmp_org@Gu.T

dCmp_gmag_e = Ge@Cmp_multi@Ge.T
dCmp_gmag_n = Gn@Cmp_multi@Gn.T
dCmp_gmag_u = Gu@Cmp_multi@Gu.T

del Ge, Gn, Gu

Be_sig_org = np.sqrt(np.diag(dCmp_org_e))
Bn_sig_org = np.sqrt(np.diag(dCmp_org_n))
Bu_sig_org = np.sqrt(np.diag(dCmp_org_u))

Be_sig_gmag = np.sqrt(np.diag(dCmp_gmag_e))
Bn_sig_gmag = np.sqrt(np.diag(dCmp_gmag_n))
Bu_sig_gmag = np.sqrt(np.diag(dCmp_gmag_u))

plt.ioff()
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(14, 11))

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

'''
args = dict(grid=grid, data=data, t0=t0, t1=t1, RI=RI, cmap='Reds', clevels=np.linspace(0.75, 2.25, 40), cbar=False, mesh=True)

_, _, cc = P1.plot_map_of_matrix(np.log10(Be_sig_org), ax=axs[0, 0], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bn_sig_org), ax=axs[0, 1], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bu_sig_org), ax=axs[0, 2], **args)

_, _, cc = P1.plot_map_of_matrix(np.log10(Be_sig_gmag), ax=axs[1, 0], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bn_sig_gmag), ax=axs[1, 1], **args)
_, _, cc = P1.plot_map_of_matrix(np.log10(Bu_sig_gmag), ax=axs[1, 2], **args)

cax = fig.add_axes([0.15, 0.09, 0.73, 0.02])
cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
cax.set_xticks([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])
'''

for ax in axs[1, :].flatten():
    ax.scatter(xi_st_multi, eta_st_multi, 300,  marker = '*', color='cyan', edgecolor='k', zorder=100)

args_text = dict(va='center', ha='center', fontsize=20)

#cax.text(0.5, -2.5, 'log$_{10}$(B$_i$) [nT]', transform=cax.transAxes, **args_text)
cax.text(0.5, -2.5, 'nT', transform=cax.transAxes, **args_text)

axs[0,0].text(-0.1, 0.5, 'Without G-mag', transform=axs[0,0].transAxes, rotation='vertical', **args_text)
axs[1,0].text(-0.1, 0.5, 'With 10 G-mag', transform=axs[1,0].transAxes, rotation='vertical', **args_text)

axs[0,0].text(0.5, 1.05, 'B$_{e}$', transform=axs[0,0].transAxes, **args_text)
axs[0,1].text(0.5, 1.05, 'B$_{n}$', transform=axs[0,1].transAxes, **args_text)
axs[0,2].text(0.5, 1.05, 'B$_{u}$', transform=axs[0,2].transAxes, **args_text)

plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_multi.png', bbox_inches='tight')
plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/model_variance_d_multi.pdf', format='pdf', bbox_inches='tight')
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


