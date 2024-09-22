#%% Import
import numpy as np
import datetime as dt
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices, CSgrid, CSprojection
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dipole import geo2mag
#from dipole import Dipole
#geo2mag = Dipole.geo2mag
from secsy import spherical 
from EZIE.inversion_code.simulation_utils import get_MHD_jeq, get_MHD_dB, get_MHD_dB_new
import pandas as pd
import os
import EZIE.inversion_code.cases_new as cases
from importlib import reload
import datetime
import pickle
import matplotlib.patheffects as mpe
from kneed import KneeLocator
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, BSpline
from scipy.optimize import curve_fit
import scipy.linalg

import lmfit
from lmfit.lineshapes import gaussian
import copy

#%% get_data_from_case

def get_data_from_case(info):
    
    DT = info['DT']
    timeres = info['timeres']
    d2r = np.pi / 180
    
    # Load data file
    print(info['filename'])
    data = pd.read_pickle(info['filename'])
    
    # convert all geographic coordinates and vector components in data to geomagnetic:
    for i in range(4):
        i = i + 1
        #_, _, data['dbe_measured_' + str(i)], data['dbn_measured_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_measured_' + str(i)].values, data['dbn_measured_' + str(i)].values, epoch = 2020)
        #data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
        data['lat_' + str(i)], data['lon_' + str(i)], data['dbe_' + str(i)], data['dbn_' + str(i)] = geo2mag(data['lat_' + str(i)].values, data['lon_' + str(i)].values, data['dbe_' + str(i)].values, data['dbn_' + str(i)].values, epoch = 2020)
    data['sat_lat'], data['sat_lon'] = geo2mag(data['sat_lat'].values, data['sat_lon'].values, epoch = 2020)
    
    ####
    for key in ['lon_1', 'lon_2', 'lon_3', 'lon_4', 'sat_lon']:
        data[key] = (data[key] - int(info['mlt'])*15)%360
    ####
    
    # calculate SC velocity
    te, tn = spherical.tangent_vector(data['sat_lat'][:-1].values, data['sat_lon'][:-1].values,
                                      data['sat_lat'][1 :].values, data['sat_lon'][1: ].values)    
    data['ve'] = np.hstack((te, np.nan))
    data['vn'] = np.hstack((tn, np.nan))
    
    # get index of central point of analysis interval:
    max_id = np.argmax(data['lat_1'])
    q = abs(data['lat_1'] - info['central_lat'])
    if info['segment'] == 1:        
        q[:max_id] = np.max(q)
    elif info['segment'] == 2:
        q[max_id:] = np.max(q)
    tm = data.index[np.argmin(q)]
                
    #tm = data.index[np.argmin(abs(data['lat_1'] - info['central_lat']))]
    tm = datetime.datetime(tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second)
    #tm = info['tm']
    
    # limits of analysis interval:
    t0 = data.index[data.index.get_loc(tm - dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]
    t1 = data.index[data.index.get_loc(tm + dt.timedelta(seconds = DT//timeres * 60), method = 'nearest')]

    # get unit vectors pointing at satellite (Cartesian vectors)
    rs = []
    for t in [t0, tm, t1]:
        rs.append(np.array([np.cos(data.loc[t, 'sat_lat'] * d2r) * np.cos(data.loc[t, 'sat_lon'] * d2r),
                            np.cos(data.loc[t, 'sat_lat'] * d2r) * np.sin(data.loc[t, 'sat_lon'] * d2r),
                            np.sin(data.loc[t, 'sat_lat'] * d2r)]))

    # Create data dict
    obs = {'lat': [], 'lon': [], 'Be': [], 'Bn': [], 'Bu': [], 'cov_ee': [], 'cov_nn': [], 'cov_uu': [], 'cov_en': [], 'cov_eu': [], 'cov_nu': [], 'time_tag': []}
    for i in range(4):
        obs['lat'] += list(data.loc[t0:t1, 'lat_' + str(i + 1)].values)
        obs['lon'] += list(data.loc[t0:t1, 'lon_' + str(i + 1)].values)
        obs['Be' ] += list(data.loc[t0:t1, 'dbe_measured_'  + str(i + 1)].values * info['signs'][0])
        obs['Bn' ] += list(data.loc[t0:t1, 'dbn_measured_'  + str(i + 1)].values * info['signs'][1])
        obs['Bu' ] += list(data.loc[t0:t1, 'dbu_measured_'  + str(i + 1)].values * info['signs'][2])
        obs['cov_ee'] += list(data.loc[t0:t1, 'cov_ee_' + str(i + 1)].values)
        obs['cov_nn'] += list(data.loc[t0:t1, 'cov_nn_' + str(i + 1)].values)
        obs['cov_uu'] += list(data.loc[t0:t1, 'cov_uu_' + str(i + 1)].values)
        obs['cov_en'] += list(data.loc[t0:t1, 'cov_en_' + str(i + 1)].values)
        obs['cov_eu'] += list(data.loc[t0:t1, 'cov_eu_' + str(i + 1)].values)
        obs['cov_nu'] += list(data.loc[t0:t1, 'cov_nu_' + str(i + 1)].values)
        obs['time_tag'] += list(data.loc[t0:t1].index)

    return data, obs, rs, tm, t0, t1

#%%

def get_projection_variable(data, rs, tm, RI, LRES, WRES, wshift, W = 600, L = 2000):
    # dimensions of analysis region/grid (in km)
    W = W + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    #W = 2500 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    L = L
    #L = 1400
    print(L, W)

    # set up the cubed sphere projection
    v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))
    #angle = np.arctan2(v[1], v[0]) / d2r + np.pi/2 

    orientation = np.array([v[1], -v[0]]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    
    p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
    projection = CSprojection(p, orientation)
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)
    
    return projection, grid


def get_projection(data, rs, tm, RI, LRES, WRES, wshift):
    # dimensions of analysis region/grid (in km)
    W = 600 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    #W = 2500 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    L = 2000
    #L = 1400
    print(L, W)

    # set up the cubed sphere projection
    v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))
    #angle = np.arctan2(v[1], v[0]) / d2r + np.pi/2 

    orientation = np.array([v[1], -v[0]]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    
    p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
    projection = CSprojection(p, orientation)
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)
    
    return projection, grid

def get_projection2(data, rs, tm, RI, LRES, WRES, wshift):
    # dimensions of analysis region/grid (in km)
    W = 1600 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    #W = 2500 + RI * np.arccos(np.sum(rs[0]*rs[-1])) * 1e-3 # km
    L = 2600
    #L = 1400
    print(L, W)

    # set up the cubed sphere projection
    v  = np.array((data.loc[tm, 've'], data.loc[tm, 'vn']))
    #angle = np.arctan2(v[1], v[0]) / d2r + np.pi/2 

    orientation = np.array([v[1], -v[0]]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
    
    p = data.loc[tm, 'sat_lon'], data.loc[tm, 'sat_lat']
    projection = CSprojection(p, orientation)
    grid = CSgrid(projection, L, W, LRES, WRES, wshift = wshift)
    
    return projection, grid

#%%

def get_roughening_matrix(grid):
    Le, Ln = grid.get_Le_Ln()
    LL = Le.T.dot(Le) # matrix for calculation of eastward gradient - eastward in magnetic since all coords above have been converted to dipole coords
    return LL

#%%

def get_covariance_matrix(obs, not_inv=False):
    # construct covariance matrix and invert it
    Wen = np.diagflat(obs['cov_en'])
    Weu = np.diagflat(obs['cov_eu'])
    Wnu = np.diagflat(obs['cov_nu'])
    Wee = np.diagflat(obs['cov_ee'])
    Wnn = np.diagflat(obs['cov_nn'])
    Wuu = np.diagflat(obs['cov_uu'])
    We = np.hstack((Wee, Wen, Weu))
    Wn = np.hstack((Wen, Wnn, Wnu))
    Wu = np.hstack((Weu, Wnu, Wuu))
    if not_inv:
        Q  = np.vstack((We, Wn, Wu))
    else:
        Q  = np.linalg.inv(np.vstack((We, Wn, Wu)))
    return Q

#%%

def do_inversion(r, lat, lon, Br, Btheta, Bphi, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1, gcv=False, Lassi=False, scipy_lsq=False, lapack_driver='gelsd', no_phi=False, no_horizontal=False, no_r=False, no_theta=False):
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack((Bphi, Btheta, Br))
    if no_phi:
        G = np.vstack((Gn, Gu))
        d = np.hstack((Btheta, Br))
    if no_theta:
        G = np.vstack((Ge, Gn))
        d = np.hstack((Bphi, Btheta))
    if no_r:
        G = np.vstack((Ge, Gn))
        d = np.hstack((Bphi, Btheta))
    if no_horizontal:
        G = np.vstack((Gu))
        d = np.hstack((Br))
        Q = Q[2*len(d):, 2*len(d):]

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0400
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0582
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 1727
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 3000

    if scipy_lsq:
        #m, _, _, _ = scipy.linalg.lstsq(GTQG + R, G.T.dot(Q).dot(d), lapack_driver=lapack_driver)
        m = scipy.linalg.solve(GTQG + R, G.T.dot(Q).dot(d))
    else:
        SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
        m = SS.dot(d).flatten()
        m = np.ravel(m)
    
    if gcv:
        res = G@m - d
        num = len(d)*(res@Q@res)
        #num = len(d)*np.sum(res**2)
        denom = np.sum(1 - np.diag(G@SS))**2
        gcv = num/denom
        
        #mnorm = np.sqrt(m@R@m)
        mnorm = np.linalg.norm(m)
        rnorm = np.sqrt(res@Q@res)
        return m, scale, gcv, mnorm, rnorm
    
    if Lassi:
        return Ge, Gn, Gu, Q, LL, scale, Bphi, Btheta, Br, m
    else:
        return m, scale

#%%

def do_inversion_AIC(r, lat, lon, Br, Btheta, Bphi, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1):
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack((Bphi, Btheta, Br))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2

    m, _, _, _ = scipy.linalg.lstsq(GTQG + R, G.T.dot(Q).dot(d), lapack_driver='gelsy')

    return m, scale, Ge, Gn, Gu

#%% Stuff for Pystan

def do_inversion_pystan_data(r, lat, lon, Br, Btheta, Bphi, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1, gcv=False, Lassi=False):
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    d = np.hstack((Bphi, Btheta, Br))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0400
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0582
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 1727
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 3000

    SS = np.linalg.inv(GTQG + R).dot(G.T.dot(Q))
    m = SS.dot(d).flatten()
    m = np.ravel(m)
    
    if gcv:
        res = G@m - d
        num = len(d)*(res@Q@res)
        #num = len(d)*np.sum(res**2)
        denom = np.sum(1 - np.diag(G@SS))**2
        gcv = num/denom
        
        mnorm = np.sqrt(m@R@m)
        #mnorm = np.linalg.norm(m)
        rnorm = np.sqrt(res@Q@res)
        return m, scale, gcv, mnorm, rnorm
    
    return G, np.eye(GTQG.shape[0]), LL, scale, np.abs(LL).max(), d, m
    

#%%

def plot_function_compare(m1, m2, data, grid, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (17, 20))
    axe_true      = plt.subplot2grid((16, 3), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 3), (0 , 1), rowspan = 5)
    axe_secs_2    = plt.subplot2grid((16, 3), (0 , 2), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 3), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 3), (5 , 1), rowspan = 5)
    axn_secs_2    = plt.subplot2grid((16, 3), (5 , 2), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 3), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 3), (10, 1), rowspan = 5)
    axr_secs_2    = plt.subplot2grid((16, 3), (10, 2), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_secs_2.contourf(grid.xi, grid.eta, Gdu.dot(m2).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_secs_2.contourf(grid.xi, grid.eta, Gde.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_secs_2.contourf(grid.xi, grid.eta, Gdn.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e10
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::2, ::2].flatten()
    jlon = grid.lon_mesh[::2, ::2].flatten()    
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    
    je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
    xi, eta, jxi_2, jeta_2 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for (ax_1, ax_2) in zip([axe_secs_1, axn_secs_1, axr_secs_1], [axe_secs_2, axn_secs_2, axr_secs_2]):
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)
        ax_2.quiver(xi, eta, jxi_2, jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid.lat_mesh, grid.lon_mesh, RE + OBSHEIGHT * 1e3, grid.lat[::2, ::2], grid.lon[::2, ::2], RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe, mhdBn, mhdBu)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid.lat[::2, ::2], grid.lon[::2, ::2], current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        #ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    # save plots:
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.png', dpi = 250)
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.pdf')


    # save the relevant parts of the datafile for publication
    #columns = ['lat_1', 'lon_1', 'dbe_1', 'dbn_1', 'dbu_1', 'lat_2', 'lon_2', 'dbe_2', 'dbn_2', 'dbu_2', 'lat_3', 'lon_3', 'dbe_3', 'dbn_3', 'dbu_3', 'lat_4', 'lon_4', 'dbe_4', 'dbn_4', 'dbu_4', 'sat_lat', 'sat_lon', 'dbe_measured_1', 'dbn_measured_1', 'dbu_measured_1', 'cov_ee_1', 'cov_nn_1', 'cov_uu_1', 'cov_en_1', 'cov_eu_1', 'cov_nu_1', 'dbe_measured_2', 'dbn_measured_2', 'dbu_measured_2', 'cov_ee_2', 'cov_nn_2', 'cov_uu_2', 'cov_en_2', 'cov_eu_2', 'cov_nu_2', 'dbe_measured_3', 'dbn_measured_3', 'dbu_measured_3', 'cov_ee_3', 'cov_nn_3', 'cov_uu_3', 'cov_en_3', 'cov_eu_3', 'cov_nu_3', 'dbe_measured_4', 'dbn_measured_4', 'dbu_measured_4', 'cov_ee_4', 'cov_nn_4', 'cov_uu_4', 'cov_en_4', 'cov_eu_4', 'cov_nu_4']
    #savedata = data[t0:t1][columns]
    #savedata.index = dt = (savedata.index-savedata.index[0]).seconds
    #savedata.index.name = 'seconds'
    #savedata.to_csv(info['outputfn'] + 'electrojet_inversion_data.csv')

    #plt.show()
    #return axe_true


    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]

#%% Fewer and larger quivers

def plot_function_compare_2(m1, m2, data, grid, grid2, js, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (17, 20))
    axe_true      = plt.subplot2grid((16, 3), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 3), (0 , 1), rowspan = 5)
    axe_secs_2    = plt.subplot2grid((16, 3), (0 , 2), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 3), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 3), (5 , 1), rowspan = 5)
    axn_secs_2    = plt.subplot2grid((16, 3), (5 , 2), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 3), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 3), (10, 1), rowspan = 5)
    axr_secs_2    = plt.subplot2grid((16, 3), (10, 2), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=7, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_secs_2.contourf(grid.xi, grid.eta, Gdu.dot(m2).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_secs_2.contourf(grid.xi, grid.eta, Gde.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_secs_2.contourf(grid.xi, grid.eta, Gdn.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e4
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m    
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    
    je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m
    xi, eta, jxi_2, jeta_2 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for (ax_1, ax_2) in zip([axe_secs_1, axn_secs_1, axr_secs_1], [axe_secs_2, axn_secs_2, axr_secs_2]):
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)
        ax_2.quiver(xi, eta, jxi_2, jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    mhd_je *= 10**-6 # Go from nA/m to mA/m
    mhd_jn *= 10**-6 # Go from nA/m to mA/m
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 20)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        #ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    # save plots:
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.png', dpi = 250)
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.pdf')


    # save the relevant parts of the datafile for publication
    #columns = ['lat_1', 'lon_1', 'dbe_1', 'dbn_1', 'dbu_1', 'lat_2', 'lon_2', 'dbe_2', 'dbn_2', 'dbu_2', 'lat_3', 'lon_3', 'dbe_3', 'dbn_3', 'dbu_3', 'lat_4', 'lon_4', 'dbe_4', 'dbn_4', 'dbu_4', 'sat_lat', 'sat_lon', 'dbe_measured_1', 'dbn_measured_1', 'dbu_measured_1', 'cov_ee_1', 'cov_nn_1', 'cov_uu_1', 'cov_en_1', 'cov_eu_1', 'cov_nu_1', 'dbe_measured_2', 'dbn_measured_2', 'dbu_measured_2', 'cov_ee_2', 'cov_nn_2', 'cov_uu_2', 'cov_en_2', 'cov_eu_2', 'cov_nu_2', 'dbe_measured_3', 'dbn_measured_3', 'dbu_measured_3', 'cov_ee_3', 'cov_nn_3', 'cov_uu_3', 'cov_en_3', 'cov_eu_3', 'cov_nu_3', 'dbe_measured_4', 'dbn_measured_4', 'dbu_measured_4', 'cov_ee_4', 'cov_nn_4', 'cov_uu_4', 'cov_en_4', 'cov_eu_4', 'cov_nu_4']
    #savedata = data[t0:t1][columns]
    #savedata.index = dt = (savedata.index-savedata.index[0]).seconds
    #savedata.index.name = 'seconds'
    #savedata.to_csv(info['outputfn'] + 'electrojet_inversion_data.csv')

    #plt.show()
    #return axe_true


    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]

#%% Fewer and larger quivers - dark mode - Disputas

def plot_function_compare_disputas(m1, data, grid, grid2, js, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (30, 24))
    axe_true      = plt.subplot2grid((33, 3), (0, 0), rowspan = 15)
    axn_true      = plt.subplot2grid((33, 3), (0, 1), rowspan = 15)
    axr_true      = plt.subplot2grid((33, 3), (0, 2), rowspan = 15)
    
    axe_secs_1    = plt.subplot2grid((33, 3), (15, 0), rowspan = 15)    
    axn_secs_1    = plt.subplot2grid((33, 3), (15, 1), rowspan = 15)
    axr_secs_1    = plt.subplot2grid((33, 3), (15, 2), rowspan = 15)
    
    ax_cbar = plt.subplot2grid((33, 6), (31, 1), colspan=4)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=11, foreground='k',alpha=1), mpe.Normal()]
    for i, tc in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1]:
            ax.plot(xi, eta, color = tc, linewidth = 9, path_effects=pe1)

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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT', fontsize=30)
    ax_cbar.set_yticks([])
    ax_cbar.set_xticks([-1000, -750, -500, -250, 0, 250, 500, 750, 1000])
    ax_cbar.set_xticklabels(ax_cbar.get_xticks().astype(int), fontsize=25)

    scale_j = 1e4
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m    
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for ax_1 in [axe_secs_1, axn_secs_1, axr_secs_1]:
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    mhd_je *= 10**-6 # Go from nA/m to mA/m
    mhd_jn *= 10**-6 # Go from nA/m to mA/m
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = 1.5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = 1.5, zorder = 1)

        ax.axis('off')


    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 35, color='k')


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]


#%% Fewer and larger quivers - dark mode

def plot_function_compare_2_dark(m1, m2, data, grid, grid2, js, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (17, 20))
    axe_true      = plt.subplot2grid((16, 3), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 3), (0 , 1), rowspan = 5)
    axe_secs_2    = plt.subplot2grid((16, 3), (0 , 2), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 3), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 3), (5 , 1), rowspan = 5)
    axn_secs_2    = plt.subplot2grid((16, 3), (5 , 2), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 3), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 3), (10, 1), rowspan = 5)
    axr_secs_2    = plt.subplot2grid((16, 3), (10, 2), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=7, foreground='k',alpha=1), mpe.Normal()]
    for i, tc in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
            ax.plot(xi, eta, color = tc, linewidth = 5, path_effects=pe1)

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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_secs_2.contourf(grid.xi, grid.eta, Gdu.dot(m2).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_secs_2.contourf(grid.xi, grid.eta, Gde.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_secs_2.contourf(grid.xi, grid.eta, Gdn.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e4
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m    
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    
    je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m
    xi, eta, jxi_2, jeta_2 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for (ax_1, ax_2) in zip([axe_secs_1, axn_secs_1, axr_secs_1], [axe_secs_2, axn_secs_2, axr_secs_2]):
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)
        ax_2.quiver(xi, eta, jxi_2, jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    mhd_je *= 10**-6 # Go from nA/m to mA/m
    mhd_jn *= 10**-6 # Go from nA/m to mA/m
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
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
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 20, color='k')


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]


#%% Fewer and larger quivers - with diff

def plot_function_compare_2_wdiff(m1, m2, data, grid, grid2, js, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (28, 25))
    axe_true      = plt.subplot2grid((16, 4), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 4), (0 , 1), rowspan = 5)
    axe_secs_2    = plt.subplot2grid((16, 4), (0 , 2), rowspan = 5)
    axe_diff      = plt.subplot2grid((16, 4), (0 , 3), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 4), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 4), (5 , 1), rowspan = 5)
    axn_secs_2    = plt.subplot2grid((16, 4), (5 , 2), rowspan = 5)
    axn_diff      = plt.subplot2grid((16, 4), (5 , 3), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 4), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 4), (10, 1), rowspan = 5)
    axr_secs_2    = plt.subplot2grid((16, 4), (10, 2), rowspan = 5)
    axr_diff      = plt.subplot2grid((16, 4), (10, 3), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=10, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2, axe_diff, axn_diff, axr_diff]:
            ax.plot(xi, eta, color = 'C' + str(i), linewidth = 7, path_effects=pe1)

    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)


    # set up G matrices for the magnetic field evaluated on a grid - for plotting maps
    Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                          grid.lat.flatten(), grid.lon.flatten(), 
                                          current_type = 'divergence_free', RI = RI)
    
    Be_s1 = Gde.dot(m1)
    Bn_s1 = Gdn.dot(m1)
    Bu_s1 = Gdu.dot(m1)
    
    Be_s2 = Gde.dot(m2)
    Bn_s2 = Gdn.dot(m2)
    Bu_s2 = Gdu.dot(m2)
    
    dBe = Be_s1 - Be_s2
    dBn = Bn_s1 - Bn_s2
    dBu = Bu_s1 - Bu_s2
    
    # get maps of MHD magnetic fields:
    mhdBu =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

    # plot magnetic field in upward direction (MHD and retrieved)    
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Bu_s1.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_secs_2.contourf(grid.xi, grid.eta, Bu_s2.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_diff.contourf(grid.xi, grid.eta, dBu.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Be_s1.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_secs_2.contourf(grid.xi, grid.eta, Be_s2.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_diff.contourf(grid.xi, grid.eta, dBe.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Bn_s1.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_secs_2.contourf(grid.xi, grid.eta, Bn_s2.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_diff.contourf(grid.xi, grid.eta, dBn.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT', fontsize=40)
    #ax_cbar.set_xticks([-1000, -750, -500, -250, 0, 250, 500, 750, 1000])
    ax_cbar.set_xticklabels(ax_cbar.get_xticks().astype(int), fontsize=35)
    ax_cbar.set_yticks([])

    scale_j = 1e4
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m    
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    
    je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m
    xi, eta, jxi_2, jeta_2 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for (ax_1, ax_2) in zip([axe_secs_1, axn_secs_1, axr_secs_1], [axe_secs_2, axn_secs_2, axr_secs_2]):
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)
        ax_2.quiver(xi, eta, jxi_2, jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    for ax in [axe_diff, axn_diff, axr_diff]:
        ax.quiver(xi, eta, jxi_1-jxi_2, jeta_1-jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    #Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    #mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mj = np.zeros(Ge_j.shape[1])
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    mhd_je *= 10**-6 # Go from nA/m to mA/m
    mhd_jn *= 10**-6 # Go from nA/m to mA/m
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2, axe_diff, axn_diff, axr_diff]:
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
#    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2, axe_diff, axn_diff, axr_diff],
#                         ['$\Delta$B$_{\phi}$', '$\Delta$B$_{\u03b8}$', '$\Delta$B$_r$', '$\Delta$B$_{\phi}$', '$\Delta$B$_{\u03b8}$', '$\Delta$B$_r$', '$\Delta$B$_{\phi}$', '$\Delta$B$_{\u03b8}$', '$\Delta$B$_r$', '$\Delta$B$_{\phi}$', '$\Delta$B$_{\u03b8}$', '$\Delta$B$_r$']):
#        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, fontsize = 30)

    for (ax, label) in zip([axe_true, axn_true, axr_true], ['$\Delta$B$_{\phi}$', '$\Delta$B$_{\u03b8}$', '$\Delta$B$_r$']):
        ax.text(-0.05, 0.5, label, ha='right', va='center', transform=ax.transAxes, fontsize=40)

    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2, axe_diff, axn_diff, axr_diff]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2, axe_diff, axn_diff, axr_diff]

#%% Fewer and larger quivers

def plot_function(m1, data, grid, grid2, js, t0, t1, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1]:
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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e10
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for ax_1 in [axe_secs_1, axn_secs_1, axr_secs_1]:
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        #ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    # save plots:
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.png', dpi = 250)
    #plt.savefig('./figures/' + info['outputfn'] + 'inversion_example.pdf')


    # save the relevant parts of the datafile for publication
    #columns = ['lat_1', 'lon_1', 'dbe_1', 'dbn_1', 'dbu_1', 'lat_2', 'lon_2', 'dbe_2', 'dbn_2', 'dbu_2', 'lat_3', 'lon_3', 'dbe_3', 'dbn_3', 'dbu_3', 'lat_4', 'lon_4', 'dbe_4', 'dbn_4', 'dbu_4', 'sat_lat', 'sat_lon', 'dbe_measured_1', 'dbn_measured_1', 'dbu_measured_1', 'cov_ee_1', 'cov_nn_1', 'cov_uu_1', 'cov_en_1', 'cov_eu_1', 'cov_nu_1', 'dbe_measured_2', 'dbn_measured_2', 'dbu_measured_2', 'cov_ee_2', 'cov_nn_2', 'cov_uu_2', 'cov_en_2', 'cov_eu_2', 'cov_nu_2', 'dbe_measured_3', 'dbn_measured_3', 'dbu_measured_3', 'cov_ee_3', 'cov_nn_3', 'cov_uu_3', 'cov_en_3', 'cov_eu_3', 'cov_nu_3', 'dbe_measured_4', 'dbn_measured_4', 'dbu_measured_4', 'cov_ee_4', 'cov_nn_4', 'cov_uu_4', 'cov_en_4', 'cov_eu_4', 'cov_nu_4']
    #savedata = data[t0:t1][columns]
    #savedata.index = dt = (savedata.index-savedata.index[0]).seconds
    #savedata.index.name = 'seconds'
    #savedata.to_csv(info['outputfn'] + 'electrojet_inversion_data.csv')

    #plt.show()
    #return axe_true


    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% Padding

def plot_function_padding(m1, grid, grid2, js, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
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
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e10
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    for ax_1 in [axe_secs_1, axn_secs_1, axr_secs_1]:
        ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        #ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% Fewer and larger quivers

def plot_function_compare_2_padding(m1, m2, grid, grid2, js, info, OBSHEIGHT, RI, RE):
        
    fig = plt.figure(figsize = (17, 20))
    axe_true      = plt.subplot2grid((16, 3), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 3), (0 , 1), rowspan = 5)
    axe_secs_2    = plt.subplot2grid((16, 3), (0 , 2), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 3), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 3), (5 , 1), rowspan = 5)
    axn_secs_2    = plt.subplot2grid((16, 3), (5 , 2), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 3), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 3), (10, 1), rowspan = 5)
    axr_secs_2    = plt.subplot2grid((16, 3), (10, 2), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    #ximin = np.min(grid.xi)
    #ximax = np.max(grid.xi)
    #etamin = np.min(grid.eta)
    #etamax = np.max(grid.eta)

    ximin = np.min(grid2.xi)
    ximax = np.max(grid2.xi)
    etamin = np.min(grid2.eta)
    etamax = np.max(grid2.eta)


    # set up G matrices for the magnetic field evaluated on a grid - for plotting maps
    Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                          grid.lat.flatten(), grid.lon.flatten(), 
                                          current_type = 'divergence_free', RI = RI)

    # get maps of MHD magnetic fields:
    mhdBu =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

    # plot magnetic field in upward direction (MHD and retrieved)    
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_secs_2.contourf(grid.xi, grid.eta, Gdu.dot(m2).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_secs_2.contourf(grid.xi, grid.eta, Gde.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_secs_2.contourf(grid.xi, grid.eta, Gdn.dot(m2).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    scale_j = 1e4
    # calculate the equivalent current of retrieved magnetic field:
    jlat = grid.lat_mesh[::js, ::js].flatten()
    jlon = grid.lon_mesh[::js, ::js].flatten()
    Gje, Gjn = get_SECS_J_G_matrices(jlat, jlon, grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    
    je, jn = Gje.dot(m1).flatten(), Gjn.dot(m1).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m    
    xi, eta, jxi_1, jeta_1 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)
    
    je, jn = Gje.dot(m2).flatten(), Gjn.dot(m2).flatten()
    je *= 10**-6 # Go from nA/m to mA/m
    jn *= 10**-6 # Go from nA/m to mA/m
    xi, eta, jxi_2, jeta_2 = grid.projection.vector_cube_projection(je.flatten(), jn.flatten(), jlon, jlat)

    # plot the equivalent current in the SECS panels:
    #for (ax_1, ax_2) in zip([axe_secs_1, axn_secs_1, axr_secs_1], [axe_secs_2, axn_secs_2, axr_secs_2]):
        #ax_1.quiver(xi, eta, jxi_1, jeta_1, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)
        #ax_2.quiver(xi, eta, jxi_2, jeta_2, linewidth = 2, scale = scale_j, zorder = 40, color = 'black')#, scale = 1e10)

    # calcualte the equivalent current corresponding to MHD output with perfect coverage:
    mhdBu_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe_j =  info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn_j = -info['mhdfunc'](grid2.lat_mesh.flatten(), grid2.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])
    #Ge_Bj, Gn_Bj, Gu_Bj = get_SECS_B_G_matrices(grid2.lat_mesh, grid2.lon_mesh, RE + OBSHEIGHT * 1e3, grid2.lat, grid2.lon, RI = RI)
    #mj = np.linalg.lstsq(np.vstack((Ge_Bj, Gn_Bj, Gu_Bj)), np.hstack((mhdBe_j, mhdBn_j, mhdBu_j)), rcond = 1e-2)[0]
    Ge_Bj = np.zeros((grid2.lat_mesh.size, grid2.lat.size))
    Gn_Bj = np.zeros((grid2.lat_mesh.size, grid2.lat.size))
    Gu_Bj = np.zeros((grid2.lat_mesh.size, grid2.lat.size))
    mj = np.zeros(Ge_Bj.shape[1])

    Ge_j, Gn_j = get_SECS_J_G_matrices(jlat, jlon, grid2.lat, grid2.lon, current_type = 'divergence_free', RI = RI)
    mhd_je, mhd_jn = Ge_j.dot(mj), Gn_j.dot(mj)
    mhd_je *= 10**-6 # Go from nA/m to mA/m
    mhd_jn *= 10**-6 # Go from nA/m to mA/m
    #mhd_je, mhd_jn = get_MHD_jeq(jlat, jlon + info['mapshift'])
    xi, eta, mhd_jxi, mhd_jeta = grid.projection.vector_cube_projection(mhd_je, mhd_jn, jlon, jlat)

    # plot the MHD equivalent current in eaach panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        ax.quiver(xi, eta, mhd_jxi, mhd_jeta, linewidth = 2, scale = scale_j, color = 'grey', zorder = 38)#, scale = 1e10)

    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 20)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]:
        #ax.set_xlim(ximin - 25/(RI * 1e-3), ximax + 25/(RI * 1e-3))
        #ax.set_ylim(etamin + 55/(RI * 1e-3), etamax - 55/(RI * 1e-3))
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1, axe_secs_2, axn_secs_2, axr_secs_2]

#%% Fewer and larger quivers

def plot_function_model_selection(m1, lat_secs, lon_secs, data, grid, t0, t1, info, OBSHEIGHT, RI, RE, singularity_limit=-1):
        
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1]:
            ax.plot(xi, eta, color = 'C' + str(i), linewidth = 5, path_effects=pe1)

    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)

    # set up G matrices for the magnetic field evaluated on a grid - for plotting maps
    if singularity_limit == -1:
        Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                              lat_secs, lon_secs, 
                                              current_type = 'divergence_free', RI = RI)
    else:
        Gde, Gdn, Gdu = get_SECS_B_G_matrices(grid.lat.flatten()+.1, grid.lon.flatten(), np.ones(grid.lon.size) * (6371.2 + OBSHEIGHT) * 1e3, 
                                              lat_secs, lon_secs, 
                                              current_type = 'divergence_free', RI = RI, singularity_limit=singularity_limit)

    # get maps of MHD magnetic fields:
    mhdBu =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], fn = info['mhd_B_fn'])
    mhdBe =  info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])
    mhdBn = -info['mhdfunc'](grid.lat_mesh.flatten(), grid.lon_mesh.flatten() + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])

    # plot magnetic field in upward direction (MHD and retrieved)    
    cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Gdu.dot(m1).reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    axe_secs_1.contourf(grid.xi, grid.eta, Gde.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    axn_secs_1.contourf(grid.xi, grid.eta, Gdn.dot(m1).reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])

    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

        #ax.set_adjustable('datalim') 
        #ax.set_aspect('equal')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)


    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% Only B

def plot_function_B(mhdBu, mhdBn, mhdBe, Bu, Bn, Be, data, grid, t0, t1, info, RI, mesh=False):
        
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    pe1 = [mpe.Stroke(linewidth=6, foreground='white',alpha=1), mpe.Normal()]
    for i in range(4):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        for ax in [axe_secs_1, axn_secs_1, axr_secs_1]:
            ax.plot(xi, eta, color = 'C' + str(i), linewidth = 5, path_effects=pe1)

    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)

    # plot magnetic field in upward direction (MHD and retrieved)
    if mesh:
        cntrs = axr_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bu.reshape(grid.xi_mesh.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    else:
        cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Bu.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    if mesh:        
        axe_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Be.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axe_secs_1.contourf(grid.xi, grid.eta, Be.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    if mesh:        
        axn_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bn.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axn_secs_1.contourf(grid.xi, grid.eta, Bn.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% Only B - for EZIE_v2

def plot_function_B_Ev2(mhdBu, mhdBn, mhdBe, Bu, Bn, Be, b_xi, b_eta, grid, info, RI, mesh=False):
    
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    for ax in [axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.plot(b_xi, b_eta, '.', markersize=6, color='w')
        ax.plot(b_xi, b_eta, '.', markersize=5, color='k')
        
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)

    # plot magnetic field in upward direction (MHD and retrieved)
    if mesh:
        cntrs = axr_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bu.reshape(grid.xi_mesh.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
    else:
        cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Bu.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    if mesh:        
        axe_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Be.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axe_secs_1.contourf(grid.xi, grid.eta, Be.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    if mesh:        
        axn_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bn.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axn_secs_1.contourf(grid.xi, grid.eta, Bn.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% Only B - for EZIE_v2 - circle

def plot_function_B_Ev2_circ(mhdBu, mhdBn, mhdBe, Bu, Bn, Be, b_xi, b_eta, grid, info, RI, mesh=False, ef=0, circ=0):
    
    fig = plt.figure(figsize = (12, 20))
    axe_true      = plt.subplot2grid((16, 2), (0 , 0), rowspan = 5)
    axe_secs_1    = plt.subplot2grid((16, 2), (0 , 1), rowspan = 5)
    
    axn_true      = plt.subplot2grid((16, 2), (5 , 0), rowspan = 5)
    axn_secs_1    = plt.subplot2grid((16, 2), (5 , 1), rowspan = 5)
    
    axr_true      = plt.subplot2grid((16, 2), (10, 0), rowspan = 5)
    axr_secs_1    = plt.subplot2grid((16, 2), (10, 1), rowspan = 5)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    for ax in [axe_true, axn_true, axr_true, axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.plot(b_xi, b_eta, '.', markersize=10, color='w')
        ax.plot(b_xi, b_eta, '.', markersize=9, color='k')
        
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)

    # plot magnetic field in upward direction (MHD and retrieved)
    if mesh:
        #cntrs = axr_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bu.reshape(grid.xi_mesh.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')    
        cntrs = axr_secs_1.tricontourf(grid.xi_mesh.flatten()[ef], grid.eta_mesh.flatten()[ef], Bu[ef], levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        cntrs = axr_secs_1.contourf(grid.xi, grid.eta, Bu.reshape(grid.shape), levels = info['clevels'], cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axr_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBu.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in eastward direction (MHD and retrieved)    
    if mesh:        
        #axe_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Be.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
        axe_secs_1.tricontourf(grid.xi_mesh.flatten()[ef], grid.eta_mesh.flatten()[ef], Be[ef], levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axe_secs_1.contourf(grid.xi, grid.eta, Be.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    
    axe_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBe.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot magnetic field in northward direction (MHD and retrieved)
    if mesh:        
        #axn_secs_1.contourf(grid.xi_mesh, grid.eta_mesh, Bn.reshape(grid.xi_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
        axn_secs_1.tricontourf(grid.xi_mesh.flatten()[ef], grid.eta_mesh.flatten()[ef], Bn[ef], levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    else:
        axn_secs_1.contourf(grid.xi, grid.eta, Bn.reshape(grid.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')
    axn_true.contourf(grid.xi_mesh, grid.eta_mesh, mhdBn.reshape(grid.lat_mesh.shape), levels = cntrs.levels, cmap = plt.cm.bwr, zorder = 0, extend = 'both')

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap = plt.cm.bwr, levels = cntrs.levels)
    ax_cbar.set_xlabel('nT')
    ax_cbar.set_yticks([])
    
    for ax in [axe_true, axn_true, axr_true, axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.plot(circ[0], circ[1], color='w', linewidth=2.5)
        ax.plot(circ[0], circ[1], color='k', linewidth=2)
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [axe_true, axn_true, axr_true, axe_secs_1, axn_secs_1, axr_secs_1]:
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

    # Write labels:
    for ax, label in zip([axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1],
                         ['$\Delta$B$_{\phi}$ MHD', '$\Delta$B$_{\u03b8}$ MHD', '$\Delta$B$_r$ MHD', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS', '$\Delta$B$_{\phi}$ SECS', '$\Delta$B$_{\u03b8}$ SECS', '$\Delta$B$_r$ SECS']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # plot grid in top left panel to show spatial dimensions:
    xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                  np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])
    for i in range(xigrid.shape[0]):
        axe_true.plot(xigrid[i], etagrid[i], 'k-', linewidth = .7)
    for j in range(xigrid.shape[1]):
        axe_true.plot(xigrid[:, j], etagrid[:, j], 'k-', linewidth = .7)

    # set plot limits:
    for ax in [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [axe_true, axn_true, axr_true , axe_secs_1, axn_secs_1, axr_secs_1]

#%% EZIE_v2 - spatial resolution - circle

def plot_function_resolution_Ev2_circ(xi, eta, xi_FWHM, eta_FWHM, b_xi, b_eta, grid, info, RI, mesh=False, ef=0, circ=0, levels=0, mask=0):
    
    fig = plt.figure(figsize = (15, 9))
    ax1 = plt.subplot2grid((16, 2), (0 , 0), rowspan = 15)
    ax2 = plt.subplot2grid((16, 2), (0 , 1), rowspan = 15)
    
    ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    # plot the data tracks:
    for ax in [ax1, ax2]:
        ax.plot(b_xi, b_eta, '.', markersize=10, color='w')
        ax.plot(b_xi, b_eta, '.', markersize=9, color='k')
        
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)

    if isinstance(levels, int):
        vmax = np.max([np.max(xi_FWHM), np.max(eta_FWHM)])
        levels = np.linspace(0, vmax, 40)
    
    if isinstance(mask, int):
        mask = np.ones(xi_FWHM.size).astype(int)

    for (ax, var) in zip([ax1, ax2], [xi_FWHM, eta_FWHM]):
        #var = np.ma.array(var, mask=mask < 0.6)
        cntrs = ax.tricontourf(xi[ef], eta[ef], var[ef], levels=levels, cmap='Reds', zorder = 0)

    # plot colorbar:
    ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap='Reds', levels = cntrs.levels)
    ax_cbar.set_xlabel('km')
    ax_cbar.set_yticks([])
    
    for ax in [ax1, ax2]:
        ax.plot(circ[0], circ[1], color='w', linewidth=2.5)
        ax.plot(circ[0], circ[1], color='k', linewidth=2)
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [ax1, ax2]:
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

    # Write labels:
    for ax, label in zip([ax1, ax2],
                         ['Cross-track', 'Along-track']):
        
        ax.text(ximin - 25/(RI * 1e-3), etamax - 25/(RI * 1e-3), label, va = 'top', ha = 'left', bbox = dict(facecolor='white', alpha=1), zorder = 101, size = 14)

    # set plot limits:
    for ax in [ax1, ax2]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return [ax1, ax2]

#%% EZIE_v2 - J - circle

def plot_function_J_Ev2_circ(lat, lon, mhd_je, mhd_jn, je, jn, b_xi, b_eta, grid, info, RI, mesh=False, ef=0, circ=0, scale=1e3):
    
    fig = plt.figure(figsize = (10, 10))
    ax = plt.gca()
    
    xi, eta, mhd_jxi, mhd_jeta  = grid.projection.vector_cube_projection(mhd_je[ef], mhd_jn[ef], lon[ef], lat[ef])
    _, _, jxi, jeta             = grid.projection.vector_cube_projection(je[ef], jn[ef], lon[ef], lat[ef])
    
    ax.quiver(xi, eta, mhd_jxi, mhd_jeta, width=3e-3, scale=scale, color='tab:orange', zorder=1)
    ax.quiver(xi, eta, jxi, jeta, width=3e-3, scale=scale, color='tab:blue', zorder=2)
    
    # plot the data tracks:
    for ax in [ax]:
        ax.plot(b_xi, b_eta, '.', markersize=15, color='w')
        ax.plot(b_xi, b_eta, '.', markersize=13, color='k')
        
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)
    
    for ax in [ax]:
        ax.plot(circ[0], circ[1], color='w', linewidth=2.5)
        ax.plot(circ[0], circ[1], color='k', linewidth=2)
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [ax]:
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = 'lightgrey', linewidth = .5, zorder = 1)

        ax.axis('off')

    # set plot limits:
    for ax in [ax]:
        ax.set_xlim(ximin, ximax)
        ax.set_ylim(etamin, etamax)
        ax.set_adjustable('datalim') 
        ax.set_aspect('equal')

    # remove whitespace
    plt.subplots_adjust(bottom = .05, top = .99, left = .01, right = .99)

    return ax


#%%

def get_resolution_matrix(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1, andCmp=False):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))

    GTQG = G.T.dot(Q).dot(G)
    GTQ = G.T.dot(Q)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0400
    #R = np.eye(GTQG.shape[0]) * scale*1e0 + LL / np.abs(LL).max() * scale * 1e3 # K 0582
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 1727
    #R = np.eye(GTQG.shape[0]) * scale*1e1 + LL / np.abs(LL).max() * scale * 1e4 # K 3000
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    
    if andCmp:
        Cmp = np.linalg.solve(GTQG + R, np.eye(R.shape[0]))
        R = Cmp.dot(GTQ@G)
        return R, Cmp
    else:
        R = np.linalg.solve(GTQG + R, GTQ@G)        
        #R = scipy.linalg.lstsq(GTQG + R, GTQ@G)[0]
        return R

#%%

def get_resolution_metrics(R, grid):
    
    row_split = grid.shape[1]
    
    x = np.arange(0, grid.shape[1])
    y = np.arange(0, grid.shape[0])
    
    xx, yy = np.meshgrid(x, y)
    
    eta_mu = np.zeros(R.shape[0])
    eta_sig = np.zeros(R.shape[0])
    xi_mu = np.zeros(R.shape[0])
    xi_sig = np.zeros(R.shape[0])
    
    for i in range(R.shape[0]):
        if np.mod(i, 1000) == 0:
            print(i, '/', R.shape[0])
        
        row = int(i/row_split)
        col = i%row_split
        
        PSF = R[:, i]
        
        i_max = np.argmax(PSF)        
        eta_mu[i] = int(i_max/row_split)
        xi_mu[i] = i_max%row_split
        
        PSF = PSF.reshape(grid.shape)
                
        num = np.sum((xx - col)**2 * PSF**2)
        denom = 1e-20 + np.sum(PSF**2)
        xi_sig[i] = np.sqrt(num/denom)
        
        num = np.sum((yy - row)**2 * PSF**2)
        denom = 1e-20 + np.sum(PSF**2)
        eta_sig[i] = np.sqrt(num/denom)
    
    return xi_mu, eta_mu, xi_sig, eta_sig


#%%
def get_L_metric(R, grid):
    
    row_split = grid.shape[1]
    
    x = np.arange(0, grid.shape[1])
    y = np.arange(0, grid.shape[0])
    
    xx, yy = np.meshgrid(x, y)
    
    L = np.zeros(R.shape[0])
    
    for i in range(R.shape[0]):        
        row = int(i/row_split)
        col = i%row_split
        
        PSF = R[:, i]
        
        i_max = np.argmax(PSF)
        
        row_PSF = int(i_max/row_split)
        col_PSF = i_max%row_split
        
        L[i] = np.sqrt((grid.Wres * abs(row - row_PSF))**2 + (grid.Lres * abs(col - col_PSF))**2)
    
    return L

#%%

def plot_map_of_matrix(var, grid, data, t0, t1, RI, cmap = 'Spectral', clevels=1, cbar=True, ax=-1, figsize=(9, 12), extend=None, mesh=False, mask=-1, gridcol='lightgrey', fg='white', track_col=[], fc='k'):

    if ax == -1:
        fig = plt.figure(figsize = figsize)
        ax = plt.subplot2grid((33, 1), (0 , 0), rowspan = 32)
    else:
        fig = -1
    if cbar:
        ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    if mesh:
        var = var.reshape(grid.xi_mesh.shape)
    else:
        var = var.reshape(grid.shape)
        
    vmax = np.max(abs(var))
    vmin = np.min(abs(var))
    if isinstance(clevels, int):
        clevels = np.linspace(vmin, vmax, 20)
    
    fill = False
    if np.all(mask != -1):
        var = np.ma.array(var, mask=mask < 0.6)
        fill = True
    
    if mesh:
        cntrs = ax.contourf(grid.xi_mesh, grid.eta_mesh, var, cmap=cmap, levels=clevels, extend=extend, zorder=0)
    else:    
        cntrs = ax.contourf(grid.xi, grid.eta, var, cmap=cmap, levels=clevels, extend=extend, zorder=0)
    
    # plot the data tracks:
    if len(track_col) == 0:
        track_col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    if len(track_col) == 1:
        track_col = [track_col[0], track_col[0], track_col[0], track_col[0]]
    pe1 = [mpe.Stroke(linewidth=5, foreground=fg, alpha=1), mpe.Normal()]
    for i, tc in enumerate(track_col):
        lon, lat = data.loc[t0:t1, 'lon_' + str(i + 1)].values, data.loc[t0:t1, 'lat_' + str(i + 1)].values
        xi, eta = grid.projection.geo2cube(lon, lat)
        ax.plot(xi, eta, color = tc, linewidth = 3, path_effects=pe1, zorder=2)
    
    # plot grid in top left panel to show spatial dimensions:
        xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                      np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])

    # plot colorbar:
    if cbar:
        ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap=cmap, levels = cntrs.levels)
        ax_cbar.set_yticks([])
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [ax]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = gridcol, linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = gridcol, linewidth = .5, zorder = 1)

        ax.axis('off')
    
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)
    
    if fill:
        ax.fill_between([ximin, ximax], [etamin]*2, [etamax]*2, color=fc, alpha=0.8, zorder=-1)
    
    # set plot limits:
    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)

    ax.set_aspect('equal')
    
    if cbar:
        return fig, ax, ax_cbar, cntrs
    else:
        return fig, ax, cntrs

#%% Find edge of domain

def simple_edge_algo(image, grid):
    ID_pre = simple_scan(image, grid)
    ID_unique = remove_duplicates(ID_pre)
    ID_boundary = sort_boundary(ID_unique)
    return ID_boundary

def brute_edge_algo(image, grid):
    ID_pre = brute_force_scan(image, grid)
    ID_unique = remove_duplicates(ID_pre)
    ID_boundary = sort_boundary(ID_unique)
    return ID_boundary

def simple_scan(image, grid):
    ID_pre = []
    for i in range(grid.shape[0]):
        line = image.reshape(grid.shape)[i, :]
        if np.all(line == 0):
            continue    
        ii = np.argmax(line)
        ID_pre.append([i, ii])    
        ii = grid.shape[1] - np.argmax(np.flip(line)) - 1
        ID_pre.append([i, ii])

    for i in range(grid.shape[1]):
        line = image.reshape(grid.shape)[:, i]
        if np.all(line == 0):
            continue    
        ii = np.argmax(line)
        ID_pre.append([ii, i])    
        ii = grid.shape[0] - np.argmax(np.flip(line)) - 1
        ID_pre.append([ii, i])
    return ID_pre

def brute_force_scan(image, grid):
    ID_pre = []

    for i in range(grid.shape[0]):
        
        if i == 0:
            continue
        if i == (grid.shape[0]-1):
            continue
        
        for j in range(grid.shape[1]):
        
            if j == 0:
                continue
            if j == (grid.shape[1]-1):
                continue
        
            if image.reshape(grid.shape)[i, j] == 0:
                continue
            
            if np.any(image.reshape(grid.shape)[i-1:i+2, j-1:j+2].flatten() == 0):
                ID_pre.append([i, j])                      

    # Get row edges
    for i in [0, grid.shape[0]-1]:
        line = image.reshape(grid.shape)[i, :]
        for ii in range(len(line)):
            if line[ii] == 1:
                ID_pre.append([i, ii])
    
    # Get col edges
    for i in [0, grid.shape[1]-1]:
        line = image.reshape(grid.shape)[:, i]
        for ii in range(len(line)):
            if line[ii] == 1:
                ID_pre.append([ii, i])
    
    return ID_pre

def remove_duplicates(ID_pre):
    ID_unique = []
    for i, c_i in enumerate(ID_pre):
        if i == 0:
            ID_unique.append(c_i)
            continue
        unique = True
        for b_j in ID_unique:
            if c_i == b_j:
                unique = False
        if unique:
            ID_unique.append(c_i)
    return ID_unique

def sort_boundary(ID_unique):
    ID_boundary = [ID_unique[0]]
    ID_unique.pop(0)
    while len(ID_unique) > 1:
        row, col = ID_boundary[-1]
        candidates = np.array(ID_unique)
        ii = np.argmin(np.sqrt((20*(candidates[:, 0] - row))**2 + (40*(candidates[:, 1] - col))**2))
        ID_boundary.append(ID_unique[ii])
        ID_unique.pop(ii)
    ID_boundary.append(ID_unique[0])
    ID_boundary.append(ID_boundary[0])
    return ID_boundary

'''
def simple_edge_algo(image, grid):
    ID_pre = []
    for i in range(grid.shape[0]):
        line = image.reshape(grid.shape)[i, :]
        if np.all(line == 0):
            continue    
        ii = np.argmax(line)
        ID_pre.append([i, ii])    
        ii = grid.shape[1] - np.argmax(np.flip(line)) - 1
        ID_pre.append([i, ii])

    for i in range(grid.shape[1]):
        line = image.reshape(grid.shape)[:, i]
        if np.all(line == 0):
            continue    
        ii = np.argmax(line)
        ID_pre.append([ii, i])    
        ii = grid.shape[0] - np.argmax(np.flip(line)) - 1
        ID_pre.append([ii, i])

    # Remove duplicates
    ID_unique = []
    for i, c_i in enumerate(ID_pre):
        if i == 0:
            ID_unique.append(c_i)
            continue
        unique = True
        for b_j in ID_unique:
            if c_i == b_j:
                unique = False
        if unique:
            ID_unique.append(c_i)

    # Sort boundary:
    ID_boundary = [ID_unique[0]]
    ID_unique.pop(0)
    while len(ID_unique) > 1:
        row, col = ID_boundary[-1]
        candidates = np.array(ID_unique)
        ii = np.argmin(np.sqrt((20*(candidates[:, 0] - row))**2 + (40*(candidates[:, 1] - col))**2))
        ID_boundary.append(ID_unique[ii])
        ID_unique.pop(ii)
    ID_boundary.append(ID_unique[0])
    ID_boundary.append(ID_boundary[0])
        
    return ID_boundary
'''

#%%

def plot_map_of_matrix_padding(var, grid, RI, cmap = 'Spectral', clevels=1, cbar=True, ax=-1, figsize=(9, 12), extend=None, mesh=False, mask=-1, gridcol='lightgrey', fg='white'):

    if ax == -1:
        fig = plt.figure(figsize = figsize)
        ax = plt.subplot2grid((33, 1), (0 , 0), rowspan = 32)
    else:
        fig = -1
    if cbar:
        ax_cbar = plt.subplot2grid((33, 2), (32, 0), colspan = 2)
    
    if mesh:
        var = var.reshape(grid.xi_mesh.shape)
    else:
        var = var.reshape(grid.shape)
        
    vmax = np.max(abs(var))
    vmin = np.min(abs(var))
    if isinstance(clevels, int):
        clevels = np.linspace(vmin, vmax, 20)
    
    fill = False
    if np.all(mask != -1):
        var = np.ma.array(var, mask=mask < 0.6)
        fill = True
    
    if mesh:
        cntrs = ax.contourf(grid.xi_mesh, grid.eta_mesh, var, cmap=cmap, levels=clevels, extend=extend, zorder=0)
    else:    
        cntrs = ax.contourf(grid.xi, grid.eta, var, cmap=cmap, levels=clevels, extend=extend, zorder=0)
    
    # plot grid in top left panel to show spatial dimensions:
        xigrid, etagrid = np.meshgrid(np.r_[grid.xi.min():grid.xi.max() + 200/RI*1e3:200/RI*1e3], 
                                      np.r_[grid.eta.min():grid.eta.max() + 200/RI*1e3:200/RI*1e3])

    # plot colorbar:
    if cbar:
        ax_cbar.contourf(np.vstack((cntrs.levels, cntrs.levels)), np.vstack((np.zeros(cntrs.levels.size), np.ones(cntrs.levels.size))), np.vstack((cntrs.levels, cntrs.levels)), cmap=cmap, levels = cntrs.levels)
        ax_cbar.set_yticks([])
    
    # plot coordinate grids, fix aspect ratio and axes in each panel
    for ax in [ax]:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for l in np.r_[60:90:5]:
            xi, eta = grid.projection.geo2cube(np.linspace(0, 360, 360), np.ones(360)*l)
            ax.plot(xi, eta, color = gridcol, linewidth = .5, zorder = 1)

        for l in np.r_[0:360:15]:
            xi, eta = grid.projection.geo2cube(np.ones(360)*l, np.linspace(50, 90, 360))
            ax.plot(xi, eta, color = gridcol, linewidth = .5, zorder = 1)

        ax.axis('off')
    
    ximin = np.min(grid.xi)
    ximax = np.max(grid.xi)
    etamin = np.min(grid.eta)
    etamax = np.max(grid.eta)
    
    if fill:
        ax.fill_between([ximin, ximax], [etamin]*2, [etamax]*2, color='k', alpha=0.8, zorder=-1)
    
    # set plot limits:
    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)

    ax.set_aspect('equal')
    
    if cbar:
        return fig, ax, ax_cbar, cntrs
    else:
        return fig, ax, cntrs


#%% PSF fit

from functools import partial
from multiprocessing import Pool

def gaussian2d_rot(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 rotation=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(rotation) - (y - centery)*np.sin(rotation)
    yp = (x - centerx)*np.sin(rotation) + (y - centery)*np.cos(rotation)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*gaussian(R)/(np.pi*sigmax*sigmay)

def do_nonlinear_PSF_fit(grid, PSF):
    # Define grid
    x = np.arange(0, grid.lat.shape[1])
    y = np.arange(0, grid.lat.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Initial guess
    model = lmfit.models.Gaussian2dModel()
    params = model.guess(abs(PSF.flatten()), x.flatten(), y.flatten())

    # Define model with rotation and parameter boundaries
    model = lmfit.Model(gaussian2d_rot, independent_vars=['x', 'y'])
    model.set_param_hint('amplitude', min=0, max=5*params['amplitude'], vary=True)
    model.set_param_hint('centerx', min=0, max=grid.xi.shape[1], vary=True)
    model.set_param_hint('centery', min=0, max=grid.xi.shape[0], vary=True)
    model.set_param_hint('sigmax', min=1e-20, max=25, vary=True)
    model.set_param_hint('sigmay', min=1e-20, max=60, vary=True)
    model.set_param_hint('rotation', min=-np.pi/2, max=np.pi/2, vary=True)
    params = model.make_params(amplitude=params['amplitude'], centerx=params['centerx'], centery=params['centery'], sigmax=params['sigmax'], sigmay=params['sigmay'])
    #model.set_param_hint('rotation', min=-np.pi/2, max=np.pi/2, vary=False)
    #params = model.make_params(amplitude=params['amplitude'], centerx=params['centerx'], centery=params['centery'], sigmax=params['sigmax'], sigmay=params['sigmay'], rotation=0)
    
    # Run model
    result = model.fit(abs(PSF.flatten()), x=x.flatten(), y=y.flatten(), params=params, method='dual_annealing')

    return result

def do_nonlinear_PSF_fit_2(grid, PSF):
    # Define grid
    x = grid.xi.flatten()
    y = grid.eta.flatten()
    
    # Initial guess
    model = lmfit.models.Gaussian2dModel()
    params = model.guess(abs(PSF.flatten()), x, y)

    # Define model with rotation and parameter boundaries
    model = lmfit.Model(gaussian2d_rot, independent_vars=['x', 'y'])
    model.set_param_hint('amplitude', min=0, max=5*params['amplitude'], vary=True)
    model.set_param_hint('centerx', min=grid.xi.min(), max=grid.xi.max(), vary=True)
    model.set_param_hint('centery', min=grid.eta.min(), max=grid.eta.max(), vary=True)
    model.set_param_hint('sigmax', min=1e-20, max=1, vary=True)
    model.set_param_hint('sigmay', min=1e-20, max=1, vary=True)
    model.set_param_hint('rotation', min=-np.pi/2, max=np.pi/2, vary=True)
    params = model.make_params(amplitude=params['amplitude'], centerx=params['centerx'], centery=params['centery'], sigmax=params['sigmax'], sigmay=params['sigmay'])
    #model.set_param_hint('rotation', min=-np.pi/2, max=np.pi/2, vary=False)
    #params = model.make_params(amplitude=params['amplitude'], centerx=params['centerx'], centery=params['centery'], sigmax=params['sigmax'], sigmay=params['sigmay'], rotation=0)
    
    # Run model
    result = model.fit(abs(PSF.flatten()), x=x, y=y, params=params, method='dual_annealing')

    return result

def get_lmfit_from_R(R, grid, parallel=False):
    
    if parallel:
        params = np.zeros((R.shape[0], 6))
        for i in range(R.shape[0]):
            print(i)
            
            PSF = abs(copy.deepcopy(R[:, i].reshape(grid.shape)))
            result = do_nonlinear_PSF_fit(PSF, grid)
            
            params[i, 0] = result.best_values['amplitude']
            params[i, 1] = result.best_values['centerx']
            params[i, 2] = result.best_values['centery']
            params[i, 3] = result.best_values['sigmax']
            params[i, 4] = result.best_values['sigmay']
            params[i, 5] = result.best_values['rotation']
    else:
        PSFs = []
        for i in range(R.shape[0]):
            PSFs.append(abs(R[:, i].reshape(grid.shape)))
    
        func = partial(do_nonlinear_PSF_fit, grid)
        pool = Pool(10)
        result = pool.map(func, PSFs)
        pool.close()
        pool.join()
    
        params = np.zeros((R.shape[0], 6))
        for i in range(R.shape[0]):
            params[i, 0] = result[i].best_values['amplitude']
            params[i, 1] = result[i].best_values['centerx']
            params[i, 2] = result[i].best_values['centery']
            params[i, 3] = result[i].best_values['sigmax']
            params[i, 4] = result[i].best_values['sigmay']
            params[i, 5] = result[i].best_values['rotation']
    
    return params

#%% Find Full Width Half Max indices

def left_right(PSF_i, fraq=0.5, inside=False, x='', x_min='', x_max=''):
    
    if inside:
        PSF_ii = copy.deepcopy(PSF_i)
        valid = False
        while not valid:
            i_max = np.argmax(PSF_ii)
            
            x_i = x[i_max]
            if (x_i >= x_min) and (x_i <= x_max):
                valid = True
            else:
                PSF_ii[i_max] = np.min(PSF_i)            
                        
    else:
        i_max = np.argmax(PSF_i)    
    
    PSF_max = PSF_i[i_max]
        
    j = 0
    i_left = 0
    left_edge = True
    while (i_max - j) >= 0:
        if PSF_i[i_max - j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
            dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
            i_left = i_max - j + dx
            
            left_edge = False
            
            break
        else:
            j += 1

    j = 0
    i_right = len(PSF_i) - 1
    right_edge = True
    while (i_max + j) < len(PSF_i):
        if PSF_i[i_max + j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
            dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
            i_right = i_max + j - 1 + dx 
            
            right_edge = False
            
            break
        else:
            j += 1
    
    flag = True
    if left_edge and right_edge:
        print('I think something is wrong')
        flag = False
    elif left_edge:
        i_left = i_max - (i_right - i_max)
        flag = False
    elif right_edge:
        i_right = i_max + (i_max - i_left)
        flag = False
    
    return i_left, i_right, flag

#%%

def get_posterior_model_covariance(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2

    #Cmp = np.linalg.inv(GTQG + R)
    Cmp, _, _, _ = scipy.linalg.lstsq(GTQG + R, np.eye(R.shape[0]), lapack_driver='gelsy')

    return Cmp

def error_covariance_matrix(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    
    Cp = np.linalg.inv(R)
    
    q = Cp - Cp@G.T@np.linalg.inv(G@Cp@G.T + np.linalg.inv(Q))@G@Cp
    
    return q

def stuff(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    
    return np.linalg.inv(GTQG + R)  

def stuff_2(r, lat, lon, lat_secs, lon_secs, LL, Q, RI, current_type = 'divergence_free', l1=1e0, l2=1e3, scale=-1):
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, current_type = current_type, RI = RI)
    G = np.vstack((Ge, Gn, Gu))

    GTQG = G.T.dot(Q).dot(G)
    
    if scale == -1:
        #scale = np.max(abs(GTQG))
        scale = np.median(abs(GTQG))
        
    R = np.eye(GTQG.shape[0]) * scale*l1 + LL / np.abs(LL).max() * scale * l2
    
    return 1/(GTQG + R), (GTQG + R)

#%% Combine lambda

def combine_lambda_analysis(info, LRES, WRES, RI, wshift, OBSHEIGHT, gmag=False, row_st=54, col_st=27):
    # Get information
    data, obs, rs, tm, t0, t1 = get_data_from_case(info)

    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    
    LL = get_roughening_matrix(grid)
    
    ####
    if gmag:
        xi_st = grid.xi_mesh[row_st, col_st]
        eta_st = grid.eta_mesh[row_st, col_st]
    
        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    ###

    Q = get_covariance_matrix(obs)

    # Loop it
    #l1_list = np.arange(-3, 3.1, 0.1)
    l1_list = np.array([np.linspace(-3, 3, 100)[10]])
    l2_list = np.linspace(-5, 10, 200)

    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    GTQG = G.T.dot(Q).dot(G)
    scale = np.median(abs(GTQG))
    #scale = np.max(abs(GTQG))
    
    Rm = np.zeros((len(l1_list), len(l2_list), grid.shape[0], grid.shape[1]))
    mags = np.zeros((len(l1_list), len(l2_list)))
    for i, l1 in enumerate(l1_list):
        print (i+1, '/', len(l1_list))
        func = partial(par_func, GTQG, scale, l1, LL, grid)
        pool = Pool(2)
        result = pool.map(func, l2_list)
        pool.close()
        pool.join()
    
        for j in range(len(l2_list)):
            Rm[i, j, :, :] = result[j][0]
            mags[i, j] = result[j][1]
    
    return Rm, mags

def par_func(GTQG, scale, l1, LL, grid, l2):
    regularization = np.eye(GTQG.shape[0]) * scale * 10**l1 + LL / np.abs(LL).max() * scale * 10**l2
    #R = np.linalg.inv(GTQG + regularization)@GTQG
    #R = abs(np.linalg.inv(GTQG + regularization)@GTQG)
    R = np.linalg.solve(GTQG+regularization, GTQG)
    R = abs(R)
    Rm = np.sum(R, axis=1)
    Rm = Rm.reshape(grid.shape)
    return Rm, np.sum(abs(R))

#%% LL test

def combine_lambda_analysis_LL_test(info, LRES, WRES, RI, wshift, OBSHEIGHT, gmag=False, row_st=54, col_st=27):
    # Get information
    data, obs, rs, tm, t0, t1 = get_data_from_case(info)

    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    
    LL = get_roughening_matrix(grid)

    ####
    if gmag:
        xi_st = grid.xi_mesh[row_st, col_st]
        eta_st = grid.eta_mesh[row_st, col_st]
    
        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    ###

    Q = get_covariance_matrix(obs)
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3, 
                                       grid.lat.flatten(), grid.lon.flatten(), current_type = 'divergence_free', RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    GTQG = G.T.dot(Q).dot(G)
    scale = np.median(abs(GTQG))
    
    l1_list = np.array([np.linspace(-3, 3, 100)[10]])
    l2_list = np.linspace(-5, 10, 200)
    
    mags = np.zeros((len(l1_list), len(l2_list)))
    for i, l1 in enumerate(l1_list):
        for j, l2 in enumerate(l2_list):
            regularization = np.eye(GTQG.shape[0]) * scale * 10**l1 + LL / np.abs(LL).max() * scale * 10**l2
            mags[i, j] = np.sum(abs(np.linalg.inv(regularization)))
    
    return mags

#%% Rm data for SSH
def combine_lambda_analysis_SSH(info, LRES, WRES, RI, wshift, OBSHEIGHT, gmag=False, row_st=54, col_st=27, gmag_multi=False, randomizer=0, case=0, lat_shift=0):
    # Get information
    data, obs, rs, tm, t0, t1 = get_data_from_case(info)

    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    
    LL = get_roughening_matrix(grid)
    
    ####
    if gmag:
        xi_st = grid.xi_mesh[row_st, col_st]
        eta_st = grid.eta_mesh[row_st, col_st]
    
        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    elif gmag_multi:
        n_st = 10

        np.random.seed(1337+randomizer)
        row_st = np.ones(n_st).astype(int)
        while len(np.unique(row_st)) != n_st:
            row_st = np.random.uniform(10, 98, n_st).astype(int)
        col_st = np.ones(n_st).astype(int)
        while len(np.unique(col_st)) != n_st:    
            col_st = np.random.uniform(12, 37, n_st).astype(int)

        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    
    else:
        r = np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3
    ###

    Q = get_covariance_matrix(obs)

    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], r,
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = RI)
    G = np.vstack((Ge, Gn, Gu))
    GTQG = G.T.dot(Q).dot(G)
    
    if gmag:
        file = 'gmag'
    elif gmag_multi:
        file = 'gmag_multi'
    else:
        file = 'org'
     
    postfix = ''
    if case != 0:
        postfix +='_c{}'.format(case)
    if lat_shift != 0:
        postfix +='_l{}'.format(lat_shift)
    if randomizer != 0:
        postfix +='_r{}'.format(randomizer)
     
    np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/GTQG{}.npy'.format(file, postfix), GTQG)
    np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/LL{}.npy'.format(file, postfix), LL)
        
    ''' 
    if randomizer != 0:
        np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/GTQG_r{}.npy'.format(file, randomizer), GTQG)
        np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/LL_r{}.npy'.format(file, randomizer), LL)
    else:
        np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/GTQG.npy'.format(file), GTQG)
        np.save('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/LL.npy'.format(file), LL)
    '''
    #ff = open('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/Rm_{}/grid.pkl'.format(file), 'wb')
    #pickle.dump(grid, ff)
    #ff.close()

#%% L_curve and GCV

def calc_curvature(rnorm, mnorm):
    x_t = np.gradient(rnorm)
    y_t = np.gradient(mnorm)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)
    curvature = (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    return curvature

def L_curve_and_GCV(info, LRES, WRES, RI, wshift, OBSHEIGHT, l1_gcv, l2_fit, gmag=False, ground_file='', row_st=54, col_st=27, gmag_multi=False, gcv_f=False):
    
    # Get information
    data, obs, rs, tm, t0, t1 = get_data_from_case(info)

    #projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    projection, grid = get_projection(data, rs, tm, RI, LRES, WRES, wshift)
    
    LL = get_roughening_matrix(grid)
     
    ####
    if gmag:
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
    elif gmag_multi:
        n_st = 10

        np.random.seed(1337)
        row_st = np.ones(n_st).astype(int)
        while len(np.unique(row_st)) != n_st:
            row_st = np.random.uniform(10, 98, n_st).astype(int)
        col_st = np.ones(n_st).astype(int)
        while len(np.unique(col_st)) != n_st:    
            col_st = np.random.uniform(12, 37, n_st).astype(int)

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
    else:
        r = np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3
    ###

    Q = get_covariance_matrix(obs)
    
    d = np.hstack((obs['Be'], obs['Bn'], obs['Bu']))
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], r, 
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = RI)
    
    G = np.vstack((Ge, Gn, Gu))
    GTQG = G.T.dot(Q).dot(G)
    GTQ = G.T.dot(Q)
    GTQd = GTQ@d
    scale = np.median(abs(GTQG))
    LL_L_curve = np.eye(GTQG.shape[0]) * scale + LL / np.abs(LL).max() * scale
    
    gcv = np.zeros(len(l1_gcv))
    mnorm = np.zeros(len(l1_gcv))
    rnorm = np.zeros(len(l1_gcv))    
    for i, (l1, l2) in enumerate(zip(l1_gcv, l2_fit)):
        print(i+1, '/', len(l1_gcv))
        regularization = np.eye(GTQG.shape[0]) * scale * 10**l1 + LL / np.abs(LL).max() * scale * 10**l2
        m = np.linalg.solve(GTQG + regularization, GTQd)
        #SS = np.linalg.inv(GTQG + regularization)@GTQ
        #m = SS@d
        
        res = G@m - d
        if gcv_f:
            SS = np.linalg.inv(GTQG + regularization)@GTQ
            num = len(d)*(res@Q@res)
            denom = np.sum(1 - np.diag(G@SS))**2
            gcv[i] = num/denom
                
        mnorm[i] = np.sqrt(np.sum(m@LL_L_curve@m))
        
        #qqq = np.eye(GTQG.shape[0]) * scale
        #mnorm[i] = np.sqrt(np.sum(m@qqq@m))
        #qqq = LL / np.abs(LL).max() * scale
        #mnorm[i] += np.sqrt(np.sum(m@qqq@m))
        
        
        rnorm[i] = np.sqrt(np.sum(res@Q@res))        
            
    return gcv, rnorm, mnorm
    
#%% GCV L-curve tester

def GCV_L_curve_tester(info, LRES, WRES, RI, wshift, OBSHEIGHT, l1_gcv, l2_fit, gmag=False, row_st=54, col_st=27, gmag_multi=False, l_curve=False, sep=False):
    # Get information
    data, obs, rs, tm, t0, t1 = get_data_from_case(info)

    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    
    LL = get_roughening_matrix(grid)
     
    ####
    if gmag:
        xi_st = grid.xi_mesh[row_st, col_st]
        eta_st = grid.eta_mesh[row_st, col_st]
    
        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    elif gmag_multi:
        n_st = 10

        np.random.seed(1337)
        row_st = np.ones(n_st).astype(int)
        while len(np.unique(row_st)) != n_st:
            row_st = np.random.uniform(10, 98, n_st).astype(int)
        col_st = np.ones(n_st).astype(int)
        while len(np.unique(col_st)) != n_st:    
            col_st = np.random.uniform(12, 37, n_st).astype(int)

        lat_st = grid.lat_mesh[row_st, col_st]
        lon_st = grid.lon_mesh[row_st, col_st]

        mhdBu =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], fn = info['mhd_B_fn'])[0]
        mhdBe =  info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Bphi [nT]', fn = info['mhd_B_fn'])[0]
        mhdBn = -info['mhdfunc'](np.array([lat_st]), np.array([lon_st]) + info['mapshift'], component = 'Btheta [nT]', fn = info['mhd_B_fn'])[0]

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
    else:
        r = np.ones_like(obs['lat']) * (6371.2 + OBSHEIGHT) * 1e3
    ###

    Q = get_covariance_matrix(obs)
    
    d = np.hstack((obs['Be'], obs['Bn'], obs['Bu']))
    
    Ge, Gn, Gu = get_SECS_B_G_matrices(obs['lat'], obs['lon'], r, 
                                       grid.lat.flatten(), grid.lon.flatten(), 
                                       current_type = 'divergence_free', RI = RI)
    
    G = np.vstack((Ge, Gn, Gu))
    GTQG = G.T.dot(Q).dot(G)
    GTQ = G.T.dot(Q)
    scale = np.median(abs(GTQG))
    
    
    gamma = np.linspace(0, 1, 200)
    gcv = np.zeros(len(l1_gcv))
    rgcv = np.zeros((len(l1_gcv), len(gamma)))
    if l_curve:
        mnorm = np.zeros(len(l1_gcv))
        rnorm = np.zeros(len(l1_gcv))
    if sep:
        rnorm_EZIE = np.zeros(len(l1_gcv))
        rnorm_gmag = np.zeros(len(l1_gcv))
        
    for i, (l1, l2) in enumerate(zip(l1_gcv, l2_fit)):
        print(i+1, '/', len(l1_gcv))
        regularization = np.eye(GTQG.shape[0]) * scale * 10**l1 + LL / np.abs(LL).max() * scale * 10**l2
        SS = np.linalg.inv(GTQG + regularization)@GTQ
        m = SS@d
        
        res = G@m - d
        num = len(d)*(res@Q@res)
        denom = np.sum(1 - np.diag(G@SS))**2
        gcv[i] = num/denom
        
        k = len(d)*np.trace((G@SS)**2)
        for j, g_i in enumerate(gamma):
            rgcv[i, j] = gcv[i]*g_i + (1-g_i)*gcv[i]*k        
        
        if l_curve:
            regularization = np.eye(GTQG.shape[0]) * scale + LL / np.abs(LL).max() * scale
            mnorm[i] = np.sqrt(np.sum(m@regularization@m))
            rnorm[i] = np.sqrt(np.sum(res@Q@res))
        if sep:
            #Q_EZIE = copy.deepcopy(Q)
            #Q_EZIE[Q_EZIE == 1] = 0
            #rnorm_EZIE[i] = np.sqrt(np.sum(res@Q_EZIE@res))
            
            #Q_gmag = copy.deepcopy(Q)
            #Q_gmag[Q_gmag != 1] = 0
            #rnorm_gmag[i] = np.sqrt(np.sum(res@Q_gmag@res))
            
            res_q = copy.deepcopy(res).reshape((3, int(len(res)/3))).T            
            res_q[-10:, :] = 0
            res_q=res_q.flatten()
            Q_q = copy.deepcopy(Q)
            Q_q[Q_q == 1] = 0
            rnorm_EZIE[i] = np.sqrt(np.sum(res_q@Q_q@res_q))
            
            res_q = copy.deepcopy(res).reshape((3, int(len(res)/3))).T
            res_q[:-10, :] = 0
            res_q=res_q.flatten()
            Q_q = copy.deepcopy(Q)
            Q_q[Q_q != 1] = 0
            rnorm_gmag[i] = np.sqrt(np.sum(res_q@Q_q@res_q))
            
    if l_curve and sep:
        return gcv, rgcv, mnorm, rnorm, rnorm_EZIE, rnorm_gmag
    elif l_curve:
        return gcv, rgcv, mnorm, rnorm
    else:
        return gcv, rgcv, mnorm, rnorm

#%% Kn robust
def robust_Kneedle(rnorm, mnorm):
    kn_id = 0
    i = 0
    while kn_id < 1:
        kn = KneeLocator(np.log10(rnorm[i:]), np.log10(mnorm[i:]), curve='convex', direction='decreasing')
        kn_id = np.argmin(abs((np.log10(rnorm) - kn.knee)))
        i += 1
    return kn_id, i-1

#%% Home brew l_curve thingy
def home_brew(rnorm, mnorm, fig=False):
    #rnorm = copy.deepcopy(rnorm_multi)
    #mnorm = copy.deepcopy(mnorm_multi)

    curv = calc_curvature(rnorm, mnorm)
    i_left, i_right, flag = left_right(-curv, fraq=0.1)

    knees = []
    for i in range(int(i_left)):
        kn = KneeLocator(np.log10(rnorm[i:]), np.log10(mnorm[i:]), curve='convex', direction='decreasing')
        knees.append(np.argmin(abs((np.log10(rnorm) - kn.knee))))

    knees = np.array(knees)
    knees = knees[(knees > i_left) & (knees < i_right)]
    #kn_id = int(np.median(knees))
    kn_id = knees[0]

    if fig:
        plt.figure()
        plt.loglog(rnorm, mnorm)
        plt.loglog(rnorm[int(i_left):int(i_right)+1], mnorm[int(i_left):int(i_right)+1])
        plt.loglog(rnorm[knees], mnorm[knees], '.')
        plt.loglog(rnorm[kn_id], mnorm[kn_id], '*')
    
    return kn_id

#%% Find parameter furthest away from data
def find_parameter_with_longest_distance(obs, data, rs, tm, RI, LRES, WRES, wshift):
    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    
    xi, eta = grid.projection.geo2cube(obs['lon'], obs['lat'])
    
    xi_max = np.max(xi)
    xi_min = np.min(xi)
    eta_max = np.max(eta)
    eta_min = np.min(eta)

    deta = eta_max - eta_min
    eta_mid = eta_min + (deta)/2
    eta_window = deta/3

    f = (grid.xi.flatten() > xi_min) & (grid.xi.flatten() < xi_max) & (grid.eta.flatten() > eta_mid-eta_window/2) & (grid.eta.flatten() < eta_mid+eta_window/2)

    dists = np.zeros((len(grid.xi.flatten()), len(obs['lat'])))
    obs_xi, obs_eta = grid.projection.geo2cube(obs['lon'], obs['lat'])
    for i in range(dists.shape[1]):
        dists[:, i] = np.sqrt((grid.xi.flatten() - obs_xi[i])**2 + (grid.eta.flatten() - obs_eta[i])**2)

    dists_sort = np.zeros(dists.shape)
    for i in range(dists.shape[0]):
        dists_sort[i, :] = np.sort(dists[i, :])

    q = copy.deepcopy(dists_sort[:, 0])

    q[~f] = -1
    id_max = np.argmax(q)
    row = int(id_max/48)
    col = id_max%48

    return row, col

def find_param_with_lowest_Hoyer(H, obs, data, rs, tm, RI, LRES, WRES, wshift):
    #projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    projection, grid = get_projection(data, rs, tm, RI, LRES, WRES, wshift)
    xi, eta = grid.projection.geo2cube(obs['lon'], obs['lat'])

    xi_max = np.max(xi)
    xi_min = np.min(xi)
    eta_max = np.max(eta)
    eta_min = np.min(eta)

    deta = eta_max - eta_min
    eta_mid = eta_min + (deta)/2
    eta_window = deta/3

    f = (grid.xi.flatten() > xi_min) & (grid.xi.flatten() < xi_max) & (grid.eta.flatten() > eta_mid-eta_window/2) & (grid.eta.flatten() < eta_mid+eta_window/2)

    H_0 = copy.deepcopy(H[0, 0, :, :].flatten())
    H_0[~f] = np.max(H_0)+1
    H_0 = H_0.reshape(grid.shape)

    return int(np.argmin(H_0)/grid.shape[1]), np.argmin(H_0)%grid.shape[1]


def find_param_with_lowest_Hoyer_list(H, obs, data, rs, tm, RI, LRES, WRES, wshift):
    projection, grid = get_projection(data, 600, 1400, rs, tm, RI, LRES, WRES, wshift)
    xi, eta = grid.projection.geo2cube(obs['lon'], obs['lat'])

    xi_max = np.max(xi)
    xi_min = np.min(xi)
    eta_max = np.max(eta)
    eta_min = np.min(eta)

    deta = eta_max - eta_min
    eta_mid = eta_min + (deta)/2
    eta_window = deta/3

    f = (grid.xi.flatten() > xi_min) & (grid.xi.flatten() < xi_max) & (grid.eta.flatten() > eta_mid-eta_window/2) & (grid.eta.flatten() < eta_mid+eta_window/2)
    
    row_list = np.zeros(100).astype(int)
    col_list = np.zeros(100).astype(int)    
    
    for i in range(100):
        
        H_i = copy.deepcopy(H[i, 0, :, :].flatten())
        H_i[~f] = np.max(H_i)+1
        H_i = H_i.reshape(grid.shape)
        
        row_list[i] = int(np.argmin(H_i)/grid.shape[1])
        col_list[i] = np.argmin(H_i)%grid.shape[1]
        
    return row_list, col_list





#%% Find optimal l2

def check_for_outliers_2(var):
    
    dvar = np.zeros(len(var))
    dvar[1:] = abs(np.diff(var))
    dvar[0] = dvar[1]
    
    mu = np.mean(dvar[1:])
    std = np.std(dvar[1:])
    
    if any(dvar > (mu+2*std)):
        G = np.ones((len(var), 2))
        G[:, 0] = np.arange(len(var))
        m = np.linalg.solve(G.T@G, G.T@var)
        pred = G@m
        res = abs(pred-var)
        mu = np.median(res)
        std = np.std(res)
        
        var[res > (mu+2*std)] = -1    
    
    return var

def check_for_outliers(var):
    
    dvar = np.zeros(len(var))
    dvar[1:] = abs(np.diff(var))
    dvar[0] = dvar[1]
    
    mu = np.mean(dvar[1:])
    std = np.std(dvar[1:])
    
    var[dvar > (mu+2*std)] = -1
    
    return var

def inter_and_extrapolate(var):
    
    y = copy.deepcopy(var)

    if y[0] == -1:
        id_nan_stop = np.argmax(y>-1)
        y[:id_nan_stop] = y[id_nan_stop]
    
    if y[-1] == -1:
        y = np.flip(y)
        id_nan_stop = np.argmax(y>-1)
        y[:id_nan_stop] = y[id_nan_stop]
        y = np.flip(y)

    x = np.arange(len(var))
    x = x[y > -1]

    y = y[y > -1]
        
    f = interp1d(x, y, 'cubic')
    
    return f(np.arange(len(var))).astype(int)

def get_l2_id(y, test=False):
    
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    
    id_start = 0
    
    curve = 'concave'
    if ddy[id_start] > 0:        
        curve = 'convex'
    
    direction = 'decreasing'
    if dy[id_start] > 0:
        direction = 'increasing'
    
    types = np.zeros(len(y))
    for i in range(len(y)):
        if id_start > i:
            curve_i = curve
            direction_i = direction
        else:
            curve_i = 'concave'  
            if ddy[i] > 0:        
                curve_i = 'convex'
                
            direction_i = 'decreasing'
            if dy[i] > 0:
                direction_i = 'increasing'
        
        if (curve_i == 'concave') and (direction_i == 'increasing'):
            types[i] = 1
        elif (curve_i == 'concave') and (direction_i == 'decreasing'):
            types[i] = 2
        elif (curve_i == 'convex') and (direction_i == 'increasing'):
            types[i] = 3
        elif (curve_i == 'convex') and (direction_i == 'decreasing'):
            types[i] = 4
        
    count = 1
    t = types[0]
    i = 0
    while count != 10:
        i += 1
        if types[i] == t:
            count += 1
        else:
            t = types[i]
            count = 1
    
    types[:i] = t
        
    if types[0] == 2:
        id_kn_stop = np.argmax(types != 2)
        if test:
            id_kn_stop = np.min([np.argmax(types == 1), np.argmax(types == 3)])
        kn = KneeLocator(range(len(y[:id_kn_stop])), y[:id_kn_stop], curve='concave', direction='decreasing')
        id_kn = kn.knee
    
    else:
        
        id_kn_start = np.argmax(types == 1)
        types[:id_kn_start] = 1
        id_kn_stop = np.argmax(types != 1)
        
        try:
            kn = KneeLocator(range(len(y[id_kn_start:id_kn_stop])), y[id_kn_start:id_kn_stop], curve='concave', direction='increasing')
            id_kn = id_kn_start + kn.knee
        except:
            #id_kn = -1
            id_kn = id_kn_stop - 1
        
    return id_kn

#%% Diagnostic plot for finding optimal l2

def diagnostic_plot(y, folder=''):
    plt.ioff()
    l2_list = 10**np.linspace(-5, 10, 200)
    for i in range(100):
        
        #fig, axs = plt.subplots(5, 1, sharex=True, sharey=False, gridspec_kw={'wspace': 0}, figsize=(30, 30))
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'wspace': 0}, figsize=(30, 30))
        
        axs[0].plot(y[i, :])

        l2_id = get_l2_id(y[i, :])
        
        axs[0].plot(l2_id, y[i, l2_id], '.', markersize=20, alpha=0.5)
        
        axs[0].set_title(i)
        axs[0].set_xticks(np.arange(0, 205, 5))
        axs[0].xaxis.tick_top()
        axs[0].grid()
        
        axs[1].plot(np.gradient(y[i, :]))
        axs[1].plot(l2_id, np.gradient(y[i, :])[l2_id], '.', markersize=20, alpha=0.5)
        axs[1].grid()
    
        axs[2].plot(np.gradient(np.gradient(y[i, :])))
        axs[2].plot(l2_id, np.gradient(np.gradient(y[i, :]))[l2_id], '.', markersize=20, alpha=0.5)
        axs[2].grid()
        
        '''
        axs[3].plot(np.gradient(y[i, :], np.gradient(l2_list))[5:])
        #axs[3].plot(l2_id, np.gradient(y[i, :])[l2_id], '.', markersize=20, alpha=0.5)
        axs[3].grid()
    
        axs[4].plot(np.gradient(np.gradient(y[i, :], np.gradient(l2_list)), np.gradient(l2_list))[5:])
        #axs[4].plot(l2_id, np.gradient(np.gradient(y[i, :]))[l2_id], '.', markersize=20, alpha=0.5)
        axs[4].grid()
        '''
        
        plt.tight_layout()
        plt.savefig('/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/{}/{}.png'.format(folder, i), bbox_inches='tight')
        plt.close('all')
    plt.ion()

#%%

def get_opt_l2(H, row_list, col_list):
    
    l1_list = np.linspace(-3, 3, 100)
    l2_list = np.linspace(-5, 10, 200)
    
    # Get selected rows and columns
    for i, (row, col) in enumerate(zip(row_list, col_list)):
        H[i, :, 0, 0] = H[i, :, row, col]

    H = H[:, :, 0, 0]
        
    # Check for negatives
    for i in range(100):
        if np.any(H[i, :] < 0):
            H[i, :] -= np.min(H[i, :])

    l2_opt_id = np.zeros((len(l1_list), 10)).astype(int)
    for i in range(len(l1_list)):
        try:
            #print(H[i, :])
            l2_opt_id[i, 0] = np.argmax(H[i, :])
            if l2_opt_id[i, 0] < 5:                
                kn, popt, pcov = if_no_max_2(H[i, :])
                
                ids = [kn.knee]
                left = kn.knee - 1
                right = kn.knee + 1
                while len(ids) < 10:
                    if H[i, left] > H[i, right]:
                        ids.append(left)
                        left -= 1
                        #print(left)
                    else:
                        ids.append(right)
                        right += 1
                        #print(right)
                
                l2_opt_id[i, :] = ids
                #print(ids)
                '''
                l2_opt_id[i, :] = np.arange(kn.knee-4, kn.knee+6)
                '''                
            else:                
                H[i, :np.argmax(H[i, :])-10] = -1
                H[i, np.argmax(H[i, :])+10:] = -1
                for j in range(10):
                    l2_opt_id[i, j] = np.argmax(H[i, :])
                    H[i, l2_opt_id[i, j]] = -1
        except:
            l2_opt_id[i, :] = -1
            print('error')

    return l2_opt_id

def make_spline_fit(xx, yy, steps=100, s=10500):
    tck = splrep(xx, yy, s=s, k=3)
    x_new = np.linspace(np.min(xx), np.max(xx), steps)
    l2_opt_fit = BSpline(*tck)(x_new)
    
    return l2_opt_fit, tck

def if_no_max_2(y):

    def step_function(x, a0, a1, a2, a3):
        return a0 / (1 + np.exp(a2*(x-a3))) + a1
    
    def dd_step_function(x, a0, a1, a2, a3):
        return (a0 * a2**2 * np.exp(a2*x) * (np.exp(a2*x) - 1)) / (np.exp(a2*x)+1)**3

    x = np.linspace(-100, 100, 200)
    
    ymax = np.max(y)
    ymin = np.min(y)
    
    bounds = ([0, 0, 1e-20, -90], [1.5*ymax, ymax, 10, 90])
    p0 = [ymax-ymin, ymin, 0.5, 0]
    
    popt, pcov = curve_fit(step_function, x, y, bounds=bounds, p0=p0, loss='huber')
    yy = step_function(x, *popt)
    
    yy_kn = yy[:100+int(popt[-1])]
    kn = KneeLocator(np.arange(len(yy_kn)), yy_kn, curve='concave', direction='decreasing')
    
    return kn, popt, pcov

def surface_plot(var, title, cmap='Spectral', tri=True):
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-5, 10, 200)
    y, x = np.meshgrid(y, x)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if tri:
        ax.plot_trisurf(x.flatten(), y.flatten(), var.flatten(), cmap=cmap)
    else:
        ax.plot_surface(x, y, var, cmap=cmap)    
    
    ax.set_xlabel('log$_{10}(\lambda_1)$')
    ax.set_ylabel('log$_{10}(\lambda_2)$')
    plt.title(title)
    ax.view_init(50, 35+180)
    plt.tight_layout()
    return ax

def do_surface_diagnostic_plot(H, l1_list, l2_list, l2_opt_fit, l2_opt, l2_opt_id, row_list, col_list, title='', cmap='Spectral', tri=True, color='tab:blue', dcolor='k'):

    l2_min = abs(np.min(l2_list))
    l2_max = np.max(l2_list)

    l2_int_id = np.round((l2_opt_fit + l2_min)/(l2_max+l2_min)*len(l2_list), 0).astype(int)

    var = np.zeros(H.shape[:2])
    var_line = np.zeros(len(l1_list))
    var_dots = np.zeros(l2_opt.shape)
    for i, (row, col) in enumerate(zip(row_list, col_list)):
        var[i, :] = copy.deepcopy(H[i, :, row, col])
        var_line[i] = var[i, l2_int_id[i]]
        var_dots[i, :] = var[i, l2_opt_id[i, :]]

    ax = surface_plot(var, title, cmap=cmap, tri=tri)
    ax.plot(l1_list, l2_opt_fit, var_line, color=color, zorder=10)
    ax.plot(np.tile(l1_list, (10, 1)).T.flatten(), l2_opt.flatten(), var_dots.flatten(),
            '.', color=dcolor, markersize=1, zorder=9)
    
    return ax

#%% Surfaces

def if_no_max(y):

    def step_function(x, a0, a1, a2, a3):
        return a0 / (1 + np.exp(a2*(x-a3))) + a1
    
    def dd_step_function(x, a0, a1, a2, a3):
        return (a0 * a2**2 * np.exp(a2*x) * (np.exp(a2*x) - 1)) / (np.exp(a2*x)+1)**3

    x = np.linspace(-100, 100, 200)
    '''
    ymax = np.max(y)
    ymin = np.min(y)

    model = lmfit.Model(step_function)
    model.set_param_hint('a0', min=ymin, max=ymax, vary=True)
    model.set_param_hint('a1', min=ymin, max=ymax, vary=True)
    model.set_param_hint('a2', min=0, max=10, vary=True)
    model.set_param_hint('a3', min=-90, max=90, vary=True)

    params = model.make_params(a0=np.max(y)-np.min(y), a1=np.min(y), a2=2, a3=0)

    
    #result = model.fit(y, x=x, params=params, method='dual_annealing')
    result = model.fit(y, x=x, params=params, method='leastsq')

    yy = step_function(x, result.best_values['a0'], result.best_values['a1'], result.best_values['a2'], result.best_values['a3'])
    '''
    ymax = np.max(y)
    ymin = np.min(y)
    
    bounds = ([0, 0, 1e-20, -90], [1.5*ymax, ymax, 10, 90])
    p0 = [ymax-ymin, ymin, 0.5, 0]
    
    '''
    sigma = np.ones(len(y))
    dy = (ymax - ymin) / 10
    sigma[(y < (ymax-dy)) & (y > (ymin+dy))] = 100
    '''
    
    #popt, pcov = curve_fit(step_function, x, y, bounds=bounds, p0=p0, sigma=sigma)
    popt, pcov = curve_fit(step_function, x, y, bounds=bounds, p0=p0, loss='huber')
    yy = step_function(x, *popt)
    
    #yy_kn = yy[:100+int(result.best_values['a3'])]
    yy_kn = yy[:100+int(popt[-1])]
    kn = KneeLocator(np.arange(len(yy_kn)), yy_kn, curve='concave', direction='decreasing')
    #return kn.knee, result.best_values
    
    return kn.knee, popt
    
    #curv = calc_curvature(np.arange(len(y)), yy)
    #return np.argmax(curv), popt

def if_no_max_2(y):

    def step_function(x, a0, a1, a2, a3):
        return a0 / (1 + np.exp(a2*(x-a3))) + a1
    
    def dd_step_function(x, a0, a1, a2, a3):
        return (a0 * a2**2 * np.exp(a2*x) * (np.exp(a2*x) - 1)) / (np.exp(a2*x)+1)**3

    x = np.linspace(-100, 100, 200)
    
    ymax = np.max(y)
    ymin = np.min(y)
    
    bounds = ([0, 0, 1e-20, -90], [1.5*ymax, ymax, 10, 90])
    p0 = [ymax-ymin, ymin, 0.5, 0]
    
    popt, pcov = curve_fit(step_function, x, y, bounds=bounds, p0=p0, loss='huber')
    yy = step_function(x, *popt)
    
    yy_kn = yy[:100+int(popt[-1])]
    kn = KneeLocator(np.arange(len(yy_kn)), yy_kn, curve='concave', direction='decreasing')
    
    return kn, popt, pcov

def surface_plot(var, title, cmap='Spectral', tri=True):
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-5, 10, 200)
    y, x = np.meshgrid(y, x)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if tri:
        ax.plot_trisurf(x.flatten(), y.flatten(), var.flatten(), cmap=cmap)
    else:
        ax.plot_surface(x, y, var, cmap=cmap)    
    
    ax.set_xlabel('log$_{10}(\lambda_1)$')
    ax.set_ylabel('log$_{10}(\lambda_2)$')
    plt.title(title)
    ax.view_init(50, 35+180)
    plt.tight_layout()
    return ax

def get_id_list(Rm, row_list, col_list, method_max=False, test=False):
    
    l1_list = np.linspace(-3, 3, 100)
    l2_list = np.linspace(-5, 10, 200)

    model_constraint = np.zeros((len(l1_list), len(l2_list)))
    for i in range(100):
        model_constraint[i, :] = Rm[i, :, row_list[i], col_list[i]]

    # Check for negatives
    for i in range(100):
        if np.any(model_constraint[i, :] < 0):
            model_constraint[i, :] -= np.min(model_constraint[i, :])

    l2_opt_id = np.zeros(len(l1_list)).astype(int)

    for i in range(len(l1_list)):
    
        try:
            if test:
                #'''                
                l2_opt_id[i] = np.argmax(model_constraint[i, :])
                #'''
                if l2_opt_id[i] < 5:
                    #kn = KneeLocator(range(model_constraint.shape[1]), model_constraint[i, :], curve='concave', direction='decreasing')
                    #l2_opt_id[i] = kn.knee
                    #l2_opt_id[i] = get_l2_id(model_constraint[i, :])
                    #l2_opt_id[i] = get_l2_id(model_constraint[i, :], test=True)
                    l2_opt_id[i], _ = if_no_max(model_constraint[i, :])
                #'''
                '''
                l2_opt_id[i], _ = if_no_max(model_constraint[i, :])
                '''
            else:            
                if method_max:
                    l2_opt_id[i] = np.argmax(model_constraint[i, :])
                else:
                    l2_opt_id[i] = get_l2_id(model_constraint[i, :])
            
        except:
            l2_opt_id[i] = 0
            print('error')
    
    #l2_opt_id = check_for_outliers_2(l2_opt_id)
    #l2_opt_id = inter_and_extrapolate(l2_opt_id)

    return l2_opt_id

def get_id_list_2(Rm, row_list, col_list):
    
    l1_list = np.linspace(-3, 3, 100)
    l2_list = np.linspace(-5, 10, 200)

    model_constraint = np.zeros((len(l1_list), len(l2_list)))
    for i in range(100):
        model_constraint[i, :] = Rm[i, :, row_list[i], col_list[i]]

    # Check for negatives
    for i in range(100):
        if np.any(model_constraint[i, :] < 0):
            model_constraint[i, :] -= np.min(model_constraint[i, :])

    l2_opt_id = np.zeros((len(l1_list), 10)).astype(int)

    for i in range(len(l1_list)):
    
        try:
            l2_opt_id[i, 0] = np.argmax(model_constraint[i, :])
            if l2_opt_id[i, 0] < 5:
                l2_opt_id[i, :], _, pcov = if_no_max(model_constraint[i, :])
            else:
                for j in range(10):
                    l2_opt_id[i, j] = np.argmax(model_constraint[i, :])
                    model_constraint[i, l2_opt_id[i, j]] = -1
        except:
            l2_opt_id[i, :] = 0
            print('error')
    
    return l2_opt_id

def get_surface(var, l2_id):
    q = []
    for i in range(100):    
        q.append(var[i, l2_id[i]])    
    return q

def surf_wrap(var, tit, l2_id, l2_id_max, l2_id_test, l1_list, l2_list, filename):
    
    #for i in range(100):
    #    var[i, :] = (var[i, :]-np.median(var[i, :])) / np.std(var[i, :])
    
    ax = surface_plot(var, tit)
    
    var_line = get_surface(var, l2_id)
    ax.plot(l1_list, l2_list[l2_id], var_line, color='k', zorder=10, label='Concavity')
    
    var_line = get_surface(var, l2_id_max)
    ax.plot(l1_list, l2_list[l2_id_max], var_line, color='magenta', zorder=10, label='Max')
    
    var_line = get_surface(var, l2_id_test)
    ax.plot(l1_list, l2_list[l2_id_test], var_line, color='tab:blue', zorder=10, label='Test')
    
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')
    
def id_wrapper(Rm, row_list, col_list):
    l2_id_1 = get_id_list(Rm, row_list, col_list)
    l2_id_2 = get_id_list(Rm, row_list, col_list, method_max=True)
    l2_id_3 = get_id_list(Rm, row_list, col_list, test=True)
    return l2_id_1, l2_id_2, l2_id_3

def calc_Hoyer(Rm_i, folder, postfix=''):
    base = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/{}'.format(folder)
    
    q = np.load('{}/norms{}.npy'.format(base, postfix))    
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = (np.sqrt(108*48) - (Rm[:, :, row, col] / q[:, :, i])) / (np.sqrt(108*48) - 1)
    
    return Rm

def do_surfaces(Rm_i, row_list, col_list, folder, plot=True):
    row_i = row_list[0]
    col_i = col_list[0]
    
    l1_list = np.linspace(-3, 3, 100)
    l2_list = np.linspace(-5, 10, 200)

    l2_ids = np.zeros((100, 14)).astype(int)
    l2_ids_max = np.zeros((100, 14)).astype(int)
    l2_ids_test = np.zeros((100, 14)).astype(int)
    
    base = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/APL/P1/{}'.format(folder)
    # Surface plot - Sum of AF
    Rm = copy.deepcopy(Rm_i)
    var = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    id_i = 0
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
        
    if plot:
        surf_wrap(var, 'Sum of AF', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_sum.png'.format(base))
    
    # Surface plot - Sum of AF scaled with sum of R
    var = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    sum_R = np.load('{}/mag_abs.npy'.format(base))
    var /= sum_R
    Rm = copy.deepcopy(Rm_i)
    for i in range(108):
        for j in range(48):
            Rm[:, :, i, j] /= sum_R
    id_i = 1
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Sum of AF scaled with the sum of R', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_sum_sum.png'.format(base))
    
    # Surface plot - Sum of AF scaled with norm of R
    var = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    norm_R = norm_R = np.load('{}/norm_R_all.npy'.format(base))
    var /= norm_R
    Rm = copy.deepcopy(Rm_i)
    for i in range(108):
        for j in range(48):
            Rm[:, :, i, j] /= norm_R
    id_i = 2
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Sum of AF scaled with the norm of R', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_sum_norm_R.png'.format(base))        
    
    # Surface plot - Sum of AF scaled with the maximum norm of the AFs
    var = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    norms = np.load('{}/norms.npy'.format(base))
    norm_max = np.max(norms, axis=2)
    var /= norm_max
    Rm = copy.deepcopy(Rm_i)
    for i in range(108):
        for j in range(48):
            Rm[:, :, i, j] /= norm_max
    id_i = 3
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Sum of AF scaled with the maximum norm of the AFs', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_sum_norm_max.png'.format(base))

    # Surface plot - Sum of AF scaled with the sum of norms of the AFs
    var = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    norms = np.load('{}/norms.npy'.format(base))
    sum_norm = np.sum(norms, axis=2)
    var /= sum_norm
    Rm = copy.deepcopy(Rm_i)
    for i in range(108):
        for j in range(48):
            Rm[:, :, i, j] /= sum_norm
    id_i = 4
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)

    if plot:
        surf_wrap(var, 'Sum of AF scaled with the sum of norm of the AFs', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_sum_sum_norm.png'.format(base))

    # Surface plot - Norm of AF
    var = np.load('{}/norms.npy'.format(base))
    var = var[:, :, row_i*48+col_i]
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i]
    id_i = 5
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Norm of AF', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_norm.png'.format(base))

    # Surface plot - Norm of AF scaled with sum of R
    var = np.load('{}/norms.npy'.format(base))
    var = var[:, :, row_i*48+col_i]
    sum_R = np.load('{}/mag_abs.npy'.format(base))
    var /= sum_R
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i] / sum_R
    id_i = 6
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
        
    if plot:
        surf_wrap(var, 'Norm of AF scaled with the sum of R', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_norm_sum.png'.format(base))

    # Surface plot - Norm of AF scaled with norm of R
    var = np.load('{}/norms.npy'.format(base))
    var = var[:, :, row_i*48+col_i]
    norm_R = norm_R = np.load('{}/norm_R_all.npy'.format(base))
    var /= norm_R
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i] / norm_R
    id_i = 7
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)

    if plot:
        surf_wrap(var, 'Norm of AF scaled with the norm of R', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_norm_norm_R.png'.format(base))

    # Surface plot - Norm of AF scaled with the maximum norm of the AFs
    var = np.load('{}/norms.npy'.format(base))
    var = var[:, :, row_i*48+col_i]
    norms = np.load('{}/norms.npy'.format(base))
    norm_max = np.max(norms, axis=2)
    var /= norm_max
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i] / norm_max
    id_i = 8
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)

    if plot:
        surf_wrap(var, 'Norm of AF scaled with the maximum norm of the AFs', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_norm_norm_max.png'.format(base))

    # Surface plot - Norm of AF scaled with the sum of norms of the AFs
    var = np.load('{}/norms.npy'.format(base))
    var = var[:, :, row_i*48+col_i]
    norms = np.load('{}/norms.npy'.format(base))
    sum_norm = np.sum(norms, axis=2)
    var /= sum_norm
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i] / sum_norm
    id_i = 9
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Norm of AF scaled with the sum of norm of the AFs', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_norm_sum_norm.png'.format(base))

    # Surface plot - Norm of AF scaled with 1-norm of the AF
    l1 = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    l2 = np.load('{}/norms.npy'.format(base))
    l2 = l2[:, :, row_i*48+col_i]
    var = l2 / l1
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = q[:, :, i] / Rm[:, :, row, col]
    id_i = 10
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Ratio', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_ratio.png'.format(base))
    
    # Surface plot - Hoyer
    l1 = copy.deepcopy(Rm_i[:, :, row_i, col_i])
    l2 = np.load('{}/norms.npy'.format(base))
    l2 = l2[:, :, row_i*48+col_i]
    var = (np.sqrt(108*48) - (l1 / l2)) / (np.sqrt(108*48) - 1)
    q = np.load('{}/norms.npy'.format(base))
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = (np.sqrt(108*48) - (Rm[:, :, row, col] / q[:, :, i])) / (np.sqrt(108*48) - 1)
    id_i = 11
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Hoyer', 
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_Hoyer.png'.format(base))
    
    # Surface plot - Shannon
    Hs = -np.load('{}/Hs.npy'.format(base))
    var = copy.deepcopy(Hs[:, :, row_i*48+col_i])
    
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = Hs[:, :, i]
    id_i = 12
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Shannon',
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_Shannon.png'.format(base))    
    
    # Surface plot - Gini
    S = np.load('{}/Gini.npy'.format(base))
    var = copy.deepcopy(S[:, :, row_i*48+col_i])
    
    Rm = copy.deepcopy(Rm_i)
    for i in range(108*48):
        row = int(i/48)
        col = i%48
        Rm[:, :, row, col] = S[:, :, i]
    id_i = 13
    l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i] = id_wrapper(Rm, row_list, col_list)
    
    if plot:
        surf_wrap(var, 'Gini',
                  l2_ids[:, id_i], l2_ids_max[:, id_i], l2_ids_test[:, id_i], l1_list, l2_list, 
                  '{}/surface/surf_Gini.png'.format(base))
    
    return l2_ids, l2_ids_max, l2_ids_test
