import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #this line is for macOS
import torch
import numpy as np
import xarray as xr

nlats=34 #lats
nlons=34 #lons
surface_chans=10 #sfc
pl_levels=4 #pl
env_chans = 6 #env
env_grids = 30 #station

device = torch.device("cuda")  #change to 'cpu',if you want
model = torch.jit.load("BiXiaoCastNet3DENV.ts", map_location=device)
model.to(device)
model.eval()

input_env_root = 'input_env/'
output_met_root = 'output_met/'
output_env_root = 'output_env/'
files = []

for item in os.listdir(input_env_root):
    if item[:5] == 'env_x':
        files.append(item)

target_list = []
t0_list = []
model_list = []
for file in files:
    #Read init env data
    file_date = file.split('_')[2].replace('.npy','')
    print(file_date)
    env_x_path = os.path.join(input_env_root,'env_x_%s.npy'%(file_date))
    #The related met forecasting data.
    met_path = os.path.join(output_met_root,f'infer_{file_date}.nc')
    #If the met data is ready.
    if os.path.exists(met_path):
        print('Infering:',env_x_path)
        ds = xr.open_dataset(met_path)
        lats = ds['lat'].data
        lons = ds['lon'].data
        #use smaller met area for inference
        lat_start = np.argmin(abs(lats-43.0))
        lat_end = np.argmin(abs(lats-34.75))
        lon_start = np.argmin(abs(lons-112.5))
        lon_end = np.argmin(abs(lons-120.75))
        u10 = ds['u10'].data[:,lat_start:lat_end+1,lon_start:lon_end+1]
        v10 = ds['v10'].data[:,lat_start:lat_end+1,lon_start:lon_end+1]
        t2m = ds['t2m'].data[:,lat_start:lat_end+1,lon_start:lon_end+1]
        d2m = ds['d2m'].data[:,lat_start:lat_end+1,lon_start:lon_end+1]
        mslp = ds['mslp'].data[:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_sp = ds['pl_sp'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_u = ds['pl_u'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_v = ds['pl_v'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_z = ds['pl_z'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_w = ds['pl_w'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        pl_t = ds['pl_t'].data[:,:,lat_start:lat_end+1,lon_start:lon_end+1]
        sfc_init = np.stack([u10,v10,t2m,d2m,mslp],axis=1) #sfc (13, 5, 34, 34)
        pl_init = np.stack([pl_sp,pl_u,pl_v,pl_z,pl_w,pl_t],axis=1) #pl (13, 6, 6, 34, 34)
        sfc_init = np.flip(sfc_init,axis=2)
        pl_init = np.flip(pl_init,axis=3)
        #read env data
        env_init = np.load(env_x_path)
        #All ready
        #Start the inference
        env_output_list = []
        for ii in range(12):
            #infer 12 steps for 72 hours forecasting
            print('step:',ii+1)
            sfc_mean = np.load("stats_env/sfc_avg.npy").astype(np.float32)
            sfc_std = np.load("stats_env/sfc_std.npy").astype(np.float32)
            pl_mean = np.load("stats_env/pl_avg.npy").astype(np.float32)
            pl_std = np.load("stats_env/pl_std.npy").astype(np.float32)
            env_mean = np.load("stats_env/env_avg.npy").astype(np.float32)
            env_std = np.load("stats_env/env_std.npy").astype(np.float32)
            #Met T0 add T1 and normalization
            sfc_input = np.concatenate((sfc_init[ii,:,:,:],sfc_init[ii+1,:,:,:]),axis=0)
            pl_input = np.concatenate((pl_init[ii,:,2:,:,:],pl_init[ii+1,:,2:,:,:]),axis=0) #only us [500,600,700,850,925,1000] levels
            sfc_input = (sfc_input - sfc_mean) / sfc_std
            pl_input = (pl_input - pl_mean)/pl_std
            if ii  == 0:
                env_input = (env_init - env_mean) / env_std
            #add the batch size
            sfc_input = sfc_input[np.newaxis]
            pl_input = pl_input[np.newaxis]
            env_input = env_input[np.newaxis]
            input_sfc = torch.from_numpy(sfc_input.astype(np.float32))
            input_pl = torch.from_numpy(pl_input.astype(np.float32))
            input_env = torch.from_numpy(env_input.astype(np.float32))
            with torch.no_grad():
                output_env = model(input_pl.to(device),input_sfc.to(device),input_env.to(device))
                output_env = output_env.cpu().numpy()
                env_input = output_env[0] 

                env_record = output_env[0] * env_std + env_mean
                # print(env_record)
                env_output_list.append(env_record)
        #save env forecasts
        env_output_name = "env_fc_%s.npy"%(file_date)
        env_output_path = os.path.join(output_env_root,env_output_name)
        np.save(env_output_path,np.array(env_output_list))