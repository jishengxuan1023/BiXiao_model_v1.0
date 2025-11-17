import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #this line is for macOS
import torch
import numpy as np
import xarray as xr
import datetime as dt

nlats=160 #lats
nlons=300 #lons
air_chans=6 #pl levels
surface_chans=5 #sfc
device = torch.device("cuda") #change to 'cpu',if you want
model = torch.jit.load("BiXiaoCastNet3D.ts", map_location=device)
model.to(device)
model.eval()

#root for input met data.
input_met_root = 'input_met/'
output_met_root = 'output_met/'

for file in os.listdir(input_met_root):
    print(file)
    infer_time_str = file.split('_')[-1].replace('.nc','')
    #Read initial SFC and PL data.
    ds = xr.load_dataset(os.path.join(input_met_root,file))
    u10 = ds['u10'].data
    v10 = ds['v10'].data
    t2m = ds['t2m'].data
    d2m = ds['d2m'].data
    mslp = ds['mslp'].data
    pl_sp = ds['pl_sp'].data
    pl_u = ds['pl_u'].data
    pl_v = ds['pl_v'].data
    pl_z = ds['pl_z'].data
    pl_w = ds['pl_w'].data
    pl_t = ds['pl_t'].data

    #establish sfc and pl
    sfc = np.stack([u10,v10,t2m,d2m,mslp],axis=0)
    pl = np.stack([pl_sp,pl_u,pl_v,pl_z,pl_w,pl_t],axis=0)
    #flip data (model need)
    sfc = np.flip(sfc,axis=1)
    pl = np.flip(pl,axis=2)
    #normalization
    sfc_mean = np.load("stats_met/sfc_avg_new.npy").astype(np.float32)
    sfc_std = np.load("stats_met/sfc_std_new.npy").astype(np.float32)
    pl_mean = np.load("stats_met/pl_avg_new.npy").astype(np.float32)
    pl_std = np.load("stats_met/pl_std_new.npy") .astype(np.float32)
    sfc = (sfc - sfc_mean) / sfc_std
    pl = (pl - pl_mean) / pl_std
    #add batch dim
    sfc = sfc[np.newaxis]
    pl = pl[np.newaxis]
    #set up a list for later use
    output_pl_list = []
    output_sfc_list = []
    #save the first init data in the list
    output_sfc_list.append(sfc)
    output_pl_list.append(pl)
    #12 steps inference (72 hours)
    for infer_ii in range(12):
        sfc_infer = torch.from_numpy(sfc)
        pl_infer = torch.from_numpy(pl)
        with torch.no_grad():
            print('step:',infer_ii+1)
            output_pl,output_sfc = model(pl_infer.to(device),sfc_infer.to(device))
            output_pl = output_pl.cpu().numpy()
            output_sfc = output_sfc.cpu().numpy()
            output_pl_list.append(output_pl)
            output_sfc_list.append(output_sfc)
            sfc = output_sfc
            pl = output_pl

    #normalization
    for time_ii in range(len(output_sfc_list)):
        output_sfc_list[time_ii] = output_sfc_list[time_ii][0] * sfc_std + sfc_mean
        output_pl_list[time_ii] = output_pl_list[time_ii][0] * pl_std + pl_mean
    output_sfc_output = np.array(output_sfc_list)
    output_pl_output = np.array(output_pl_list)

    #Save the output as NETCDF file
    lats = ds['lat'].data
    lons = ds['lon'].data
    levels = ds['level'].data
    times = []
    for ii in range(output_sfc_output.shape[0]):
        times.append(dt.datetime.strptime(infer_time_str,'%Y%m%d%H') + dt.timedelta(hours=ii*6))
    times = np.array(times)
    output_sfc_output = np.flip(output_sfc_output,axis=2)
    output_pl_output = np.flip(output_pl_output,axis=3)   
    ds_write = xr.Dataset(
        coords=dict(
            level=(["level"], levels),
            lat=(["lat"], lats),
            lon=(["lon"], lons),
            time=(["time"], times)
        ),
        data_vars=dict(
            u10=(["time","lat", "lon"], output_sfc_output[:,0,:,:]),
            v10=(["time","lat", "lon"], output_sfc_output[:,1,:,:]),
            t2m=(["time","lat", "lon"], output_sfc_output[:,2,:,:]),
            d2m=(["time","lat", "lon"], output_sfc_output[:,3,:,:]),
            mslp=(["time","lat", "lon"], output_sfc_output[:,4,:,:]),
            pl_sp=(["time","level", "lat", "lon"], output_pl_output[:,0,:,:,:]),
            pl_u=(["time","level", "lat", "lon"], output_pl_output[:,1,:,:,:]),
            pl_v=(["time","level", "lat", "lon"], output_pl_output[:,2,:,:,:]),
            pl_z=(["time","level", "lat", "lon"], output_pl_output[:,3,:,:,:]),
            pl_w=(["time","level", "lat", "lon"], output_pl_output[:,4,:,:,:]),
            pl_t=(["time","level", "lat", "lon"], output_pl_output[:,5,:,:,:]),
        )
    )
    encoding = {
        'u10': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'v10': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        't2m': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'd2m': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'mslp': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_sp': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_u': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_v': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_z': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_w': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
        'pl_t': {'zlib': True, 'complevel': 9, 'dtype': 'float32'},
    }
    #write file
    output_filename = 'infer_%s.nc'%(infer_time_str)
    output_file_path = os.path.join(output_met_root,output_filename)

    print('Writing output...')
    ds_write.to_netcdf(path=output_file_path,mode='w',engine='netcdf4',encoding=encoding)
