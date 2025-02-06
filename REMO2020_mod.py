# This module is for the REMO2020 paper plotting
# J-PP 2024
#
import numpy as np
import xarray as xr
import pyremo as pr
import xesmf as xe
import cmaps
import datetime
from scipy import stats
import cartopy.crs as ccrs
import matplotlib as mpl
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import cartopy.feature as cf
import regionmask
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import CRS, Transformer
import geopandas as gpd
import pyproj
############################################################################################
############################################################################################
# create variable info class for model
class modelclass:
    def __init__(self, userexp, runame, r2020file, varname=''):
        self.userexp = userexp # XXXYYY
        self.runame = runame # name of the run
        self.r2020file = r2020file # is the file produced by remo2020
        self.filenames = []
        self.vardiff = []
        self.mask = []
        self.varname = varname
############################################################################################
############################################################################################
# define routine to open remo datafiles (m-files)
def open_mfdataset(
    files,
    use_cftime=True,
    parallel=True,
    data_vars="minimal",
    chunks={"time": 1},
    coords="minimal",
    compat="override",
    drop=None,
    **kwargs
    ):
    """optimized function for opening large cf datasets.

    based on https://github.com/pydata/xarray/issues/1385#issuecomment-561920115

    """

    def drop_all_coords(ds):
        # ds = ds.drop(drop)
        return ds.reset_coords(drop=True)

    ds = xr.open_mfdataset(
        files,
        parallel=parallel,
        decode_times=False,
        combine="by_coords",
        preprocess=drop_all_coords,
        decode_cf=False,
        chunks=chunks,
        data_vars=data_vars,
        coords="minimal",
        compat="override",
        **kwargs
    )
    return xr.decode_cf(ds, use_cftime=use_cftime)
############################################################################################
############################################################################################
# define routine to open calipso files
def open_mfdataset_calipso(
    files,
    use_cftime=True,
    parallel=True,
    data_vars="minimal",
    chunks={"time": 1},
    coords="minimal",
    compat="override",
    drop=None,
    **kwargs
    ):
    """optimized function for opening large cf datasets.

    based on https://github.com/pydata/xarray/issues/1385#issuecomment-561920115

    """

    def drop_all_coords(ds):
        # ds = ds.drop(drop)
        return ds.reset_coords(drop=True)
    
    def add_time_dim(ds):
        return ds.expand_dims(time = [datetime.datetime.now()])

    ds = xr.open_mfdataset(
        files,
        parallel=parallel,
        decode_times=False,
        combine="by_coords",
        preprocess=add_time_dim,
        decode_cf=False,
        chunks=chunks,
        data_vars=data_vars,
        coords="minimal",
        compat="override",
        **kwargs
    )
    return xr.decode_cf(ds, use_cftime=use_cftime)
############################################################################################
############################################################################################
# define domain routine
def get_domain(domain):
    ds = pr.remo_domain(domain, add_vertices=True)
    return ds
############################################################################################
############################################################################################
# define E-OBS reading routine
def eobs_preci_open(eobsfile,ystart,yend,trggrid,dlim,lakemask,lakemasklim):
    # Create target grid
    target_grid = get_domain(trggrid)
    # open E-OBS data
    eobs_daydata = xr.open_dataset(
    eobsfile, chunks={"time": 1, "latitude": 201, "longitude": 464}, use_cftime=False).sel(time=slice(str(ystart), str(yend)))
    # chunk file
    eobs_daydatac = eobs_daydata.rename({'longitude': 'lon','latitude': 'lat'}).chunk(dict(lat=-1)).chunk(dict(lon=-1))
    # create maski variable that can be used to mask out other values than data after remapping (including regridder zeros outside domain)
    eobs_daydatac["maski"] = xr.where(eobs_daydatac.rr >= 0.0,500,-500)
    # regrid EOBS data (0.1 -> 0.11)
    regridder = xe.Regridder(eobs_daydatac, target_grid, "conservative_normed", ignore_degenerate=True)
    eobs_daydatarmap = regridder(eobs_daydatac)
    # mask out artifacts from remapping
    eobs_daydatarmap_masked = eobs_daydatarmap.rr.where(eobs_daydatarmap.maski > 0.0)
    # create mask to cut off months less than dlim days
    ndays_pm = xr.where(eobs_daydatarmap.maski > 0.0, 1, 0) # daily mask
    ndays_ps = ndays_pm.groupby("time.season").sum(skipna=True).compute() # monthly number of days included
    eobs_mask = xr.where(ndays_pm.resample(time='1MS').sum() >= dlim, 1, 0).rename("mask").compute() # at least dlim days per gridbox per month
    # calculate monthly values
    eobs_mdata = (eobs_daydatarmap_masked.resample(time='1MS').sum(skipna=False)).compute()
    # create final mask using lake mask
    merged_mask = xr.merge([eobs_mask, lakemask.squeeze()], compat="override", join="override")
    final_mask = xr.where((merged_mask.mask > 0.5) & (merged_mask.FLFRA < lakemasklim),1,0).compute()
    # clean memory
    del target_grid, eobs_daydata, eobs_daydatac, regridder, eobs_daydatarmap, eobs_daydatarmap_masked
    del ndays_pm, eobs_mask, merged_mask
    # 
    return eobs_mdata, final_mask, ndays_ps
############################################################################################
############################################################################################ 
# define remo precipitation reading routine
def calc_remo_precipitation(ds):
    #
    import calendar
    #
    # Sum up large scale (stratiform) and convective precipitation
    preci_out = (pr.parse_dates(ds["APRL"])+pr.parse_dates(ds["APRC"])).rename("totpre") # old 142+143
    # Calculate the number of days spesific for the dates of the input data
    days_per_month = []
    for tstep in preci_out.time:
        days_per_month = np.append(days_per_month,[calendar.monthrange(int(tstep.dt.year),int(tstep.dt.month))[1]])
    # Calculate how many hours there were in month
    dayspm = xr.DataArray(days_per_month*24.,dims=["time"])
    # Set correct dates to time
    dayspm["time"] = preci_out.time
    # tranform precipitation from [mm / hours_per_month] to [mm]
    preci_out = preci_out * dayspm
    # re-use dayspm to calculate how many days per seasons
    dayspm = xr.DataArray(days_per_month,dims=["time"])
    dayspm["time"] = preci_out.time
    days_per_season = dayspm.groupby("time.season").sum(skipna=False)
    # clean memory
    del dayspm
    #
    return preci_out, days_per_season
############################################################################################
############################################################################################ 
def cut_rotated(extent,pole):
    #
    import cartopy.crs as ccrs
    #
    # Cut maximum domain from rotated data based on lon/lat box
    data_crs = ccrs.RotatedPole(*pole)
    # western central gives lon_min
    min_lon, tmp_lat = data_crs.transform_point(extent[0], (extent[2]+extent[3])/2., src_crs=ccrs.PlateCarree())
    # norther central gives lat_max
    tmp_lon, max_lat = data_crs.transform_point((extent[0]+extent[1])/2., extent[3], src_crs=ccrs.PlateCarree())
    # easter gives lon_max
    max_lon, tmp_lat = data_crs.transform_point(extent[1], (extent[2]+extent[3])/2., src_crs=ccrs.PlateCarree())
    # southern gives lat_min
    tmp_lon, min_lat = data_crs.transform_point((extent[0]+extent[1])/2., extent[2], src_crs=ccrs.PlateCarree())
    #
    return min_lon,max_lon,min_lat,max_lat
############################################################################################
############################################################################################ 
def cut_rotated_latlon(extent,pole):
    #
    import cartopy.crs as ccrs
    #
    # Cut maximum domain from rotated data based on lon/lat box
    data_crs = ccrs.RotatedPole(*pole)
     # western top gives lon_min
    min_lon, tmp_lat = data_crs.transform_point(extent[0], extent[3], src_crs=ccrs.PlateCarree())
    # norther central gives lat_max
    tmp_lon, max_lat = data_crs.transform_point((extent[0]+extent[1])/2., extent[3], src_crs=ccrs.PlateCarree())
    # easter top gives lon_max
    max_lon, tmp_lat = data_crs.transform_point(extent[1], extent[3], src_crs=ccrs.PlateCarree())
    # southern bottom gives lat_min
    tmp_lon, min_lat = min([data_crs.transform_point(extent[0], extent[2], src_crs=ccrs.PlateCarree()),data_crs.transform_point(extent[1], extent[2], src_crs=ccrs.PlateCarree())])
    #
    return min_lon,max_lon,min_lat,max_lat


############################################################################################
############################################################################################ 
# next some from Dr. L. Buntemeyer
# https://github.com/regionmask/regionmask/issues/529#issuecomment-2486162378
def create_polygon(ll_lon, ll_lat, ur_lon, ur_lat, pole):
    """create polygon and crs from domain corner points and pole"""
    # corner points of the domain
    coords = (
        (ll_lon, ll_lat),
        (ll_lon, ur_lat),
        (ur_lon, ur_lat),
        (ur_lon, ll_lat),
        (ll_lon, ll_lat),
    )
    # i go via from_cf to get the correct crs
    crs_attrs = {
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude": pole[1],
        "grid_north_pole_longitude": pole[0],
        "north_pole_grid_longitude": 0.0,
    }
    # proj4 = f"+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p={dm.pollat} +lon_0={180+dm.pollon} +datum=WGS84 +no_defs +type=crs"
    return Polygon(coords), CRS.from_cf(crs_attrs)
#
def transform_polygon(polygon, crs, segmentize=None):
    """segmentize and transform polygon to lat/lon"""
    transformer = Transformer.from_proj(crs, pyproj.Proj(init="epsg:4326"))
    if segmentize:
        polygon = polygon.segmentize(segmentize)
    return transform(transformer.transform, polygon)
############################################################################################
############################################################################################ 
def prepare_rotated_mask(extent,pole,lon,lat):
    #
    data_crs = ccrs.RotatedPole(*pole)
    # get coordinates of the extent in the target projection
    ll_lon, ur_lon, ll_lat, ur_lat = cut_rotated(extent,pole)
    # get polygon
    polygon, crs = create_polygon(ll_lon, ll_lat, ur_lon, ur_lat,pole)
    # create regionmask
    regmask = regionmask.Regions({transform_polygon(polygon, crs, segmentize=1.0)})
    # create mask in lot/lat
    maskout = regmask.mask(lon,lat)
    return maskout, ll_lon, ur_lon, ll_lat, ur_lat
############################################################################################
############################################################################################ 
def open_snowcci_swemon(snCCidata,trggrid,ystart,yend):
    # Create target grid
    target_grid = get_domain(trggrid)
    # Open SnowCCI data
    SnowCCI_mons = ["01","02","03","04","05","10","11","12"]
    SnowCCI_filenames = []
    yearlist = np.arange(ystart,yend+1)
    for year in yearlist:
        for mon in SnowCCI_mons:
            pattern = snCCidata+"SWE/"+str(year)+"/"+mon+"/*.nc"
            SnowCCI_filenames += glob.glob(pattern)
    SnowCCI_filenames.sort()
    dsSnowCCI = open_mfdataset(SnowCCI_filenames, parallel=False, chunks='auto')
    # monthly means
    dsSnowCCImon = dsSnowCCI[["swe"]].resample(time='1MS').mean()
    # prepare regridder
    regridder = xe.Regridder(dsSnowCCImon, target_grid, "bilinear")
    # do remapping
    dsSnowCCImon_trggrid = regridder(dsSnowCCImon)
    #
    return dsSnowCCImon_trggrid
############################################################################################
############################################################################################
# create domain class
class domainclass:
    def __init__(self, namein, extentin,hspace=0.2, wspace=0.275, xlocs=range(-180,180,10), ylocs=range(-90,90,10)):
        self.name = namein # short name of the domain (use also in output filename)
        self.extent = extentin # lon-lon-lat-lat boundaries
        self.hspace = hspace # the amount of height reserved for white space between subplots
        self.wspace = wspace # the amount of width reserved for blank space between subplots
        self.xlocs = xlocs # xlabels
        self.ylocs = ylocs # ylabels
############################################################################################
############################################################################################
def height_correction(height1, height2):
    """Returns height correction in [K].

    Parameters
    ------------

    height1: array like
        topography of model data [m]
    height2: array like
        topography of *true* data [m], e.g., observations

    Returns
    --------

    height correction: array like
        temperature height correction due to different topographies
    
    By Lars Buntemeyer, GERICS/HEREON
    """
    return (height2 - height1) * 0.0065
############################################################################################
############################################################################################
def prepare_hcorrection(remotopofile, eobstopofile):
    """ Prepare the height correction needed for the analysis. Returns the correction in [K]
    """
    # Load REMO and EOBS topographies and do some renaming
    remo_topo = xr.open_dataset(remotopofile).rename({"FIB": "topo"})["topo"]
    eobs_topo = xr.open_dataset(eobstopofile).rename({"elevation": "topo", 'longitude': 'lon','latitude': 'lat'})
    #
    # regrid EOBS data to remo grid (0.1 -> 0.11)
    regridder = xe.Regridder(eobs_topo, remo_topo, "bilinear")
    eobs_topo_regrid = regridder(eobs_topo.topo)
    #
    # Calculate final correction
    hcorrection = height_correction(remo_topo.where(eobs_topo_regrid > 0, drop=False),eobs_topo_regrid).compute()
    # clean memory
    del remo_topo, eobs_topo, eobs_topo_regrid
    #
    return hcorrection, regridder
############################################################################################
############################################################################################
# Seasonal mean with weights, modified from https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
def season_mean(ds, calendar="standard"):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.season") / month_length.groupby("time.season").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.season").sum(xr.ALL_DIMS), 1.0)

    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    ds_sum = (ds * wgts).groupby("time.season").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).groupby("time.season").sum(dim="time")

    # Return the weighted average
    return ds_sum / ones_out
############################################################################################
############################################################################################
def cal_mon_mean(dsin,months):
    #
    import calendar
    #
    ii = 0
    for mon in months:
        moni = list(calendar.month_name).index(mon)
        if(ii == 0):
            dsout = dsin.where(dsin['time.month'] == moni,drop=True).mean(dim="time").expand_dims({"time": [mon]})
        else:
            dsout = xr.concat([dsout,dsin.where(dsin['time.month'] == moni,drop=True).mean(dim="time").expand_dims({"time": [mon]})],dim="time", coords="all", compat="override", join="override")
        ii = ii + 1
    #
    return dsout
############################################################################################
############################################################################################
# define E-OBS reading routine
def eobs_temp_open(var,eobsfile,ystart,yend,hcorrection,regridder,dlim,lake_mask,lakemasklim):
    # open E-OBS data
    eobs_daydata = xr.open_dataset(eobsfile, chunks={"time": 1, "latitude": 201, "longitude": 464}, use_cftime=False).sel(time=slice(str(ystart), str(yend)))
    # Chunk data
    eobs_daydatac = eobs_daydata.rename({'longitude': 'lon','latitude': 'lat'}).chunk(dict(lat=-1)).chunk(dict(lon=-1))
    # regrid data
    eobs_daydata_gridded = regridder(eobs_daydatac+273.15) # C to K at the same time
    eobs_daydata_gridded = eobs_daydata_gridded.where(eobs_daydata_gridded > 0, drop=False) # get rid off 0 zeros from regridding
    #
    eobs_mdata = (eobs_daydata_gridded.resample(time='1MS').mean()[var] + hcorrection.squeeze("time").reset_coords(["time", "lon_2", "lat_2"],drop=True)).compute()
    #
    # mask months less than dlim days
    eobs_mask_tmp = xr.where(eobs_daydata_gridded[var] > 0.0, 1, 0)
    eobs_mask = xr.where(eobs_mask_tmp.resample(time='1MS').sum() >= dlim, 1, 0).rename("mask") # at least dlim days per gridbox per month
    # create final mask using lake mask
    final_mask = xr.where((eobs_mask > 0.5) & (lake_mask.squeeze() < lakemasklim),1,0).compute()
    # clean memory
    del eobs_daydata, eobs_daydatac, eobs_daydata_gridded, eobs_mask_tmp, eobs_mask
    #
    return eobs_mdata, final_mask
############################################################################################
############################################################################################
# create class for plotting
class plotclass:
    def __init__(self, label='', lspace=[], vmin=0, vmax=0, linvert=False, cmapp=''):
        self.label = label # label
        self.lspace = lspace # spacing for the colorbar
        self.vmin = vmin # colorbar minimum
        self.vmax = vmax # colorbar maximum
        self.linvert = linvert # invert colorbar
        self.cmapp = cmapp # Colormap
############################################################################################
############################################################################################
# define E-OBS reading routine
def eobs_mslp_open(eobsfile,ystart,yend,trggrid,dlim):
    # Create target grid
    target_grid = get_domain(trggrid)
    # open E-OBS data
    eobs_daydata = xr.open_dataset(eobsfile, chunks={"time": 1, "latitude": 201, "longitude": 464}, use_cftime=False).sel(time=slice(str(ystart), str(yend)))
    # Chunk data
    eobs_daydatac = eobs_daydata.rename({'longitude': 'lon','latitude': 'lat'}).chunk(dict(lat=-1)).chunk(dict(lon=-1))
    # regrid data
    regridder = xe.Regridder(eobs_daydatac, target_grid, "bilinear")
    eobs_daydata_gridded = regridder(eobs_daydatac)
    eobs_daydata_gridded = eobs_daydata_gridded.where(eobs_daydata_gridded > 0, drop=False) # get rid off 0 zeros from regridding
    #
    eobs_mdata = (eobs_daydata_gridded.resample(time='1MS').mean().pp).compute()
    #
    # mask months less than dlim days
    eobs_mask_tmp = xr.where(eobs_daydata_gridded["pp"] > 0.0, 1, 0)
    eobs_mask = xr.where(eobs_mask_tmp.resample(time='1MS').sum() >= dlim, 1, 0).rename("mask") # at least dlim days per gridbox per month
    # create final mask
    final_mask = xr.where(eobs_mask > 0.5,1,0).compute()
    # clean memory
    del target_grid, eobs_daydata, eobs_daydatac, regridder, eobs_daydata_gridded, eobs_mask_tmp, eobs_mask
    # 
    return eobs_mdata, final_mask
############################################################################################
############################################################################################
def CLARA_cfc_open_process(claradata,ystart,yend,min_lat,max_lat,min_lon,max_lon):
    # open clara cloud cover and process data
    #
    # prepareCLARA data file names
    filenamesclara = []
    yearlist = np.arange(ystart,yend+1)
    for year in yearlist:
        pattern = claradata+"/CFCmm"+str(year)+"*.nc"
        filenamesclara += glob.glob(pattern)                            
    #
    # Clara variable (will be eventually renamed to clavar)
    clavar = "cfc"
    # open data
    dsclara_global = open_mfdataset(filenamesclara, parallel=False, chunks='auto')[[clavar]]
    #
    # Cut global data and note unit change at the end (/100.0)
    dsclara = dsclara_global.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon)).rename({clavar:"clavar"})/100.0
    #
    return dsclara
############################################################################################
############################################################################################
def CLARA_albedo_open_process(claradata,months,ystart,yend,min_lat,max_lat,min_lon,max_lon):
    # open clara albedo and process data
    #
    # prepareCLARA data file names
    filenamesclara = []
    yearlist = np.arange(ystart,yend+1)
    for year in yearlist:
        pattern = claradata+"/SALmm"+str(year)+"*.nc"
        filenamesclara += glob.glob(pattern)                            
    #
    # Clara variable (will be eventually renamed to clavar)
    clavar = "black_sky_albedo_all_mean"
    # open data
    dsclara_global = open_mfdataset(filenamesclara, parallel=False, chunks='auto')[[clavar]]
    #
    # Cut global data and note unit change at the end (/100.0)
    dsclara = dsclara_global.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon)).rename({clavar:"clavar"})/100.0
    #
    # calculate monthly means
    dsclaramon = cal_mon_mean(dsclara,months).compute()
    #
    return dsclaramon
############################################################################################
############################################################################################
def main_plotter(absplotc,modplotc, da, mask, labs=False, axin=None, transform=ccrs.PlateCarree(), projection=ccrs.PlateCarree(), vmin=None, vmax=None, borders=True, 
         xlocs=range(-180,180,10), ylocs=range(-90,90,10), extent=None, figsize=(15,10), title='', add_colorbar=False, lshade=False):
    """plot a domain using the right projections and transformations with cartopy"""
    #
    ax = plt.axes(axin)
    if extent:
        ax.set_extent(extent, crs=projection)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      xlocs=xlocs, ylocs=ylocs, x_inline=False, y_inline=False)
    gl.xlines = True
    gl.ylines = True
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    gl.top_labels = False
    ####################### Define a new colormap#########################################
    norm =mpl.colors.Normalize(vmin, vmax) # determine colormap The maximum and minimum value of 
    if(labs):
        cmap=absplotc.cmapp # quote NCL Of colormap
        newcolors=cmap(absplotc.lspace)# Shard operation , Generate 0 To 1 Of 12 An array of data intervals
        if(absplotc.linvert):
            newcmap=ListedColormap(newcolors[::-1]) # Refactoring to new colormap
        else:
            newcmap=ListedColormap(newcolors[::1]) # Refactoring to new colormap 
    else:
        cmap=modplotc.cmapp # quote NCL Of colormap
        newcolors=cmap(modplotc.lspace)# Shard operation , Generate 0 To 1 Of 12 An array of data intervals
        if(modplotc.linvert):
            newcmap=ListedColormap(newcolors[::-1]) # Refactoring to new colormap
        else:
            newcmap=ListedColormap(newcolors[::1]) # Refactoring to new colormap  
    ########################################################################### 
    #plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=newcmap))
    sbl=da.plot(ax=ax, norm=norm,cmap=newcmap,  transform=transform, vmin=vmin, vmax=vmax, add_colorbar=add_colorbar)
    if(lshade):
        density=4
        ax.contourf(mask.rlon, mask.rlat, mask,
                    transform=transform,
                    colors='none',
                    hatches=[density*'/'])
    
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    #ax.lakes(resolution='50m', color='black', linewidth=1)
    if borders: ax.add_feature(cf.BORDERS)
    if borders: ax.add_feature(cf.LAKES.with_scale('50m'), facecolor='white',
               edgecolor='black', zorder=0)
    ax.set_title(title)
    #
    return sbl
############################################################################################
############################################################################################
def remo_plotter(absplotc,modplotc,domaininfo,modelruns,abs_source,abs_vals_mean,seasons,pole,lshade,figpath,figname):
    #
    # set some plotting parameters
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (20, (len(modelruns)+1)*5.5)
    plt.rcParams["figure.subplot.hspace"] = domaininfo.hspace # the amount of height reserved for white space between subplots
    plt.rcParams["figure.subplot.wspace"] = domaininfo.wspace # the amount of width reserved for blank space between subplots
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 24
    fig, axes = plt.subplots(ncols=len(seasons), nrows=len(modelruns)+1,subplot_kw={"projection": ccrs.RotatedPole(*pole)})
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=1.0, left=0, right=1.0)
    #
    # Absolute values
    for ind in range(len(seasons)):
        min_lon,max_lon,min_lat,max_lat = cut_rotated(extent=domaininfo.extent,pole=pole)
        mean_tmp = abs_vals_mean.sel(season=seasons[ind]).sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
        weight = np.cos(np.deg2rad(mean_tmp.rlat))
        meani = mean_tmp.weighted(weight).mean(dim=("rlat", "rlon"))
        title = abs_source+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)
        mask_tmp = modelruns[0].mask[ind]
        sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=True,
                    axin=axes[0,ind],transform = ccrs.RotatedPole(*pole),
                    vmin=absplotc.vmin, vmax=absplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                    lshade=lshade)
    # get final axes position
    bbox=plt.gca().get_position()

    # Draw the colorbar
    # Add a colorbar axis after first plot
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(absplotc.label, fontsize=24)
    #
    # Calculate shift due to the colorbar (make it 50% bigger than the colobar shift)
    cbbox = cbar_ax.get_position()
    cb_shift = (cbbox.y0-bbox.y0)*1.5    
    # Differences
    for mod in range(len(modelruns)):
        for ind in range(len(seasons)):
            min_lon,max_lon,min_lat,max_lat = cut_rotated(extent=domaininfo.extent,pole=pole)
            mean_tmp = modelruns[mod].vardiff.sel(season=seasons[ind]).sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            weight = np.cos(np.deg2rad(mean_tmp.rlat))
            meani = mean_tmp.weighted(weight).mean(dim=("rlat", "rlon"))
            rsme = np.sqrt(((mean_tmp**2).weighted(weight)).mean(dim=("rlat", "rlon")))
            title = modelruns[mod].runame+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)+" RSME="+"{:.2f}".format(rsme)
            if(lshade):
                mask_tmp = modelruns[mod].mask[ind].sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            else:
                mask_tmp = modelruns[mod].mask[ind]
            sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                        axin=axes[mod+1,ind],transform = ccrs.RotatedPole(*pole),
                        vmin=modplotc.vmin, vmax=modplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                        lshade=lshade)
            # move all subplots to give space for colorbar after absolute plots
            bbox=plt.gca().get_position()
            offset=cb_shift
            plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
    
    # get final axes position
    bbox=plt.gca().get_position()
    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0

    # Draw the colorbar
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(modplotc.label, fontsize=24)
    plt.savefig(figpath+figname+'.png',format="png",bbox_inches='tight', pad_inches=0)
############################################################################################
############################################################################################
def remo_plotter_mon(absplotc,modplotc,domaininfo,modelruns,abs_source,abs_vals_mean,months,pole,lshade,figpath,figname):
    #
    # set some plotting parameters
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (20, (len(modelruns)+1)*5.5)
    plt.rcParams["figure.subplot.hspace"] = domaininfo.hspace # the amount of height reserved for white space between subplots
    plt.rcParams["figure.subplot.wspace"] = domaininfo.wspace # the amount of width reserved for blank space between subplots
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 24
    fig, axes = plt.subplots(ncols=len(months), nrows=len(modelruns)+1,subplot_kw={"projection": ccrs.RotatedPole(*pole)})
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=1.0, left=0, right=1.0)
    #
    # Absolute values
    for ind in range(len(months)):
        min_lon,max_lon,min_lat,max_lat = cut_rotated(extent=domaininfo.extent,pole=pole)
        mean_tmp = abs_vals_mean.sel(time=months[ind]).sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
        weight = np.cos(np.deg2rad(mean_tmp.rlat))
        meani = mean_tmp.weighted(weight).mean(dim=("rlat", "rlon"))
        title = abs_source+" "+months[ind]+"\n Mean="+"{:.2f}".format(meani)
        mask_tmp = modelruns[0].mask[ind]
        sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=True,
                    axin=axes[0,ind],transform = ccrs.RotatedPole(*pole),
                    vmin=absplotc.vmin, vmax=absplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                    lshade=lshade)
    # get final axes position
    bbox=plt.gca().get_position()

    # Draw the colorbar
    # Add a colorbar axis after first plot
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(absplotc.label, fontsize=24)
    #
    # Calculate shift due to the colorbar (make it 50% bigger than the colobar shift)
    cbbox = cbar_ax.get_position()
    cb_shift = (cbbox.y0-bbox.y0)*1.5    
    # Differences
    for mod in range(len(modelruns)):
        for ind in range(len(months)):
            min_lon,max_lon,min_lat,max_lat = cut_rotated(extent=domaininfo.extent,pole=pole)
            mean_tmp = modelruns[mod].vardiff.sel(time=months[ind]).sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            weight = np.cos(np.deg2rad(mean_tmp.rlat))
            meani = mean_tmp.weighted(weight).mean(dim=("rlat", "rlon"))
            rsme = np.sqrt(((mean_tmp**2).weighted(weight)).mean(dim=("rlat", "rlon")))
            title = modelruns[mod].runame+" "+months[ind]+"\n Mean="+"{:.2f}".format(meani)+" RSME="+"{:.2f}".format(rsme)
            if(lshade):
                mask_tmp = modelruns[mod].mask[ind].sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            else:
                mask_tmp = modelruns[mod].mask[ind]
            sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                        axin=axes[mod+1,ind],transform = ccrs.RotatedPole(*pole),
                        vmin=modplotc.vmin, vmax=modplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                        lshade=lshade)
            # move all subplots to give space for colorbar after absolute plots
            bbox=plt.gca().get_position()
            offset=cb_shift
            plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
    
    # get final axes position
    bbox=plt.gca().get_position()
    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0

    # Draw the colorbar
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(modplotc.label, fontsize=24)
    plt.savefig(figpath+figname+'.png',format="png",bbox_inches='tight', pad_inches=0)
############################################################################################
############################################################################################
def clara_plotter_mon(absplotc,modplotc,domaininfo,pole,modelruns,abs_source,abs_vals_mean,months,lshade,figpath,figname):
    #
    # set some plotting parameters
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (20, (len(modelruns)+1)*5.5)
    plt.rcParams["figure.subplot.hspace"] = domaininfo.hspace # the amount of height reserved for white space between subplots
    plt.rcParams["figure.subplot.wspace"] = domaininfo.wspace # the amount of width reserved for blank space between subplots
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 24

    main_proj = ccrs.RotatedPole(*pole)
    fig, axes = plt.subplots(ncols=len(months), nrows=len(modelruns)+1,subplot_kw={"projection": main_proj})
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=1.0, left=0, right=1.0)
    #
    # Absolute values
    for ind in range(len(months)):
        mask_data, ll_lon, ur_lon, ll_lat, ur_lat = prepare_rotated_mask(domaininfo.extent,pole,abs_vals_mean.lon,abs_vals_mean.lat)
        mean_tmp = abs_vals_mean.sel(time=months[ind]).where(mask_data==0.0,drop=True)
        weight = np.cos(np.deg2rad(mean_tmp.lat))
        meani = mean_tmp.weighted(weight).mean(dim=("lat", "lon"))
        title = abs_source+" "+months[ind]+"\n Mean="+"{:.2f}".format(meani)
        mask_tmp = modelruns[0].mask[ind]
        sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=True,
                    axin=axes[0,ind],projection=main_proj,
                    vmin=absplotc.vmin, vmax=absplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                    extent=[ll_lon, ur_lon, ll_lat, ur_lat],lshade=lshade)
    # get final axes position
    bbox=plt.gca().get_position()
    # Draw the colorbar
    # Add a colorbar axis after first plot
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(absplotc.label, fontsize=24)
    #
    # Calculate shift due to the colorbar (make it 50% bigger than the colobar shift)
    cbbox = cbar_ax.get_position()
    cb_shift = (cbbox.y0-bbox.y0)*1.5
    #min_lon,max_lon,min_lat,max_lat = cut_rotated_latlon(extent=[domaininfo.extent[0],domaininfo.extent[1],domaininfo.extent[2],domaininfo.extent[3]],pole=pole)
    # Differences
    for mod in range(len(modelruns)):
        for ind in range(len(months)):
            mask_data, ll_lon, ur_lon, ll_lat, ur_lat = prepare_rotated_mask(domaininfo.extent,pole,modelruns[mod].vardiff.lon,modelruns[mod].vardiff.lat)
            mean_tmp = modelruns[mod].vardiff.sel(time=months[ind]).where(mask_data==0.0,drop=True)
            weight = np.cos(np.deg2rad(mean_tmp.lat))
            meani = mean_tmp.weighted(weight).mean(dim=("lat", "lon"))
            rsme = np.sqrt(((mean_tmp**2).weighted(weight)).mean(dim=("lat", "lon")))
            title = modelruns[mod].runame+" "+months[ind]+"\n Mean="+"{:.2f}".format(meani)+" RSME="+"{:.2f}".format(rsme)
            if(lshade):
                mask_tmp = modelruns[mod].mask[ind].sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            else:
                mask_tmp = modelruns[mod].mask[ind]
            sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                        axin=axes[mod+1,ind],projection=main_proj,
                        vmin=modplotc.vmin, vmax=modplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                        extent=[ll_lon, ur_lon, ll_lat, ur_lat],lshade=lshade)
            # move all subplots to give space for colorbar after absolute plots
            bbox=plt.gca().get_position()
            offset=cb_shift
            plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
    
    # get final axes position
    bbox=plt.gca().get_position()
    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0

    # Draw the colorbar
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(modplotc.label, fontsize=24)
    plt.savefig(figpath+figname+'.png',format="png",bbox_inches='tight', pad_inches=0)
############################################################################################
############################################################################################
def clara_plotter_seas_wera5(absplotc,modplotc,domaininfo,pole,modelruns,abs_source,abs_vals_mean,era5clarad,seasons,lshade,figpath,figname):
    #
    # set some plotting parameters
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (20, (len(modelruns)+2)*5.5)
    plt.rcParams["figure.subplot.hspace"] = domaininfo.hspace # the amount of height reserved for white space between subplots
    plt.rcParams["figure.subplot.wspace"] = domaininfo.wspace # the amount of width reserved for blank space between subplots
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 24
    main_proj = ccrs.RotatedPole(*pole)
    fig, axes = plt.subplots(ncols=len(seasons), nrows=len(modelruns)+2,subplot_kw={"projection": main_proj})
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.2, top=1.0, left=0, right=1.0, wspace = domaininfo.wspace, hspace = domaininfo.hspace)
    #                                                          
    # Absolute values
    for ind in range(len(seasons)):
        mask_data, ll_lon, ur_lon, ll_lat, ur_lat = prepare_rotated_mask(domaininfo.extent,pole,abs_vals_mean.lon,abs_vals_mean.lat)
        mean_tmp = abs_vals_mean.sel(season=seasons[ind]).where(mask_data==0.0,drop=True)
        weight = np.cos(np.deg2rad(mean_tmp.lat))
        meani = mean_tmp.weighted(weight).mean(dim=("lat", "lon"))
        title = abs_source+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)
        mask_tmp = modelruns[0].mask[ind]
        sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=True,
                    axin=axes[0,ind],projection=main_proj,
                    vmin=absplotc.vmin, vmax=absplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                    extent=[ll_lon, ur_lon, ll_lat, ur_lat],lshade=lshade)
    # get final axes position
    bbox=plt.gca().get_position() 
    # Draw the colorbar
    # Add a colorbar axis after first plot
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(absplotc.label, fontsize=24)
    #
    # Calculate shift due to the colorbar (make it 20% bigger than the colobar shift)
    cbbox = cbar_ax.get_position()
    cb_shift = (cbbox.y0-bbox.y0)*1.5    
    # Differences
    # First ERA
    for ind in range(len(seasons)):
        mask_data, ll_lon, ur_lon, ll_lat, ur_lat = prepare_rotated_mask(domaininfo.extent,pole,era5clarad.lon,era5clarad.lat)
        mean_tmp = era5clarad.sel(season=seasons[ind]).where(mask_data==0.0,drop=True)
        weight = np.cos(np.deg2rad(mean_tmp.lat))
        meani = mean_tmp.weighted(weight).mean(dim=("lat", "lon"))
        rsme = np.sqrt(((mean_tmp**2).weighted(weight)).mean(dim=("lat", "lon")))
        title = 'ERA5 '+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)+" RSME="+"{:.2f}".format(rsme)
        mask_tmp = modelruns[0].mask[ind]
        sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                    axin=axes[1,ind],projection=main_proj,
                    vmin=modplotc.vmin, vmax=modplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                    extent=[ll_lon, ur_lon, ll_lat, ur_lat],lshade=lshade)
        # move all subplots to give space for colorbar after absolute plots
        bbox=plt.gca().get_position()
        offset=cb_shift
        plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
    # REMO data    
    for mod in range(len(modelruns)):
        for ind in range(len(seasons)):
            mask_data, ll_lon, ur_lon, ll_lat, ur_lat = prepare_rotated_mask(domaininfo.extent,pole,modelruns[mod].vardiff.lon,modelruns[mod].vardiff.lat)
            mean_tmp = modelruns[mod].vardiff.sel(season=seasons[ind]).where(mask_data==0.0,drop=True)
            weight = np.cos(np.deg2rad(mean_tmp.lat))
            meani = mean_tmp.weighted(weight).mean(dim=("lat", "lon"))
            rsme = np.sqrt(((mean_tmp**2).weighted(weight)).mean(dim=("lat", "lon")))
            title = modelruns[mod].runame+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)+" RSME="+"{:.2f}".format(rsme)
            if(lshade):
                mask_tmp = modelruns[mod].mask[ind].sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            else:
                mask_tmp = modelruns[mod].mask[ind]
            sbl=main_plotter(absplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                        axin=axes[mod+2,ind],projection=main_proj,
                        vmin=modplotc.vmin, vmax=modplotc.vmax, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                        extent=[ll_lon, ur_lon, ll_lat, ur_lat],lshade=lshade)
            # move all subplots to give space for colorbar after absolute plots
            bbox=plt.gca().get_position()
            offset=cb_shift
            plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
    
    # get final axes position
    bbox=plt.gca().get_position()
    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, bbox.y0-(bbox.y1-bbox.y0)*0.25, 0.6, 0.01]) # set the y-value to be 0.25 time of total height below the y0

    # Draw the colorbar
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(modplotc.label, fontsize=24)
    plt.savefig(figpath+figname+'.png',format="png",bbox_inches='tight', pad_inches=0)
############################################################################################
############################################################################################
def remo_AV_plotter(modplotc,domaininfo,modelruns,seasons,pole,lshade,figpath,figname):
    #
    # set some plotting parameters
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (20, (len(modelruns)-1)*5.5)
    plt.rcParams["figure.subplot.hspace"] = domaininfo.hspace # the amount of height reserved for white space between subplots
    plt.rcParams["figure.subplot.wspace"] = domaininfo.wspace # the amount of width reserved for blank space between subplots
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = 'serif'
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 24
    fig, axes = plt.subplots(ncols=4, nrows=(len(modelruns)-1),subplot_kw={"projection": ccrs.RotatedPole(*pole)})
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.15, top=1.0, left=0, right=1.0)
    #
    for mod in range(1,len(modelruns)):
        for ind in range(len(seasons)):
            min_lon,max_lon,min_lat,max_lat = cut_rotated(extent=domaininfo.extent,pole=pole)
            mean_tmp = modelruns[mod].vardiff.sel(season=seasons[ind]).sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            weight = np.cos(np.deg2rad(mean_tmp.rlat))
            meani = mean_tmp.weighted(weight).mean(dim=("rlat", "rlon"))
            title = modelruns[mod].runame+" "+seasons[ind]+"\n Mean="+"{:.2f}".format(meani)
            if(lshade):
                mask_tmp = modelruns[mod].mask[ind].sel(rlat=slice(min_lat,max_lat), rlon=slice(min_lon,max_lon))
            else:
                mask_tmp = modelruns[mod].mask[ind]
            sbl=main_plotter(modplotc,modplotc,mean_tmp,mask_tmp,labs=False,
                        axin=axes[mod-1,ind],transform = ccrs.RotatedPole(*pole),
                        vmin=-1, vmax=1, xlocs=domaininfo.xlocs, ylocs=domaininfo.ylocs, title = title, 
                        extent = domaininfo.extent,lshade=lshade)
    
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])

    # Draw the colorbar
    cbar=fig.colorbar(sbl, cax=cbar_ax,orientation='horizontal')
    cbar.set_label(modplotc.label, fontsize=24)
    plt.savefig(figpath+figname+'.png',format="png",bbox_inches='tight', pad_inches=0)
############################################################################################
############################################################################################