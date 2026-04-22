# Hello!
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import cartopy.crs as ccrs
from pystac_client import Client
import xarray as xr
import fsspec
from pathlib import Path
import sys
import os
import pathlib

from load_earthcare import process_earthcare_patch

output_path = pathlib.Path("/home/jovyan/my-public-bucket/earthcare_patches")
output_path.mkdir(exist_ok=True)

# setup bucket access
bucket = 's3://EarthCODE/'
endpoint_url = "https://s3.waw4-1.cloudferro.com"
region_name = "eu-west-2"
prefix = 'OSCAssets/storm-data/'


lightning_clusters = 'EC_lightning_clusters.parquet'

gdf = gpd.read_parquet(
    f"{bucket}{prefix}{lightning_clusters}",
    storage_options={
        "anon": True, 
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "region_name": region_name,
        },
    },
)

def find_shifted_centre(center_lat, lat_track, std_dev, num_points, max_shift=1):
    shift_lat_val = np.clip(
        np.random.normal(scale = std_dev),
        -max_shift,
        max_shift,
    )
    shifted_center = center_lat + shift_lat_val
    idx_peak = np.argmin(abs(lat_track - shifted_center))
    half_points = int(num_points/2)
    idx_peak = np.clip(idx_peak, half_points, len(lat_track) - half_points)
    return idx_peak

for unique_id in gdf["unique_id"].unique()[129+64:]:

    row = gdf.loc[gdf["unique_id"] == unique_id].iloc[0]
    earthcare_id = row["earthcare_id"]
    orbit_number = int(row["earthcare_id"][:-1])
    frame = row["earthcare_id"][-1]
    source = row["source"]
    cluster_id = row["cluster_id"]
    peak_lat = row["peak_lat"]

    selected_file_track = f'EC_track_lightning_{source}.parquet'

    gdf_track = gpd.read_parquet(
        f"{bucket}{prefix}{selected_file_track}",
        storage_options={
            "anon": True, 
            "client_kwargs": {
                "endpoint_url": endpoint_url,
                "region_name": region_name,
            },
        },
        # optional filtering
        filters=[('earthcare_id', "==", earthcare_id)],
    )

    gdf_sort = gdf_track.sort_values("time").reset_index().copy()
    total_lightning_counts = gdf_sort[["lightning_count_2p5", "lightning_count_5"]].groupby(gdf_sort.geometry).sum()
    total_lightning_counts["time"] = gdf_sort.time.groupby(gdf_sort.geometry).first()
    total_lightning_counts = gpd.GeoDataFrame(total_lightning_counts.sort_values("time").reset_index().copy())

    nearest_point = gpd.GeoSeries.distance(total_lightning_counts, row.geometry).argmin()
    
    idx_peak = find_shifted_centre(row.geometry.y, total_lightning_counts.geometry.y, 1, 256)
    start = idx_peak-128
    end = start+256
    
    lightning_patch = total_lightning_counts.iloc[start:end]

    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    try:
        process_earthcare_patch(
            lightning_patch, 
            unique_id, 
            product_vars = dict(
                ACM_CAP_2B=[
                    "ice_water_content", 
                    "ice_mass_flux", 
                    "ice_effective_radius", 
                    "ice_median_volume_diameter", 
                    "ice_riming_factor", 
                    "rain_rate", 
                    "rain_water_content", 
                    "rain_median_volume_diameter", 
                    "liquid_water_content", 
                    "liquid_number_concentration", 
                    "liquid_effective_radius", 
                    "aerosol_number_concentration", 
                    "aerosol_mass_content", 
                ],
                CPR_CD__2A=[
                    "doppler_velocity_best_estimate", 
                    "sedimentation_velocity_best_estimate", 
                    "spectrum_width_integrated", 
                ], 
                CPR_FMR_2A=[
                    "reflectivity_no_attenuation_correction", 
                    "reflectivity_corrected", 
                    "multiple_scattering_status", 
                ], 
                CPR_TC__2A=[
                    "simplified_convective_classification",
                ]
            ),
            save_path=output_path, 
        )
    except ValueError as e:
        pass
