"""Shared Earth Engine / figure helpers for Lab 2 and Exercise 6 notebooks.

Functions mirror the definitions used in Labs, so both the main lab notebook and the standalone Exercise 6 notebook can import the same implementations.
"""

from __future__ import annotations

import ee
import numpy as np
import matplotlib.patheffects as pe


# ---------------------------------------------------------------------------
# Sentinel-2 preprocessing
# ---------------------------------------------------------------------------

def mask_s2_clouds(image):
    """Mask clouds/cirrus via QA60 and scale reflectance to 0–1."""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)


def get_sentinel2_for_year(year, aoi, bands):
    """Median cloud-masked S2 SR composite for one calendar year (same prep as training)."""
    filtered_img = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(f'{year}-01-01', f'{year}-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .map(mask_s2_clouds)
    )
    # extract date from the first image in the collection
    img_date = filtered_img.first().getInfo()['properties']['system:index'][:8]
    img_date_formatted = img_date[:4] + '-' + img_date[4:6] + '-' + img_date[6:]
    # median composite
    selected_img = filtered_img.median().clip(aoi).select(bands)

    return img_date_formatted, selected_img


def get_sentinel2(start, end, aoi, bands):
    """Median cloud-masked S2 SR composite for one calendar year (same prep as training)."""
    filtered_img = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(f'{start}', f'{end}')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .map(mask_s2_clouds)
        .sort('CLOUDY_PIXEL_PERCENTAGE', True)
        .median()
        .clip(aoi)
        .select(bands)
        
    )
    # extract date from the first image in the collection
    # img_date = filtered_img.first().getInfo()['properties']['system:index'][:8]
    # img_date_formatted = img_date[:4] + '-' + img_date[4:6] + '-' + img_date[6:]
    # median composite

    return filtered_img



def add_indices(img):
    """Add NDVI, NDWI, NBR, NDMI and EVI bands to a Sentinel-2 image."""
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    evi = img.expression(
        '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
        {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}
    ).rename('EVI')
    return img.addBands([ndvi, ndwi, nbr, ndmi, evi])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def fc_to_lists(fc, classProp, predProp):
    """Export true and predicted class columns from an EE FeatureCollection."""
    values = fc.aggregate_array(classProp).getInfo()
    preds = fc.aggregate_array(predProp).getInfo()
    return values, preds


# ---------------------------------------------------------------------------
# Map / figure helpers
# ---------------------------------------------------------------------------

def get_bounds_coords(bounds):
    """Return (xmin, xmax, ymin, ymax) lon/lat from an EE Geometry bounds."""
    coords = bounds.coordinates().getInfo()[0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return min(lons), max(lons), min(lats), max(lats)


def add_scale_bar(ax, length_km=5, location=(0.06, 0.05), linewidth=3, fontsize=9):
    """Draw a scale bar on a lon/lat axis, sized at the panel's centre latitude."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dlon = length_km / (111.320 * np.cos(np.radians((y0 + y1) / 2)))
    xs = x0 + (x1 - x0) * location[0]
    ys = y0 + (y1 - y0) * location[1]
    ax.plot(
        [xs, xs + dlon], [ys, ys],
        color='black', linewidth=linewidth, solid_capstyle='butt',
        path_effects=[pe.withStroke(linewidth=linewidth + 3, foreground='white')],
    )
    ax.text(
        xs + dlon / 2, ys + (y1 - y0) * 0.012, f'{length_km} km',
        ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
    )


def add_north_arrow(ax, location=(0.92, 0.86), size=0.07, fontsize=11):
    """Draw a north arrow on a lon/lat axis (north is up in EPSG:4326)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * location[0]
    y = y0 + (y1 - y0) * location[1]
    dy = (y1 - y0) * size
    ax.annotate(
        '', xy=(x, y + dy), xytext=(x, y),
        arrowprops=dict(facecolor='black', edgecolor='white', width=4, headwidth=12, headlength=9),
    )
    ax.text(
        x, y + dy + (y1 - y0) * 0.008, 'N',
        ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
    )


# ---------------------------------------------------------------------------
# LCDB v5.0/v6.0 labels and colours (shared by Lab 2 Exercise 6)
# ---------------------------------------------------------------------------

# From lcdb-classes-at-version5.pdf (short names)
LCDB_NAMES = {
    0: 'Not land',
    1: 'Built-up Area',
    2: 'Urban Parkland/Open Space',
    5: 'Transport Infrastructure',
    6: 'Surface Mine or Dump',
    10: 'Sand or Gravel',
    12: 'Landslide',
    14: 'Permanent Snow and Ice',
    15: 'Alpine Grass/Herbfield',
    16: 'Gravel or Rock',
    20: 'Lake or Pond',
    21: 'River',
    22: 'Estuarine Open Water',
    30: 'Short-rotation Cropland',
    33: 'Orchards/Vineyards/Perennial Crops',
    40: 'High Producing Exotic Grassland',
    41: 'Low Producing Grassland',
    43: 'Tall Tussock Grassland',
    44: 'Depleted Grassland',
    45: 'Herbaceous Freshwater Vegetation',
    46: 'Herbaceous Saline Vegetation',
    47: 'Flaxland',
    50: 'Fernland',
    51: 'Gorse and/or Broom',
    52: 'Manuka and/or Kanuka',
    54: 'Broadleaved Indigenous Hardwoods',
    55: 'Sub Alpine Shrubland',
    56: 'Mixed Exotic Shrubland',
    58: 'Matagouri or Grey Scrub',
    64: 'Forest - Harvested',
    68: 'Deciduous Hardwoods',
    69: 'Indigenous Forest',
    70: 'Mangrove',
    71: 'Exotic Forest',
    80: 'Peat Shrubland (Chatham Is)',
    81: 'Dune Shrubland (Chatham Is)',
}

# Official LCDB v5.0 symbology colours (lcdb-v50-legend-symbology-lyrx.lyrx)
LCDB_PALETTE = {
    0: '000000',   # Not land
    1: '9c9c9c',   # Built-up Area
    2: '688578',   # Urban Parkland/Open Space
    5: 'a80000',   # Transport Infrastructure
    6: '704489',   # Surface Mine or Dump
    10: 'ffff73',  # Sand or Gravel
    12: 'ca7af5',  # Landslide
    14: 'dbd4ff',  # Permanent Snow and Ice
    15: 'abcd66',  # Alpine Grass/Herbfield
    16: '9cba9c',  # Gravel or Rock
    20: 'bee8ff',  # Lake or Pond
    21: 'bee8ff',  # River
    22: 'd6f5e8',  # Estuarine Open Water
    30: 'ffd37f',  # Short-rotation Cropland
    33: 'e69800',  # Orchards/Vineyards/Perennial Crops
    40: 'beff8c',  # High Producing Exotic Grassland
    41: 'a3d400',  # Low Producing Grassland
    43: 'e6e68c',  # Tall Tussock Grassland
    44: 'd2d25a',  # Depleted Grassland
    45: 'c2ffd6',  # Herbaceous Freshwater Vegetation
    46: 'def5de',  # Herbaceous Saline Vegetation
    47: '7af5ca',  # Flaxland
    50: '705c00',  # Fernland
    51: '7d690f',  # Gorse and/or Broom
    52: '8c7922',  # Manuka and/or Kanuka
    54: 'a8994f',  # Broadleaved Indigenous Hardwoods
    55: 'b8ab6a',  # Sub Alpine Shrubland
    56: 'c4bb89',  # Mixed Exotic Shrubland
    58: 'd4cdae',  # Matagouri or Grey Scrub
    64: 'a1ad61',  # Forest - Harvested
    68: '477f00',  # Deciduous Hardwoods
    69: '284600',  # Indigenous Forest
    70: '448989',  # Mangrove
    71: '38a800',  # Exotic Forest
    80: 'bfcdae',  # Peat Shrubland (Chatham Is)
    81: 'd4c27a',  # Dune Shrubland (Chatham Is)
}
