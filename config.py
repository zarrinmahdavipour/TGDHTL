"""Configuration file for TransHTL framework.

This module defines dataset configurations and constant parameters used across the
hyperspectral image classification pipeline.
"""

# Dataset configurations for Indian Pines, PaviaU, KSC, and Salinas
DATASETS = {
    'IndianPines': {
        'num_classes': 16,  # Number of classes
        'bands': 200,       # Number of spectral bands
        'size': (145, 145), # Spatial dimensions (height, width)
        'class_names': [
            'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
            'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
            'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
            'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
        ],
        'false_color_bands': [50, 27, 17]  # Bands for RGB false color visualization
    },
    'PaviaU': {
        'num_classes': 9,
        'bands': 103,
        'size': (610, 340),
        'class_names': [
            'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
            'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows'
        ],
        'false_color_bands': [50, 30, 10]
    },
    'KSC': {
        'num_classes': 13,
        'bands': 176,
        'size': (512, 614),
        'class_names': [
            'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak', 'Slash pine',
            'Oak/Broadleaf', 'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
            'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'
        ],
        'false_color_bands': [50, 27, 17]
    },
    'Salinas': {
        'num_classes': 16,
        'bands': 204,
        'size': (512, 217),
        'class_names': [
            'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
            'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
            'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
            'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
            'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis'
        ],
        'false_color_bands': [50, 27, 17]
    }
}

# Constant parameters
PATCH_LENGTH = 16  # Half-size of the patch for spatial context
N_COMPONENTS = 30  # Number of PCA components for dimensionality reduction