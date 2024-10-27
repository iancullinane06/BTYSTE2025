from setuptools import setup, find_packages
import sys
sys.setrecursionlimit(20)
from cx_Freeze import setup, Executable

# Define the base for the executable
base = None
if sys.platform == 'win32':
    base = 'Win32GUI'  # Use this if you're making a GUI app without a console window

# Setup function for building
setup(
    name='EcoLytix-Inference',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['assets/*']},  # Include your assets folder
    install_requires=[
        'tensorflow>=2.0',
        'keras',
        'numpy',
        'rasterio',
        'geopandas',
        'fiona',
        'shapely',
        'matplotlib',
        'PyQt5',
        'cx_Freeze',
    ],
    entry_points={
        'console_scripts': [
            'ecolytix-inference = main:main',  # Ensure 'main' module and 'main' function exist
        ]
    },
    executables=[
        Executable(
            script=r'C:\Users\rough\OneDrive\Desktop\Coding\BTYSTE2025\Application\src\EcoLytix.py',
            base=base,
            target_name='EcoLytix-Inference.exe',
            icon=r'C:\Users\rough\OneDrive\Desktop\Coding\BTYSTE2025\Application\assets\EcoLytix-Icon.ico'  # Make sure this path is correct
        )
    ]
)
