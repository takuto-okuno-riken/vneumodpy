from setuptools import setup, find_packages

setup(
    name="vneumodpy",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "matplotlib", "scikit-learn", "numpy", "pandas", "scipy", "h5py", "hdf5storage", "nibabel", "statsmodels"
    ],
)
