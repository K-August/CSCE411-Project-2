import os
from setuptools import setup, Extension
import pybind11

# IMPORTANT: Change these paths to where Eigen and HiGHS are actually installed on your machine!
EIGEN_INCLUDE_DIR = "/usr/include/eigen3" 
HIGHS_INCLUDE_DIR = "/usr/local/include/highs"
HIGHS_LIB_DIR = "/usr/local/lib"

ext_modules = [
    Extension(
        "mheight_ext", # This will be the name of the module you import in Python
        ["mheight_ext.cpp"],
        include_dirs=[
            pybind11.get_include(),
            EIGEN_INCLUDE_DIR,
            HIGHS_INCLUDE_DIR
        ],
        library_dirs=[HIGHS_LIB_DIR],
        libraries=["highs"], # Links against libhighs.so or libhighs.a
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"] # O3 enables maximum speed optimizations
    ),
]

setup(
    name="mheight_opt",
    version="1.0",
    ext_modules=ext_modules,
)