from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("_dtw",
        ["_dtw.pyx"],
              extra_compile_args = ["-O3", "-ffast-math"],
            #   extra_link_args=["-fopenmp"],
              include_dirs=[numpy.get_include()])
]

setup(
  name = "_dtw",
  ext_modules=cythonize(ext_modules),
)