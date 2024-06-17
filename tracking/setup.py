from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(
    name="pyprogress",
    sources=["pyprogress.pyx", "progress.c"],
    language="c",
    extra_link_args=["-L.", "-l:libprogress.a"],  # Link against the static library
    extra_compile_args=["-O3"],  # Enable optimization
)

setup(name="pyprogress", ext_modules=cythonize([examples_extension]))
