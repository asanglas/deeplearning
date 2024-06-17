from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="pyprogress",
        sources=["pyprogress.pyx"],
        extra_objects=["libprogress.a"],  # Specify the static library
        language="c",
        extra_compile_args=["-std=c99"],
    )
]

setup(
    name="pyprogress",
    ext_modules=cythonize(extensions),
)
