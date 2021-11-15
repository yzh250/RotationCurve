#
# Standard imports.
#
import os
import codecs
#
from distutils.command.sdist import sdist as DistutilsSdist
from setuptools import setup, dist, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext


#
# Build extensions wrapper to handle numpy includes.
#
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        print(numpy.get_include())
        self.include_dirs.append(numpy.get_include())


#
# Setup function.
#
setup(
    name='2D_RC',
    description='2D rotation curve package',
    long_description=open('../../README.md').read(),
    author='Kelly Douglass, University of Rochester',
    author_email='kellyadouglass@rochester.edu',
    license='BSD 3-clause License',
    url='https://github.com/yzh250/RotationCurve',
    version='1.0.0',

    #packages=find_packages(),

    # Requirements.
    requires=['Python (>3.7.0)'],
    #install_requires=open(path_prefix + 'requirements.txt', 'r').read().split('\n'),
    zip_safe=False,
    use_2to3=False,

    # Unit tests.
    #test_suite='tests',
    #tests_require='pytest',

    # Set up cython modules.
    setup_requires=['Cython', 'numpy'],
    ext_modules = [
          Extension('galaxy_component_functions_cython',
                    ['galaxy_component_functions_cython.pyx'], 
                    library_dirs=['m']), 
          Extension('Velocity_Map_Functions_cython', 
                    ['Velocity_Map_Functions_cython.pyx'], 
                    library_dirs=['m'], 
                    include_dirs=['.'])
    ],

    cmdclass={'build_ext':build_ext}
)