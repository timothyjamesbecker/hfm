#!/usr/env/bin/python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

def two_dot(version):
    v = version.split('.')
    return '.'.join(v[0:min(3,len(v))])

# require pysam is pre-installed
try:
    import pysam
except ImportError:
    raise Exception('pysam not found; please install pysam first')
from distutils.version import StrictVersion
required_pysam_version = '0.15.2'
if StrictVersion(two_dot(pysam.__version__)) < StrictVersion(required_pysam_version):
   raise Exception('pysam version == %s is required; found %s' %
                    (required_pysam_version, pysam.__version__))
# require numpy is pre-installed
try:
    import numpy
except ImportError:
    raise Exception('numpy not found; please install numpy first')
from distutils.version import StrictVersion
required_numpy_version = '1.16.0'
if StrictVersion(two_dot(numpy.__version__)) < StrictVersion(required_numpy_version):
    raise Exception('numpy version >= %s is required; found %s' %
                    (required_numpy_version, numpy.__version__))
# require h5py is pre-installed
try:
    import h5py
except ImportError:
    raise Exception('h5py not found; please install h5py first')
from distutils.version import StrictVersion
required_h5py_version = '2.9.0'
if StrictVersion(two_dot(h5py.__version__)) < StrictVersion(required_h5py_version):
    raise Exception('h5py version >= %s is required; found %s' %
                    (required_h5py_version, h5py.__version__))    
    
def get_version():
    """Extract version number from source file."""
    from ast import literal_eval
    with open('hfm/core.pyx') as f:
        for line in f:
            if line.startswith('__version__'):
                return literal_eval(line.partition('=')[2].lstrip())
    raise ValueError("__version__ not found")
cythonize('hfm/core.pyx')
extensions = [Extension('core',
                        sources=['hfm/core.pyx'],
                        libraries=['m'],
                        include_dirs=pysam.get_include() + [numpy.get_include()],
                        define_macros=pysam.get_defines(),
                        extra_compile_args=['-ffast-math'])]
setup(
    name = 'hfm',
    version=get_version(),
    author='Timothy Becker',
    author_email='timothyjamesbecker@gmail.com',
    url='https://github.com/timothyjamesbecker/hfm',
    license='GPL License',
    description='Exact Hierarchical Feature Moment Extraction for Analysis and Visualization of Omic Data',
    classifiers=['Intended Audience :: Developers',
                 'License :: GPL 3 License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Cython',
                 'Programming Language :: C',
                 'Operating System :: POSIX',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
    cmdclass={'build_ext': build_ext},
    ext_modules = extensions,
    packages=['hfm'],
    package_data = {'hfm': ['data/*.bam', 'data/*.bai']},
    scripts = ['bin/extractor.py']
)