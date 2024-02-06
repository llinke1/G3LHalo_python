from setuptools import setup

setup(
    name='g3lhalo',
    version='0.0.1',
    description=' Code for modelling galaxy-galaxy(-galaxy) lensing with the halo model ',
    url='',
    author='Laila Linke',
    author_email='laila.linke@uibk.ac.at',
    packages=['g3lhalo'],
    install_requires=[],
    classifiers=[
        'Development Status :: 2 - PreAlpha',
        'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
)