from setuptools import setup

setup(
    name='lsh_kernel',
    version='1.0',
    packages=['lsh_kernel'],
    install_requires=['triton', 'torch']
)