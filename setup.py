import os

from importlib.metadata import entry_points
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
    

setup(
    name='IDILI',
    version='0.0.1',
    description='ML model training to predict DILI risk from high content imaging single cell liver organoid data',
    long_description=readme(),
    packages=['idili'],
    
)