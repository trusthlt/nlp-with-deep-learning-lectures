from setuptools import setup

setup(
    name='nlpwdlex',
    version='0.9',
    packages=['nlpwdlex'],
    license='Apache License, Version 2.0',
    author='Ivan Habernal',
    description='NLPwDL exercises',
    # enable unittest discovery
    test_suite='tests',
    install_requires=[
        'notebook==7.0.6',
        'seaborn==0.13.0',
        'pandas==2.1.2',
        'torch==2.1.1',
        'matplotlib==3.8.2',
    ],
)
