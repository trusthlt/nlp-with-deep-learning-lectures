from setuptools import setup

setup(
    name='nlpwdlfw',
    version='0.6',
    packages=['nlpwdlfw'],
    license='Apache License, Version 2.0',
    author='Ivan Habernal',
    description='NLPwDL Framework',
    # enable unittest discovery
    test_suite='tests',
    install_requires=[
        'notebook==7.0.6',
        'seaborn==0.13.0',
        'pandas==2.1.2',
        'torch==2.1.1',
    ],
)
