from setuptools import setup, find_packages

setup(
    name='mlcomp',
    packages=find_packages(),
    version='0.1',
    description=('A Python package and Flask app for running'
                 'machine learning and data science competitions'),
    long_description='See https://github.com/raadk/mlcomp for more information',
    author='Raad Khraishi',
    author_email='donneradlerapps@gmail.com',
    license="MIT",
    url='https://github.com/raadk/mlcomp',
    download_url='https://github.com/raadk/mlcomp/archive/0.1.tar.gz',
    keywords=['machine learning', 'competitions', 'data science'],
    include_package_data=True,
    install_requires=[
        'Flask',
        'sklearn',
        'dill',
        'numpy',
        'pandas',
        'scipy',
        'requests',
    ]
)
