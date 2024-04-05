from setuptools import find_packages, setup

setup(
    name='seqluster',  # Replace with your package name
    version='1.0.0',  # Set your initial version
    description='KMeans Clustering Algorithm, with Sequential Learning Implementation.',
    author='Jerry Inyang',
    author_email='jerprog0@gmail.com',
    packages=find_packages(),  # Finds your packages automatically
    install_requires=[  # List any external dependencies here
        'requests',
        'pandas', 
        'numpy', 
        'matplotlib', 
        'scikit-learn'
        'stumpy', 
        'sktime', 
        'tslearn', 
        'dtaidistance', 
        'dit', 
    ],
)