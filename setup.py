from setuptools import setup, find_packages

__version__ = "0.0.1"

short_desc = (
    "Cutting Plane method to solve convex NLP with affine constratins"
)

with open("README.md") as f:
    long_description = f.read()

setup(
    name='durandal',
    version=__version__,
    author='Dustin R. Kenefake',
    author_email='Dustin.Kenefake@gmail.com',
    description=short_desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD 3.0',
    url='https://github.com/DKenefake/durandal',
    extras_require={
        'test':['pytest'],
    },
    install_requires=['numpy', 'gurobipy',],
    packages=find_packages(where='src'),
    package_dir={'':'src'},
)