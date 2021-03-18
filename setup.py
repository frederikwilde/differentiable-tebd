from setuptools import setup

with open('version.txt', 'r') as f:
    version = f.read()

setup(
    name='differentiable-tebd',
    version=version,
    description='Differentiate TEBD simulations.',
    author='Frederik Wilde',
    author_email='wilde.pysics@gmail.com',
    url=None,
    license=None,
    python_requires='>=3.7',
    packages=['differentiable_tebd'],
    install_requires=[
        'jax >=0.2.9',
        'scipy'
    ],
)