from distutils.core import setup

setup(
    name='ipc_tools',
    author='Martin Privat',
    version='0.1',
    packages=['ipc_tools','ipc_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='share numpy arrays between processes',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "pyzmq",
        "arrayqueues",
        "opencv-python",
        "pandas",
        "seaborn",
        "tqdm",
        "matplotlib",
        "scipy"
    ]
)