from distutils.core import setup

setup(
    name='ipc_tools',
    author='Martin Privat',
    version='0.1',
    packages=['video_tools','video_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='share numpy arrays between processes',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "PyQt5",
        "opencv-python",
        "qt_widgets",
        "image_tools",
        "tqdm"
    ]
)