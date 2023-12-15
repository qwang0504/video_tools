from distutils.core import setup

setup(
    name='video_tools',
    python_requires='>=3.9,<3.10',
    author='Martin Privat',
    version='0.1.15',
    packages=['video_tools','video_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='simple video reader, writer, and processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "PyQt5",
        "opencv-contrib-python-rolling @ https://github.com/ElTinmar/build_opencv/raw/main/opencv_contrib_python_rolling-4.8.0.20231215-cp39-cp39-linux_x86_64.whl",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@main",
        "tqdm"
    ]
)
