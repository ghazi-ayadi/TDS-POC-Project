from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['niryo_one_camera'],
    package_dir={'': 'src'},
    install_requires=['catkin_pkg'],
    setup_requires=['catkin_pkg'],
)

setup(**d)
