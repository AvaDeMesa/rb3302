from setuptools import find_packages, setup

package_name = 'my_robot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description='Base code: move the Wave Rover a set distance and read the lidar scan',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # command name = folder.file:function
            'move_distance = my_robot_control.move_distance:main',
            'scan_check = my_robot_control.scan_check:main',
            'path_following = my_robot_control.path_following:main',
        ],
    },
)
