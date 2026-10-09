from setuptools import find_packages, setup

package_name = 'motion_commander'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='you@example.com',
    description='Drive the Wave Rover to a relative goal and stop for obstacles.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'goto = motion_commander.goto_node:main',
        ],
    },
)
