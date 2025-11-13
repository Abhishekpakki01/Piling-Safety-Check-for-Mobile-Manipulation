from setuptools import setup

package_name = 'spot_ws'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhishek',
    maintainer_email='abhishekpakki@gmail.com',
    description='ROS2 Behavior Tree for Piling Safety',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'piling_safety_bt = spot_ws.main:main',
        ],
    },
)