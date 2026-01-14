from setuptools import setup

package_name = 'piling_safety_bt'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhishek',
    maintainer_email='abhishekpakki@gmail.com',
    description='Piling Safety Check and Swimming Noodle Grasping with Behavior Trees',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = piling_safety_bt.main:main',
            'collect_data = piling_safety_bt.collect_noodle_images:main',
            'train_yolo = piling_safety_bt.train_noodle_yolo:main',
        ],
    },
)