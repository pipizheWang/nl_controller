from setuptools import find_packages, setup

package_name = 'nl_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhe',
    maintainer_email='zhe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'attitude_controller = nl_controller.attitude_control:main',
            'traj_controller_NL = nl_controller.traj_controller_NL:main',
            'traj_controller_NF = nl_controller.traj_controller_NF:main',
            'phi_estimate = nl_controller.Phi_estimate:main',
            'demo = nl_controller.demo:main',
            'traj_controller_NL2 = nl_controller.traj_controller_NL2:main',
            'traj_controller_NF2 = nl_controller.traj_controller_NF2:main',
            'traj_controller_NFC = nl_controller.traj_controller_NFC:main',
            'traj_controller_NFC2 = nl_controller.traj_controller_NFC2:main',
            'traj_controller_position = nl_controller.traj_controller_position:main',
            'test_node = nl_controller.test_node:main',
            'traj_controller_MPC = nl_controller.traj_controller_MPC:main',
            'backstepping_controller = nl_controller.backstepping_controller:main',
        ],
    },
)