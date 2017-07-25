from setuptools import find_packages
from setuptools import setup

setup(name='trainer',
      version='0.1',
      description='EveNet Trainer for GCloud Engine',
      url='http://github.com/elggem/EveNet',
      author='Ralf Mayet',
      author_email='ralf.mayet@mindcloud.ai',
      license='Unlicense',
      install_requires=['pandas', 'numpy', 'termcolor', 'tensorflow'],
      packages=find_packages(),
      include_package_date=True)
