from setuptools import setup

setup (
  name='shears',
  version='0.0.5',
  packages=['shears'],
  keywords = ['machine-vision', 'computer-vision', 'image-processing', 'data-collection'],
  description='Extract illustrations from book page scans',
  url='https://github.com/yaledhlab/shears',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'glob2>=0.6',
    'matplotlib>=2.0.0',
    'scikit-image>=0.15.0',
    'scikit-learn>=0.20.3',
    'scipy>=1.1.0',
    'six>=1.11.0',
  ],
)
