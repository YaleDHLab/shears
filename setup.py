from setuptools import setup

setup (
  name='picsnip',
  version='0.0.3',
  packages=['picsnip'],
  keywords = ['machine-vision', 'computer-vision', 'image-processing', 'data-collection'],
  description='Extract illustrations from book page scans',
  url='https://github.com/yaledhlab/picsnip',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'glob2>=0.6',
    'matplotlib>=3.0.3',
    'scikit-image>=0.15.0',
    'scikit-learn>=0.20.3',
    'scipy>=1.2.1',
    'six>=1.11.0',
  ],
)

