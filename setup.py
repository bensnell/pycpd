from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pylgcpd',
      version='3.0.0',
      description='Pure Numpy Implementation of the Landmark-Guided Coherent Point Drift Algorithm',
      long_description=readme(),
      url='https://github.com/bensnell/pylgcpd',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering',
      ],
      keywords='image processing, point cloud, registration, mesh, surface',
      author='Ben Snell',
      author_email='bensnellstudio@gmail.com',
      license='MIT',
      packages=['pylgcpd'],
      install_requires=['numpy', 'future'],
      zip_safe=False)
