# Set up

# Write some text here

from setuptools import setup, find_packages

with open('requirements.txt') as req:
    REQUIRED = req.read().splitlines()

setup(
      name='hackaton_pipeline',
      version='0.0.1',
      description='A hackaton example',
      maintainer='Raphael for president',
      maintainer_email='marginal-uw@klarna.com',
      license='Klarna Internal',
      packages=find_packages(),
      install_requires=REQUIRED,
      zip_safe=False,
      python_requires='>=3.7',
      tests_require=[],
      # get version from git tags
      setup_requires=['setuptools_scm'],
      use_scm_version=False
)
