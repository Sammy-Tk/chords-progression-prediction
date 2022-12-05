from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='chords-prog-proj',
      version="0.0.3",
      description="Chord Progression Prediction",
      license="MIT",
      author="Sammy El Takriti",
      author_email="seltakriti@gmail.com",
      url="_",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
