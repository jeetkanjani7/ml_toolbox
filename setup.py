import setuptools
from collections import defaultdict
import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="ml_toolbox",
    version="0.0.1",
    author="Jeet Kanjani",
    description="Python package containing ml algorithms",
    packages=setuptools.find_packages(include=['ml_toolbox*']),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
