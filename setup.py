from setuptools import setup, find_namespace_packages
from codecs import open
from os import path

__version__ = "0.0.1"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")]

setup(
    name="xicam.tomography",
    version=__version__,
    description="",
    long_description=long_description,
    url="",
    license="BSD",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    keywords="Xi-cam, tomography",
    packages=find_namespace_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    author="Hari Krishnan",
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email="hkrishnan@lbl.gov",
    entry_points={
        "xicam.plugins.GUIPlugin": ["tomography = xicam.tomography:TomographyPlugin"],
        # "databroker.ingestors": ["application/x-hdf5 = xicam.spectral.ingestors:ingest_nxSTXM"],
        # "databroker.handlers": [
        #     "JPEG = xicam.catalog_viewer.image_handlers:JPEGHandler",
        #     "TIFF = xicam.catalog_viewer.image_handlers:TIFFHandler",
        #     "EDF = xicam.catalog_viewer.image_handlers:EDFHandler",
        # ],
    },
)
