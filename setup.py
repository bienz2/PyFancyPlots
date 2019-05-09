import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name = "pyfancyplots-bienz2",
        version = "0.1",
        author = "Amanda Bienz",
        author_email = "bienz2@illinois.edu",
        description = "A python plotting script containing Luke Olson's styling preferences",
        long_description=long_description,
        long_description_context_type="test/markdown",
        url="https://github.com/bienz2/PyFancyPlots",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)



