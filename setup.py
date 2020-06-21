import setuptools

# install package with 'python3 setup.py install'

setuptools.setup(
    name="gen_src",
    version="2",
    author="Woohyun Han, Seongha Eom",
    author_email="whhan8485@kaist.ac.kr, doubleb@kaist.ac.kr",
    description="srcgen package",
    url="https://github.com/ehdkacjswo/CS453_team2_project/tree/struct-refactor/srcgen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)