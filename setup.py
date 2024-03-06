import setuptools

version = {}
with open('src/version.py', 'r', encoding='utf-8') as fh:
    exec(fh.read(), version)
__version__ = version['__version__']

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='PEIM',
    version=__version__,
    author='Michael L. Li',
    author_email='mli2360@mit.edu',
    maintainer='Michael L. Li',
    maintainer_email='mli2360@mit.edu',
    url='https://bazantgroup.mit.edu/',
    description='Porous Electrode Impedance Model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/mli2360/pythonCpDII',
    license='MIT',
    packages=[
        'src',
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'configparser', 
                      'networkx'],
    extras_require={'test':['pytest','coverage', 'coveralls', 'flake8'],
                    'doc':['sphinx','sphinx_rtd_theme']},
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

