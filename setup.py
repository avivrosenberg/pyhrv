from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
    "wfdb",
]

setup(
    name='pyhrv',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Heart rate variability analysis in python",
    author="Aviv Rosenberg",
    author_email='avivr@cs.technion.ac.il',
    url='https://github.com/avivrosenberg/pyhrv',
    packages=['pyhrv'],
    entry_points={
        'console_scripts': [
            'pyhrv=pyhrv.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='pyhrv',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ]
)
