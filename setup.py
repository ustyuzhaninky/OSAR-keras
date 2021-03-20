# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for OSAR-keras.

This script will install OSAR-keras as a Python module.

See: https://github.com/ustyuzhaninky/OSAR-keras

"""

from os import path
from setuptools import find_packages
from setuptools import setup
import codecs

def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

here = path.abspath(path.dirname(__file__))

install_requires = [
                    'importlib-metadata ~= 1.0 ; python_version == "3.8.1"',
                    'tensorflow >= 2.4.1', 'gin-config >= 0.1.1', 'tf_agents[reverb]>=0.7.1',
                    'dm-reverb >= 0.2.0', 'tqdm>=4.59.0', 'atari_py>=0.2.6', 'imageio>=2.8.2',
                    'PILLOW>=7.1.2', 'pandas>=1.2.3', 'pybullet>=3.1.0'
                    ]
tests_require = ['matplotlib>=3.1.3',
                 'gym >= 0.10.5', 'graphviz >= 0.14',
                 'pydot >= 1.4.1', ]

description = (
    'OSAR: An Objective Stimuli Active Repeater')

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='OSAR',
    version=get_version("OSAR/__init__.py"),
    include_package_data=True,
    packages=find_packages(exclude=['docs']),  # Required
    # package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    tests_require=tests_require,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ustyuzhaninky/OSAR-keras',  # Optional
    author='Konstantin Ustyuzhanin',  # Optional
    classifiers=[  # Optional
        'Development Status :: 0 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    project_urls={  # Optional
        'Documentation': 'https://github.com/ustyuzhaninky/OSAR-keras',
        'Bug Reports': 'https://github.com/ustyuzhaninky/OSAR-keras/issues',
        'Source': 'https://github.com/ustyuzhaninky/OSAR-keras',
    },
    license='Apache 2.0',
    keywords='keras-tensorflow actor-critic-methods gym exploratory q-learning reinforcement-learning python machine learning'
)
