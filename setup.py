# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="volprim",
    version="1.0",
    author="Sébastien Speierer",
    packages=["volprim"],
    description="A Mitsuba extension library for ray tracing volumetric primitives",
    python_requires=">=3.8",
    install_requires=[
        'ipywidgets',
        'matplotlib',
        'numpy',
        'rich',
        'plyfile',
        'pytest'
    ],
    url='https://github.com/your-username/ray-tracing-volumetric-primitives',
)
