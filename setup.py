# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Florin Bidian.

# This source code is licensed under the MIT license found in the
# LICENSE file.



from setuptools import find_setup


setup(
    version='Prophet 1.1.4, miniprophet 0.1 ',
    name='miniprophet',
    description = 'Pure python version of Facebook Prophet time series package, without STAN dependencies',
    packages=['miniprophet'],
    license='MIT' ,
    zip_safe=False,
    url = 'https://github.com/bidianf/miniprophet.git',
    install_requires=["numpy>=1.15.4",
  "scipy>=1.5.0",
  "pandas>=1.0.4",
  "python-dateutil>=2.8.0",]
)
