tf-nightly-gpu has been removed. Please install tf-nightly instead.
The tf-nightly package supports GPU accelerated operations via Nvidia CUDA.

## Removal Information

tf-nightly and tf-nightly-gpu have been the same package since TensorFlow
2.1, released in September 2019. Although the checksums differ due to metadata,
they were built in the same way and both provide GPU support via Nvidia CUDA.
As of January 2023, tf-nightly-gpu has been removed and has been replaced with
this new, empty package that generates an error upon installation.

The TensorFlow team has stopped releasing any new tf-nightly-gpu packages, and
tf-nightly-gpu packages may disappear at any time. Please switch to tf-nightly.

## About this package

This simple package raises a warning if setup.py is executed as part of a
package installation. This intentionally prevents users from installing
the package.

To build and upload this package's source distribution (sdist) to testpypi:

```
$ vim setup.cfg  # update the version number and package name
$ python3 -m pip install --user twine
$ python3 setup.py sdist
$ twine upload --repository testpypi dist/*
$ pip3 install the_name_of_your_test_package -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
```
