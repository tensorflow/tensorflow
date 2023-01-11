tensorflow-gpu has been deprecated. Please install tensorflow instead.

## Deprecation Information

tensorflow and tensorflow-gpu have been the same package since TensorFlow
2.1, released in September 2019. Although the checksums differ due to metadata,
they were built in the same way and both provide GPU support via Nvidia CUDA.
As of December 2022, tensorflow-gpu is deprecated and has been replaced with
this new, empty package that generates an error upon installation.

All existing versions of tensorflow-gpu are still available, but the
TensorFlow team has stopped releasing any new tensorflow-gpu packages, and
will not release any patches for existing tensorflow-gpu versions.

## About this package

This simple package raises a warning if setup.py is executed as part of a
package installation. This intentionally prevents users from installing
the package.

To build and upload this package's source distribution (sdist) to testpypi:

```
$ vim setup.py  # update the version number and package name
$ python3 -m pip install --user twine
$ python3 setup.py sdist
$ twine upload --repository testpypi dist/*
$ pip3 install the_name_of_your_test_package -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
```
