# Avro Record Dataset and Parser
Why use the Avro Record Dataset and Parser?
* You want to read Avro formatted data into TensorFlow.
* You want to read data into TensorFlow as a sparse tensor.

## Installation
We depend on the Avro C library, which has minimal dependencies:
https://github.com/apache/avro/blob/master/lang/c/INSTALL

Currently, we assume that you have this avro library installed on your system. Essentially, `libavro.so` is reachable, e.g. through the library path `LD_LIBRARY_PATH`

During the configure step in the TensorFlow set the environment variable AVRO_C_HOME to "<full-path-to-your-cloned-avro-repo>/lang/c/build/avrolib/"

### Compilation and install of the avro library
A quick way to install it is as follows:
```
git clone https://github.com/apache/avro.git
cd avro/lang/c
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=avrolib -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTHREADSAFE=true
make
make test
make install
```
Note, that `make install` by default will place header files and library files into default places if you did not specify the `MAKE_INSTALL_PREFIX` directory.


### Why did we not add the compile and install of the avro library?
Several alternatives seemed to be harder to achieve this at that time
1. Lack of support for cmake in bazel; which does seem to the case yet: https://github.com/bazelbuild/bazel/issues/2532.
2. Use the cmake build chain for tensorflow -- this is an experimental feature.
3. Port the CMake build file to bazel -- as has been done, e.g. for parquet.

## Code examples
The code examples reside in the folder `avro/python/examples`. These examples are NOT part of the published artifact. However, `avro/python/utils` that contain helper methods that are also used by these examples have been added to the published artifact for convenience to run the examples and the tests.

## Type mappings
We assume the following, enforced mapping between c/c++, avro, and numpy types

| c/c++   | avro    | numpy   |
| ------- | ------- | ------- |
| bool    | boolean | bool    |
| int     | int     | int32   |
| long    | long    | int64   |
| float   | float   | float32 |
| double  | double  | float64 |
| string  | string  | bytes   |
| uint8*  | bytes   | bytes   |

Note, when using python3 the `bytes` formatting in numpy may be astounding, however, see that
tensorflow decided not to include a separate type for string.
See  https://github.com/tensorflow/tensorflow/issues/5552.


## Running tests
To run these code examples and pass the tests:
* When using python 2.x you need to install the package 'avro >= 1.7.7' in your python virtual environment.
* When using python 3.x you need to install the package 'avro-python3 >= 1.7.7' in your python virtual envrionment.

You can run the test cases using the command from the main tensorflow folder

`bazel test --verbose_failures tensorflow/contrib/avro/...`

If you need to define a specifc pyhton environment use the following

`bazel test tensorflow/contrib/avro/... --python_path=/your/path/here/venv/bin/python`