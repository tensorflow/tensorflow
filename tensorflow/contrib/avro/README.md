# Avro Record Dataset and Parser
Why use the Avro Record Dataset and Parser?
* You want to read your data from directly from Avro records into TensorFlow without converting it to TensorProtos first.
* You want to read data into TensorFlow as a sparse tensor.

## Installation

We depend on the Avro C library, which has minimal dependencies:
https://github.com/apache/avro/blob/master/lang/c/INSTALL

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

Then set the environment variable AVRO_C_HOME to "<full-path-to-your-cloned-avro-repo>/lang/c/build/avrolib/"

### Solutions to the limitations
1. Add a custom option to the configure.py that asks for the path of the avroc shared library on the system
2. Get support for cmake in bazel; which does seem to the case yet: https://github.com/bazelbuild/bazel/issues/2532.
3. Use the cmake build chain for tensorflow -- this is experimental feature.
4. Port the CMake build file to bazel.
At the moment we favor option 1.

## Code examples
The code examples reside in the folder
avro/python/examples
To run these code examples the requirements are:

* You need to install the package 'avro >= 1.7.7' in your python virtual environment that also has tensorflow installed.

## Running the test cases
You can run the test cases using the command from the main tensorflow folder

`bazel test --verbose_failures tensorflow/contrib/avro/...`

If you need to define a specifc pyhton environment use the following

`baze test tensroflow/contrib/avro/... --python_path=/your/path/here/venv/bin/python`