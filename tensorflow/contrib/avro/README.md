# Avro Record Reader
Why to use the Avro Reader?
* You want to read your data from directly from hdfs into TensorFlow w/out writing a data conversion pipeline.
* You want to read data into TensorFlow as a sparse tensor. 
* You want to load (stream) avro data into TensorFlow using queues.

## Limitations
We currently compile the third party dependency of avroc only for the linux OS. Note, that this build process uses 
cmake and not bazel.

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
* You need a python with TensorFlow installed. A clean way is to use a virtual environment 
python. For more information on how to setup TensorFlow, see https://www.tensorflow.org/install/install_linux.
* If your platform is different from linux you need to compile the shared libraries for avroc and place it under 
avro/third_party/avro_c/lib. You may also have to modify how this shared library is passed around and linked to in 
BUILD files.
* You need to install the package 'avro >= 1.7.7' in your python virtual environment that also has tensorflow installed.

## Running the test cases
You can run the test cases using the command from the main tensorflow folder

`bazel test --verbose_failures tensorflow/contrib/avro/...`

If you need to define a specifc pyhton environment use the following

`baze test tensroflow/contrib/avro/... --python_path=/your/path/here/venv/bin/python`