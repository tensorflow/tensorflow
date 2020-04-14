# Hexagon Delegate Testing

This directory contains unit-tests for Op Builders for the hexagon delegate.
To Run the all the tests use the run_tests.sh under directory and pass
the path to the directory containing libhexagon_nn_skel*.so files.
The script will copy all files to the device and build all tests and execute
them.

The test should stop if one of the tests failed.

Example:

Follow the [Instructions](https://www.tensorflow.org/lite/performance/hexagon_delegate)
and download the hexagon_nn_skel and extract the files.
For example if files are extracted in /tmp/hexagon_skel, the sample command.

`
bash tensorflow/lite/experimental/delegates/hexagon/builders/tests/run_tests.sh /tmp/hexagon_skel
`
