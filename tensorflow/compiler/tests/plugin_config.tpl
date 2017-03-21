# Configuration for a plugin
#
# Copy this file to plugin_config.py, and insert plugin specific
# information.  The unit test framework will then include a backend
# which tests the device and types named below.
#
# The loader value should name a python module that is used to load
# the device.
#
# See //tensorflow/compiler/tests/plugin/BUILD.tpl

device = "XLA_MY_DEVICE"
types = "DT_FLOAT,DT_INT32"
loader = "tensorflow.xla.plugin.loader.module"
