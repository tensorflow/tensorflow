## SavedModel Fingerprinting

This document describes the implementation details of SavedModel fingerprinting. 
The design document (RFC) can be found [here](https://github.com/tensorflow/community/pull/415).

### Implementation

The code that implements SavedModel fingerprinting can be found in :
- `tensorflow/python/saved_model/fingerprinting.py`: Public python methods for accessing the fingerprint.
- `tensorflow/python/saved_model/pywrap_saved_model_fingerprinting.*`: Python wrappers for C++ fingerprint methods. For internal use only.
- `tensorflow/cc/saved_model/fingerprint.*`: C++ methods for creating and reading the fingerprint.
- `tensorflow/core/graph/regularization/`: Code that "regularizes" the GraphDef. See the README

Generally speaking, most of the implementation for SavedModel fingerprinting is in C++. The code in this directory is meant to make these methods accessible in Python for the purposes of creating a public API
as well as instrumenting the Python side of the code base.

### Instrumentation

The current SavedModel reading and loading APIs are instrumented such that they log
the fingerprint every time they are called. The APIs that are instrumented are:
- `tf.saved_model.save`
- `tf.saved_model.load`
- `tensorflow::LoadSavedModel`
- `tensorflow::SavedModelV2Bundle::Load`


