Outline
-------

The purpose of this document is to introduce the TensorFlow framework from the
perspective of developing and training models for the IPU. To some extent,
implementing at the framework level is relatively agnostic to the underlying
hardware as it pertains to the specifics of graph definition and its various
components, (e.g. how a convolutional layer is defined), but there are critical
facets of targeting the IPU from TensorFlow that need to be understood to
use it as a training and inference engine successfully. These elements include
IPU-specific API configurations, model parallelism, error logging and report
generation, as well as strategies for dealing with out-of-memory (OOM) issues.
