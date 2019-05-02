Targeting the IPU from Tensorflow
=================================

.. gcdoc::
    :verb-borders:

The purpose of this document is to introduce the Tensorflow framework from the
perspective of developing and training models for the IPU. To some extent,
implementing at the framework level is relatively agnostic to the underlying
hardware as it pertains to the specifics of graph definition and its various
components, (e.g. how a convolutional layer is defined), but there are critical
facets of targeting the IPU from Tensorflow that need to be understood to
successfully use it as an training and inference engine. These elements include
but are not limited to IPU-specific API configurations, model parallelism, error
logging and report generation, as well as strategies for dealing with
out-of-memory (OOM) issues.

.. toctree::
    :maxdepth: 2

    intro
    tutorial
    device_selection
    variables
    logging
    perf_training
    troubleshooting
    examples
    references
    api

