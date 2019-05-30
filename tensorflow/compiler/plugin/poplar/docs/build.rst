
How to build TensorFlow from source
-----------------------------------
Build tool prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

    The install requires bazel, pip, curl, and various python packages. 

    - Curl::

        sudo apt install curl

    - Bazel requires Java (depending on the version of linux, you may need the Oracle Java, not the openjdk one)::

        sudo apt install openjdk-8-jdk

    - Bazel is the build tool - use version 0.21.0 or later versions::

        export BAZEL_VERSION=0.21.0
        wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
        ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

    This will install it in ``~/bin`` which you can add to your ``PATH`` if it isn't already.

    - Python::

        sudo apt install python-numpy python-dev python-pip python-wheel virtualenv

    - libc-ares-dev (maybe): If you get errors about missing ``ares.h``, try this::

        sudo apt install libc-ares-dev

    You will also need Poplar installation. Poplar SDK can be downloaded at: https://downloads.graphcore.ai/

Build instructions
~~~~~~~~~~~~~~~~~~

    Create your workspace::

        mkdir -p tf_build/install
        cd tf_build

    Git clone the repositories of interest::

        git clone https://placeholder@github.com/graphcore/tensorflow_packaging.git
        git clone https://placeholder@github.com/graphcore/tensorflow.git

    Check bazel version (make sure it is >= 0.21.0)::

        bazel version

    To build against a release, set an environment variable to point to the base of a built poplar installation::

        export TF_POPLAR_BASE=/path/to/poplar_sdk/poplar-ubuntu_18_04-x.x.x

    To set up the Python build environment and configure TensorFlow::

        source tensorflow_packaging/configure python3

    Using the pip wheel package generator as the final target, build TensorFlow::

        bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package  

    Once the build has completed, make the pip wheel using the package generator::

        ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ../install

    To run the suit of unit tests::

        bazel test --verbose_failures --test_output=all --test_env=TF_POPLAR_FLAGS="--use_ipu_model"  --verbose_failures --config=opt //tensorflow/contrib/ipu:poplar_test_suite

    Adding  ``--test_env TF_CPP_MIN_VLOG_LEVEL=1`` to the command line will dump out more debug information, including the work done by the XLA driver turning the XLA graph into a Poplar graph.
    
    To repeat a test multiple times, add ``--runs_per_test N``.
    
    To ensure a test is run, even when it ran successfully and is cached, add ``--no_cache_test_results``.