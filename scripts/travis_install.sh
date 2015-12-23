#!/bin/sh

# Fail on the first error
set -e

# Show every execution step
set -x


case "$TASK" in
    "lint")
        pip install pylint
    ;;

    "nosetests")
        # Create virtual env using system numpy and scipy
        deactivate || true
        virtualenv --system-site-packages testenv
        source testenv/bin/activate

        # Install dependencies
        pip install --upgrade pip
        pip install numpy
        pip install scipy
        pip install pandas
        pip install scikit-learn

        # Install TensorFlow
        case "$TRAVIS_OS_NAME" in
            "linux")
                TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl"
            ;;
            "osx")
                TENSORFLOW_PACKAGE_URL="https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py2-none-any.whl"
            ;;
        esac
        pip install "$TENSORFLOW_PACKAGE_URL"

        # Install test tools
        pip install codecov
        pip install coverage
        pip install nose

        # Install skflow
        python setup.py install
    ;;

esac
