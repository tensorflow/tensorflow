#!/bin/sh

# Fail on the first error
set -e

# Show every execution step
set -x


case "$TASK" in
    "lint")
        if [ "$TRAVIS_OS_NAME" != "osx" ]; then
            pylint skflow || exit -1
        fi
    ;;

    "nosetests")
        nosetests --with-coverage --cover-erase --cover-package=skflow
        codecov
    ;;

esac
