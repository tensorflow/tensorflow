#!/bin/bash

if [ ${TASK} == "lint" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        pylint skflow || exit -1
    fi
fi

if [ ${TASK} == "nosetests" ]; then
	nosetests --with-cov
fi
