#!/bin/bash

if [ ${TASK} == "lint" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        pylint skflow/* --rcfile pylint.rc || exit -1
    fi
fi

if [ ${TASK} == "nosetests" ]; then
	nosetests
fi
