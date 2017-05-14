# TensorFlow Studio

This is a prototype of UI for developing, running, debugging and deploying TensorFlow models.

*Warning*: There is no guarantee of complyiance for this.

## How to run

Build from sources:

    bazel build //tensorflow/contrib/studio
    ./bazel-bin/tensorflow/contrib/studio/studio <path/to/your/models>

Currently there is no pre-built binary available.

## Design

The Studio will be highly extensible UI, design around Python backend and Polymer frontend.

Components:
 * core/ - core components of webserver and frontend.
 * plugins/ - set of _official_ plugins.

## Configuration

Studio is using local files for storing state and coniguration:
 * `.studio` files are used in the model folders.
 * `~/.studio.rc` in home directory is used for general configuration.

