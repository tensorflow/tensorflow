# TPU Driver API

This repository contains the TPU driver API and network (gRPC) transport
implementation for high-performance access to TPU hardware.

# Building

Bazel is used to build the driver library and tests. Remote tests will require
access to a Cloud TPU.

## Fetching Bazel

Download the latest copy of Bazel from
https://github.com/bazelbuild/bazel/releases.

## Building

`bazel build ...`

## Testing

`bazel test ...`
