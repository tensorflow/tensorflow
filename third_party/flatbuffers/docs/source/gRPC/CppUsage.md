Use in C++    {#flatbuffers_grpc_guide_use_cpp}
==========

## Before you get started

Before diving into the FlatBuffers gRPC usage in C++, you should already be
familiar with the following:

- FlatBuffers as a serialization format
- [gRPC](http://www.grpc.io/docs/) usage

## Using the FlatBuffers gRPC C++ library

NOTE: The examples below are also in the `grpc/samples/greeter` directory.

We will illustrate usage with the following schema:

@include grpc/samples/greeter/greeter.fbs

When we run `flatc`, we pass in the `--grpc` option and generage an additional
`greeter.grpc.fb.h` and `greeter.grpc.fb.cc`.

Example server code looks like this:

@include grpc/samples/greeter/server.cpp

Example client code looks like this:

@include grpc/samples/greeter/client.cpp
