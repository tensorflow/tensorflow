GRPC implementation and test
============================

NOTE: files in `src/` are shared with the GRPC project, and maintained there
(any changes should be submitted to GRPC instead). These files are copied
from GRPC, and work with both the Protobuf and FlatBuffers code generator.

`tests/` contains a GRPC specific test, you need to have built and installed
the GRPC libraries for this to compile. This test will build using the
`FLATBUFFERS_BUILD_GRPCTEST` option to the main FlatBuffers CMake project.

## Building Flatbuffers with gRPC

### Linux

1. Download, build and install gRPC. See [instructions](https://github.com/grpc/grpc/tree/master/src/cpp).
    * Lets say your gRPC clone is at `/your/path/to/grpc_repo`.
    * Install gRPC in a custom directory by running `make install prefix=/your/path/to/grpc_repo/install`.
2. `export GRPC_INSTALL_PATH=/your/path/to/grpc_repo/install`
3. `export PROTOBUF_DOWNLOAD_PATH=/your/path/to/grpc_repo/third_party/protobuf`
4. `mkdir build ; cd build`
5. `cmake -DFLATBUFFERS_BUILD_GRPCTEST=ON -DGRPC_INSTALL_PATH=${GRPC_INSTALL_PATH} -DPROTOBUF_DOWNLOAD_PATH=${PROTOBUF_DOWNLOAD_PATH} ..`
6. `make`

## Running FlatBuffer gRPC tests

### Linux

1. `ln -s ${GRPC_INSTALL_PATH}/lib/libgrpc++_unsecure.so.6 ${GRPC_INSTALL_PATH}/lib/libgrpc++_unsecure.so.1`
2. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GRPC_INSTALL_PATH}/lib`
3. `make test ARGS=-V` 
