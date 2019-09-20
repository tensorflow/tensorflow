clang-format -i -style=file include/flatbuffers/* src/*.cpp tests/test.cpp samples/*.cpp grpc/src/compiler/schema_interface.h grpc/tests/*.cpp
git checkout include/flatbuffers/reflection_generated.h
