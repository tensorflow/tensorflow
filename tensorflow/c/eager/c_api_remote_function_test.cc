/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/c/eager/c_api_remote_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace {

void TestRemoteExecuteSilentCopiesFunc(bool async, bool remote,
                                       bool heavy_load_on_streaming_rpc,
                                       bool remote_func_outputs = false,
                                       bool has_packed_input = false) {
  return TestRemoteExecuteSilentCopies(async, remote, /*func=*/true,
                                       heavy_load_on_streaming_rpc,
                                       remote_func_outputs, has_packed_input);
}

TEST(CAPI, RemoteExecuteSilentCopiesAsyncFunc) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/true,
                                    /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesFuncRemoteOutputs) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/false, /*remote=*/true,
                                    /*heavy_load_on_streaming_rpc=*/false,
                                    /*remote_func_outputs=*/true);
}
TEST(CAPI, RemoteExecuteSilentCopiesAsyncFuncRemoteOutputs) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/true,
                                    /*heavy_load_on_streaming_rpc=*/false,
                                    /*remote_func_outputs=*/true);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsyncFunc) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/false,
                                    /*heavy_load_on_streaming_rpc=*/false);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalFuncRemoteOutputs) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/false, /*remote=*/false,
                                    /*heavy_load_on_streaming_rpc=*/false,
                                    /*remote_func_outputs=*/true);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsyncFuncRemoteOutputs) {
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/false,
                                    /*heavy_load_on_streaming_rpc=*/false,
                                    /*remote_func_outputs=*/true);
}
TEST(CAPI, RemoteExecuteSilentCopiesLocalAsyncFuncOrdering) {
  // A remote input may be not ready when we start running a function. Test that
  // the function execution should wait until the remote input is ready.
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/false,
                                    /*heavy_load_on_streaming_rpc=*/true);
}
TEST(CAPI, RemoteExecuteSilentCopiesRemoteAsyncPackedInputFuncOrdering) {
  // A remote input (packed) may be not ready when we start running a function.
  // Test that the function execution should wait until the remote input is
  // ready.
  TestRemoteExecuteSilentCopiesFunc(/*async=*/true, /*remote=*/true,
                                    /*heavy_load_on_streaming_rpc=*/true,
                                    /*remote_func_outputs*/ true,
                                    /*has_packed_input=*/true);
}

}  // namespace
