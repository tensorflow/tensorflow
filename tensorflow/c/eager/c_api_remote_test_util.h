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
#ifndef TENSORFLOW_C_EAGER_C_API_REMOTE_TEST_UTIL_H_
#define TENSORFLOW_C_EAGER_C_API_REMOTE_TEST_UTIL_H_

// Run a function containing a MatMul op and check its output.
// If heavy_load_on_streaming_rpc is true, send some rpc requests before the one
// which creates a remote input, to simulate a scenario that the remote input
// is not ready when we start running an op or a function.
void TestRemoteExecuteSilentCopies(bool async, bool remote, bool func,
                                   bool heavy_load_on_streaming_rpc,
                                   bool remote_func_outputs = false,
                                   bool has_packed_input = false);

#endif  // TENSORFLOW_C_EAGER_C_API_REMOTE_TEST_UTIL_H_
