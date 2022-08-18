/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(FallbackStateTest, CreateRendezvous) {
  // Given a FallbackState, when a function is launched by function library
  // runtime without an explicit rendezvous, it should be able to create one
  // from the rendezvous factory.
  FunctionDefLibrary flib;
  *flib.add_function() = FunctionDefHelper::Define(
      /*function_name=*/"dummy_fn",
      /*arg_def=*/{},
      /*return values=*/{},
      /*attr def=*/{},
      /*node_def=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto fallback_state,
                          tfrt_stub::FallbackState::Create({}, flib));

  const ProcessFunctionLibraryRuntime& pflr =
      fallback_state->process_function_library_runtime();
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:localhost/replica:0/task:0";
  opts.remote_execution = true;

  auto status = pflr.RunSync(opts, pflr.GetHandle("dummy_fn"), {}, nullptr);

  EXPECT_THAT(status, Not(StatusIs(error::FAILED_PRECONDITION,
                                   HasSubstr("rendezvous"))));
}

}  // namespace
}  // namespace tensorflow
