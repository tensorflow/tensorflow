/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/async/backend_async_kernel_interface.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/async/c/types.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/testing/mock_async_kernel.h"

using ::testing::_;

namespace tflite::delegates {
namespace {

TEST(BackendAsyncKernelInterfaceTest, BasicTest) {
  testing::StrictMock<async::testing::MockAsyncKernel> kernel;

  EXPECT_CALL(kernel, RegisterBuffer(_, _, _, _, _));
  EXPECT_CALL(kernel, RegisterBufferSlice(_, _, _, _));
  EXPECT_CALL(kernel, UnregisterBuffer(_, _));
  EXPECT_CALL(kernel, ReconcileRestrictions(_, _, _, _, _, _));
  EXPECT_CALL(kernel, SetAttributes(_, _, _, _));
  EXPECT_CALL(kernel, Prepare(_, _));
  EXPECT_CALL(kernel, Eval(_, _, _));
  EXPECT_CALL(kernel, Wait(_, _));
  EXPECT_CALL(kernel, Finish(_, _));

  auto* tflite_kernel = kernel.kernel();

  tflite_kernel->register_buffer(tflite_kernel, nullptr, kTfLiteIoTypeInput,
                                 nullptr, nullptr, 0);
  tflite_kernel->register_buffer_slice(tflite_kernel, nullptr, 0, nullptr, 0);
  tflite_kernel->unregister_buffer(tflite_kernel, nullptr, 0);
  tflite_kernel->reconcile_restrictions(tflite_kernel, nullptr, nullptr, 0,
                                        nullptr, nullptr, nullptr);
  tflite_kernel->set_attributes(tflite_kernel, nullptr, nullptr, 0, nullptr);
  tflite_kernel->prepare(tflite_kernel, nullptr, nullptr);
  tflite_kernel->eval(tflite_kernel, nullptr, nullptr, nullptr);
  tflite_kernel->wait(tflite_kernel, nullptr, nullptr);
  tflite_kernel->finish(tflite_kernel, nullptr, nullptr);
}

}  // namespace
}  // namespace tflite::delegates
