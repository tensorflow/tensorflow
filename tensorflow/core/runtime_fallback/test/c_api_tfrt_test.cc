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

#include <string.h>

#include <string>
#include <vector>

#include "tensorflow/c/eager/c_api.h"

// clang-format off
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace {

TFE_TensorHandle* TestMatrixTensorHandle(TFE_Context* ctx, int64_t input_size) {
  const int64_t dims[] = {input_size, input_size};
  const int num_elements = dims[0] * dims[1];
  std::vector<float> data;
  data.resize(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    data[i] = 1.0f;
  }
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

// Copied from third_party/tensorflow/c/eager/c_api_test.cc.

void BM_TF_Execute(benchmark::State& state) {
  int input_size = state.range(0);
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();

  // Disable TFRT.
  TFE_ContextOptionsSetTfrt(opts, /*use_tfrt=*/false);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string cpu_device_name;
  CHECK(GetDeviceName(ctx, &cpu_device_name, "CPU"));

  TFE_TensorHandle* m = TestMatrixTensorHandle(ctx, input_size);
  TFE_Op* matmul = TFE_NewOp(ctx, "MatMul", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  for (auto s : state) {
    TFE_OpReset(matmul, "MatMul", cpu_device_name.c_str(), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpAddInput(matmul, m, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpAddInput(matmul, m, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpSetAttrBool(matmul, "transpose_a", 0);
    TFE_OpSetAttrBool(matmul, "transpose_b", 0);
    TFE_Execute(matmul, &retvals[0], &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_TF_Execute)->Arg(2)->Arg(100)->Arg(1000);

void BM_TFRT_Execute(benchmark::State& state) {
  int input_size = state.range(0);
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();

  // Enable TFRT.
  TFE_ContextOptionsSetTfrt(opts, /*use_tfrt=*/true);

  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::string cpu_device_name;
  CHECK(GetDeviceName(ctx, &cpu_device_name, "CPU"));

  TFE_TensorHandle* m = TestMatrixTensorHandle(ctx, input_size);
  TFE_Op* matmul = TFE_NewOp(ctx, "MatMul", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;

  for (auto s : state) {
    TFE_OpReset(matmul, "MatMul", cpu_device_name.c_str(), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpAddInput(matmul, m, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpAddInput(matmul, m, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_OpSetAttrBool(matmul, "transpose_a", 0);
    TFE_OpSetAttrBool(matmul, "transpose_b", 0);
    TFE_Execute(matmul, &retvals[0], &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_TFRT_Execute)->Arg(2)->Arg(100)->Arg(1000);

// TODO(tfrt-devs): Add GPU Benchmarks.

}  // namespace
