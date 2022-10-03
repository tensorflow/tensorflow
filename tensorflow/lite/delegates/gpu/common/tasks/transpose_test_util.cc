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

#include "tensorflow/lite/delegates/gpu/common/tasks/transpose_test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/transpose.h"

namespace tflite {
namespace gpu {
namespace {
template <DataType T>
absl::Status TransposeIntTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 1, 2, 3);
  src.data = {1, 2, -3, -4, 3, 6};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 3, 2);
  ref_tensor.data = {1, -4, 2, 3, -3, 6};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 1, 3, 2));
    GPUOperation operation = CreateTranspose(op_def, attr);
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst},
        std::make_unique<GPUOperation>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status TransposeIntTest<DataType::INT32>(
    TestExecutionEnvironment* env);
template absl::Status TransposeIntTest<DataType::INT16>(
    TestExecutionEnvironment* env);
template absl::Status TransposeIntTest<DataType::INT8>(
    TestExecutionEnvironment* env);

template <DataType T>
absl::Status TransposeUintTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 1, 2, 3);
  src.data = {1, 2, 3, 4, 5, 6};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 3, 2);
  ref_tensor.data = {1, 4, 2, 5, 3, 6};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 1, 3, 2));
    GPUOperation operation = CreateTranspose(op_def, attr);
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst},
        std::make_unique<GPUOperation>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status TransposeUintTest<DataType::UINT32>(
    TestExecutionEnvironment* env);
template absl::Status TransposeUintTest<DataType::UINT16>(
    TestExecutionEnvironment* env);
template absl::Status TransposeUintTest<DataType::UINT8>(
    TestExecutionEnvironment* env);

}  // namespace

absl::Status TransposeTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 3);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f),
                     half(4.0f), half(5.0f), half(6.0f)};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTranspose(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 3, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({half(1.0f), half(4.0f), half(2.0f),
                                     half(5.0f), half(3.0f), half(6.0f)},
                                    dst_tensor.data, 0.0f));
    }
  }

  RETURN_IF_ERROR(TransposeIntTest<DataType::INT32>(env));
  RETURN_IF_ERROR(TransposeIntTest<DataType::INT16>(env));
  RETURN_IF_ERROR(TransposeIntTest<DataType::INT8>(env));
  RETURN_IF_ERROR(TransposeUintTest<DataType::UINT32>(env));
  RETURN_IF_ERROR(TransposeUintTest<DataType::UINT16>(env));
  RETURN_IF_ERROR(TransposeUintTest<DataType::UINT8>(env));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
