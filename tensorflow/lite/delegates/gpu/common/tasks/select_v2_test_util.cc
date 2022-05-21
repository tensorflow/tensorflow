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

#include "tensorflow/lite/delegates/gpu/common/tasks/select_v2_test_util.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/select_v2.h"

namespace tflite {
namespace gpu {

using CondTensor = Tensor<BHWC, DataType::FLOAT32>;
using TrueTensor = Tensor<BHWC, DataType::FLOAT32>;
using ElseTensor = Tensor<BHWC, DataType::FLOAT32>;

std::vector<float> SetUpData(CondTensor& cond_tensor, TrueTensor& true_tensor,
                             ElseTensor& false_tensor, int batch, int height,
                             int width, int channels, bool broadcast_true,
                             bool broadcast_false, bool gather_by_rows) {
  if (gather_by_rows) {
    cond_tensor.shape = BHWC(batch, height, width, 1);
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          cond_tensor.data.push_back(w % 2 == 0);
        }
      }
    }
  } else {
    cond_tensor.shape = BHWC(batch, height, 1, channels);
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < height; h++) {
        for (int c = 0; c < channels; c++) {
          cond_tensor.data.push_back(c % 2 == 0);
        }
      }
    }
  }
  true_tensor.shape =
      broadcast_true ? BHWC(1, 1, 1, 1) : BHWC(batch, height, width, channels);
  false_tensor.shape =
      broadcast_false ? BHWC(1, 1, 1, 1) : BHWC(batch, height, width, channels);
  constexpr float true_value = 99.f;
  constexpr float false_value = -99.f;
  if (broadcast_true) {
    true_tensor.data.push_back(true_value);
  }
  if (broadcast_false) {
    false_tensor.data.push_back(false_value);
  }
  std::vector<float> expected_data;
  expected_data.reserve(batch * height * width * channels);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          int expected_true_value = true_value;
          int expected_false_value = false_value;
          if (!broadcast_true) {
            // true are even values.
            expected_true_value = w * channels + c * 2 / 2;
            true_tensor.data.push_back(expected_true_value);
          }
          if (!broadcast_false) {
            // false are odd values.
            expected_false_value = w * channels + c * 2 / 2 + 1;
            false_tensor.data.push_back(expected_false_value);
          }
          if (gather_by_rows) {
            expected_data.push_back(
                cond_tensor.data[b * width * height + h * width + w]
                    ? expected_true_value
                    : expected_false_value);
          } else {
            expected_data.push_back(
                cond_tensor.data[b * height * channels + h * channels + c]
                    ? expected_true_value
                    : expected_false_value);
          }
        }
      }
    }
  }
  return expected_data;
}

absl::Status RunSelectV2(TestExecutionEnvironment* env,
                         const DataType& data_type,
                         const TensorStorageType& storage,
                         const CalculationsPrecision& precision,
                         const CondTensor& cond_tensor,
                         const TrueTensor& true_tensor,
                         const ElseTensor& false_tensor, int batch, int height,
                         int width, int channels, bool broadcast_true,
                         bool broadcast_false, TensorFloat32& dst_tensor) {
  OperationDef op_def;
  const auto layout = batch > 1 ? Layout::BHWC : Layout::HWC;
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.dst_tensors.push_back({data_type, storage, layout});
  op_def.precision = precision;
  TensorDescriptor cond_descriptor = op_def.src_tensors[0];
  TensorDescriptor true_descriptor = op_def.src_tensors[1];
  TensorDescriptor else_descriptor = op_def.src_tensors[2];
  TensorDescriptor dst_descriptor = op_def.dst_tensors[0];
  SelectV2Attributes attr = {broadcast_true, broadcast_false};
  GPUOperation operation = CreateSelectV2(op_def, attr);
  cond_descriptor.UploadData(cond_tensor);
  true_descriptor.UploadData(true_tensor);
  else_descriptor.UploadData(false_tensor);
  dst_descriptor.SetBHWCShape(BHWC(batch, height, width, channels));
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {&cond_descriptor, &true_descriptor, &else_descriptor}, {&dst_descriptor},
      std::make_unique<GPUOperation>(std::move(operation))));
  dst_descriptor.DownloadData(&dst_tensor);
  return absl::OkStatus();
}

absl::Status SelectV2Test(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, /*broadcast_true=*/false, /*broadcast_false=*/false,
                /*gather_by_rows=*/true);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(
          env, data_type, storage, precision, cond_tensor, true_tensor,
          false_tensor, kBatch, kHeight, kWidth, kChannels,
          /*broadcast_true=*/false, /*broadcast_false=*/false, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2BatchTest(TestExecutionEnvironment* env) {
  const int kBatch = 4;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, /*broadcast_true=*/false, /*broadcast_false=*/false,
                /*gather_by_rows=*/true);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(
          env, data_type, storage, precision, cond_tensor, true_tensor,
          false_tensor, kBatch, kHeight, kWidth, kChannels,
          /*broadcast_true=*/false, /*broadcast_false=*/false, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2BroadcastFalseTest(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  const bool kBroadcastTrue = false;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data = SetUpData(
      cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
      kChannels, kBroadcastTrue, kBroadcastFalse, /*gather_by_rows=*/true);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(env, data_type, storage, precision,
                                  cond_tensor, true_tensor, false_tensor,
                                  kBatch, kHeight, kWidth, kChannels,
                                  kBroadcastTrue, kBroadcastFalse, dst_tensor));

      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2BroadcastTrueTest(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  const bool kBroadcastTrue = true;
  const bool kBroadcastFalse = false;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data = SetUpData(
      cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
      kChannels, kBroadcastTrue, kBroadcastFalse, /*gather_by_rows=*/true);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(env, data_type, storage, precision,
                                  cond_tensor, true_tensor, false_tensor,
                                  kBatch, kHeight, kWidth, kChannels,
                                  kBroadcastTrue, kBroadcastFalse, dst_tensor));

      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2BroadcastBothTest(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 1;
  const bool kBroadcastTrue = true;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, kBroadcastTrue, kBroadcastFalse,
                /*gather_by_rows=*/true);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(env, data_type, storage, precision,
                                  cond_tensor, true_tensor, false_tensor,
                                  kBatch, kHeight, kWidth, kChannels,
                                  kBroadcastTrue, kBroadcastFalse, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2ChannelsTest(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 2;
  const int kChannels = 10;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, /*broadcast_true=*/false, /*broadcast_false=*/false,
                /*gather_by_rows=*/false);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(env, data_type, storage, precision,
                                  cond_tensor, true_tensor, false_tensor,
                                  kBatch, kHeight, kWidth, kChannels,
                                  /*broadcast_true=*/false,
                                  /*broadcast_false=*/false, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2ChannelsBatchTest(TestExecutionEnvironment* env) {
  const int kBatch = 3;
  const int kHeight = 1;
  const int kWidth = 2;
  const int kChannels = 10;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, /*broadcast_true=*/false, /*broadcast_false=*/false,
                /*gather_by_rows=*/false);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(env, data_type, storage, precision,
                                  cond_tensor, true_tensor, false_tensor,
                                  kBatch, kHeight, kWidth, kChannels,
                                  /*broadcast_true=*/false,
                                  /*broadcast_false=*/false, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SelectV2ChannelsBroadcastFalseTest(TestExecutionEnvironment* env) {
  const int kBatch = 1;
  const int kHeight = 3;
  const int kWidth = 2;
  const int kChannels = 4;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, DataType::FLOAT32> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData(cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
                kChannels, /*broadcast_true=*/false, kBroadcastFalse,
                /*gather_by_rows=*/false);

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (TensorStorageType storage : env->GetSupportedStorages(data_type)) {
      TensorFloat32 dst_tensor;
      RETURN_IF_ERROR(RunSelectV2(
          env, data_type, storage, precision, cond_tensor, true_tensor,
          false_tensor, kBatch, kHeight, kWidth, kChannels,
          /*broadcast_true=*/false, kBroadcastFalse, dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(expected_data, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
