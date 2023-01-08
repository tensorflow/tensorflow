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

#include "tensorflow/lite/delegates/gpu/common/tasks/cumsum_test_util.h"

#include <memory>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/cumsum.h"

namespace tflite {
namespace gpu {

template <Axis axis>
absl::Status CumsumHWC(TestExecutionEnvironment* env) {
  Tensor<HWC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = HWC(8, 6, 4);
  BHWC shape = BHWC(1, 8, 6, 4);
  src_tensor.data = {
      7,  -3, -5, 8,  1,  4,  7,  1,  4,  -9, 6,  -5, 9,  -1, 5,  3,  -6, -2,
      -3, 3,  -1, -8, -9, -1, 2,  3,  8,  -4, -9, -2, 7,  8,  6,  9,  1,  -1,
      -9, -6, -9, 1,  -5, 3,  3,  9,  7,  6,  -6, -9, 4,  -9, -6, -7, 6,  -5,
      -8, 4,  4,  5,  -6, -1, 9,  -3, 3,  9,  7,  4,  -6, -5, -2, 7,  -6, -5,
      -9, 0,  -7, -3, -3, 6,  -6, -2, -4, 4,  2,  -2, 0,  -6, -4, 5,  -6, -3,
      1,  -6, 0,  5,  -2, -8, 6,  4,  -6, -5, 8,  3,  -5, -2, -9, -4, 5,  -2,
      -7, -7, -6, 1,  5,  2,  9,  1,  -2, 9,  1,  -2, 6,  -5, -3, -6, 3,  7,
      5,  -6, 6,  3,  -3, 4,  6,  8,  7,  -3, 3,  -3, 3,  -4, -1, -4, -7, -4,
      1,  -5, -4, -5, -1, -9, 9,  -3, 7,  3,  4,  -5, 8,  9,  -6, 5,  -5, -9,
      2,  3,  4,  5,  -9, 0,  2,  0,  -2, 8,  8,  -5, -6, -6, -7, -1, 7,  -3,
      2,  6,  4,  -9, 1,  -9, -5, -6, -5, -1, 8,  2};
  std::map<Axis, std::vector<float>> expected = {
      {Axis::HEIGHT,
       {7,   -3,  -5,  8,   1,   4,   7,   1,   4,   -9,  6,   -5,  9,   -1,
        5,   3,   -6,  -2,  -3,  3,   -1,  -8,  -9,  -1,  9,   0,   3,   4,
        -8,  2,   14,  9,   10,  0,   7,   -6,  0,   -7,  -4,  4,   -11, 1,
        0,   12,  6,   -2,  -15, -10, 13,  -9,  -3,  -3,  -2,  -3,  6,   13,
        14,  5,   1,   -7,  9,   -10, -1,  13,  -4,  5,   -6,  7,   4,   5,
        -21, -15, 4,   -9,  -10, -6,  -5,  3,   0,   11,  10,  9,   3,   -9,
        9,   -16, -5,  18,  -10, 2,   -5,  1,   4,   10,  -23, -23, 10,  -5,
        -16, -11, 3,   6,   -5,  9,   1,   5,   8,   -11, 2,   -23, -11, 19,
        -5,  4,   4,   2,   2,   19,  -22, -25, 16,  -10, -19, -17, 6,   13,
        0,   3,   7,   8,   5,   -7,  8,   -15, -4,  16,  -2,  1,   7,   -2,
        1,   15,  -29, -29, 17,  -15, -23, -22, 5,   4,   9,   0,   14,  11,
        9,   -12, 16,  -6,  -10, 21,  -7,  -8,  9,   1,   5,   20,  -38, -29,
        19,  -15, -25, -14, 13,  -1,  3,   -6,  7,   10,  16,  -15, 18,  0,
        -6,  12,  -6,  -17, 4,   -5,  0,   19,  -30, -27}},
      {Axis::WIDTH,
       {7,   -3,  -5,  8,   8,   1,   2,   9,   12,  -8,  8,   4,   21,  -9,
        13,  7,   15,  -11, 10,  10,  14,  -19, 1,   9,   2,   3,   8,   -4,
        -7,  1,   15,  4,   -1,  10,  16,  3,   -10, 4,   7,   4,   -15, 7,
        10,  13,  -8,  13,  4,   4,   4,   -9,  -6,  -7,  10,  -14, -14, -3,
        14,  -9,  -20, -4,  23,  -12, -17, 5,   30,  -8,  -23, 0,   28,  -1,
        -29, -5,  -9,  0,   -7,  -3,  -12, 6,   -13, -5,  -16, 10,  -11, -7,
        -16, 4,   -15, -2,  -22, 1,   -14, -8,  -22, 6,   -16, -16, 6,   4,
        -6,  -5,  14,  7,   -11, -7,  5,   3,   -6,  -9,  -2,  -4,  -12, -8,
        3,   -2,  -3,  -7,  1,   7,   -2,  -9,  6,   -5,  -3,  -6,  9,   2,
        2,   -12, 15,  5,   -1,  -8,  21,  13,  6,   -11, 24,  10,  9,   -15,
        23,  6,   2,   -19, 1,   -5,  -4,  -5,  0,   -14, 5,   -8,  7,   -11,
        9,   -13, 15,  -2,  3,   -8,  10,  -11, 5,   -5,  14,  -6,  -4,  -5,
        2,   0,   -2,  8,   10,  -5,  -8,  2,   3,   -6,  -1,  -1,  5,   0,
        3,   -10, 6,   -9,  -2,  -16, 1,   -10, 6,   -14}},
      {Axis::CHANNELS,
       {7,   4,   -1,  7,   1,   5,   12,  13,  4,   -5,  1,  -4,  9,   8,  13,
        16,  -6,  -8,  -11, -8,  -1,  -9,  -18, -19, 2,   5,  13,  9,   -9, -11,
        -4,  4,   6,   15,  16,  15,  -9,  -15, -24, -23, -5, -2,  1,   10, 7,
        13,  7,   -2,  4,   -5,  -11, -18, 6,   1,   -7,  -3, 4,   9,   3,  2,
        9,   6,   9,   18,  7,   11,  5,   0,   -2,  5,   -1, -6,  -9,  -9, -16,
        -19, -3,  3,   -3,  -5,  -4,  0,   2,   0,   0,   -6, -10, -5,  -6, -9,
        -8,  -14, 0,   5,   3,   -5,  6,   10,  4,   -1,  8,  11,  6,   4,  -9,
        -13, -8,  -10, -7,  -14, -20, -19, 5,   7,   16,  17, -2,  7,   8,  6,
        6,   1,   -2,  -8,  3,   10,  15,  9,   6,   9,   6,  10,  6,   14, 21,
        18,  3,   0,   3,   -1,  -1,  -5,  -12, -16, 1,   -4, -8,  -13, -1, -10,
        -1,  -4,  7,   10,  14,  9,   8,   17,  11,  16,  -5, -14, -12, -9, 4,
        9,   0,   0,   2,   2,   0,   8,   8,   3,   -3,  -9, -7,  -8,  -1, -4,
        2,   8,   12,  3,   1,   -8,  -13, -19, -5,  -6,  2,  4}}};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.precision = precision;
      TensorDescriptor& src = op_def.src_tensors[0];
      TensorDescriptor& dst = op_def.dst_tensors[0];
      CumsumAttributes attr = {axis};
      Cumsum operation = CreateCumsum(op_def, attr);
      dst.SetBHWCShape(shape);
      src.UploadData(src_tensor);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src}, {&dst}, std::make_unique<Cumsum>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(expected[axis], dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

template <Axis axis>
absl::Status CumsumBHWC(TestExecutionEnvironment* env) {
  Tensor<BHWC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = BHWC(6, 8, 1, 4);
  BHWC shape = BHWC(6, 8, 1, 4);
  src_tensor.data = {
      -6, -9, -9, 3,  0,  -6, -7, 6,  -3, 5,  -2, 0,  -6, -7, -1, 0,  6,  -7,
      5,  -7, -1, 0,  9,  2,  -9, 7,  -9, 1,  -8, 1,  -4, -8, -4, -7, -5, 0,
      -2, 0,  9,  -7, 2,  7,  4,  1,  0,  0,  -7, 7,  2,  1,  6,  3,  7,  -4,
      -5, -9, 5,  9,  -5, 3,  4,  5,  -2, -1, -5, 9,  -3, 6,  3,  -1, -2, 3,
      -8, -4, -8, -4, -2, 8,  -5, 2,  -1, 8,  5,  1,  8,  5,  7,  -6, -8, -3,
      9,  3,  -6, -9, -6, 4,  0,  0,  1,  -5, 3,  6,  -6, 2,  -2, 1,  9,  2,
      -5, -8, -8, -4, 8,  -1, 7,  -3, 2,  -9, 2,  -2, -1, -6, 7,  2,  -5, -3,
      2,  0,  3,  -4, -9, 7,  -6, 9,  9,  5,  -1, 5,  8,  8,  1,  -4, -6, -7,
      2,  1,  -2, -5, 8,  -5, -4, 9,  6,  3,  3,  -9, -2, 1,  -7, 5,  1,  6,
      0,  -3, -5, 1,  -3, -5, -1, -9, -5, 7,  -7, -3, 5,  6,  8,  -5, -9, -7,
      1,  -6, -4, -3, -3, 2,  7,  6,  3,  8,  -8, -2};
  std::map<Axis, std::vector<float>> expected = {
      {Axis::BATCH,
       {-6,  -9,  -9,  3,   0,   -6,  -7,  6,   -3,  5,   -2,  0,   -6,  -7,
        -1,  0,   6,   -7,  5,   -7,  -1,  0,   9,   2,   -9,  7,   -9,  1,
        -8,  1,   -4,  -8,  -10, -16, -14, 3,   -2,  -6,  2,   -1,  -1,  12,
        2,   1,   -6,  -7,  -8,  7,   8,   -6,  11,  -4,  6,   -4,  4,   -7,
        -4,  16,  -14, 4,   -4,  6,   -6,  -9,  -15, -7,  -17, 9,   1,   -7,
        0,   2,   -9,  8,   -6,  -3,  -8,  1,   -13, 9,   7,   2,   16,  -3,
        14,  1,   11,  -13, -12, 13,  -5,  7,   -10, -3,  -12, -5,  -15, -7,
        -16, 4,   4,   -1,  -6,  4,   -11, 9,   3,   -1,  -13, -7,  -21, 5,
        15,  1,   23,  -6,  16,  -8,  13,  -15, -13, 7,   2,   9,   -15, -6,
        -10, -5,  -12, -11, -25, 11,  -2,  8,   3,   9,   -12, 14,  11,  7,
        -12, -11, -27, -2,  17,  2,   21,  -11, 24,  -13, 9,   -6,  -7,  10,
        5,   0,   -17, -5,  -17, 0,   -11, -5,  -25, 8,   -7,  9,   0,   4,
        -13, 5,   6,   14,  -19, -14, -22, 4,   25,  -3,  12,  -18, 25,  -19,
        5,   -9,  -10, 12,  12,  6,   -14, 3,   -25, -2}},
      {Axis::HEIGHT,
       {-6,  -9,  -9,  3,   -6, -15, -16, 9,   -9, -10, -18, 9,   -15, -17, -19,
        9,   -9,  -24, -14, 2,  -10, -24, -5,  4,  -19, -17, -14, 5,   -27, -16,
        -18, -3,  -4,  -7,  -5, 0,   -6,  -7,  4,  -7,  -4,  0,   8,   -6,  -4,
        0,   1,   1,   -2,  1,  7,   4,   5,   -3, 2,   -5,  10,  6,   -3,  -2,
        14,  11,  -5,  -3,  -5, 9,   -3,  6,   -2, 8,   -5,  9,   -10, 4,   -13,
        5,   -12, 12,  -18, 7,  -13, 20,  -13, 8,  -5,  25,  -6,  2,   -13, 22,
        3,   5,   -19, 13,  -3, 9,   0,   0,   1,  -5,  3,   6,   -5,  -3,  1,
        7,   4,   -1,  -4,  -1, -4,  -5,  4,   -2, 3,   -8,  6,   -11, 5,   -10,
        5,   -17, 12,  -8,  0,  -20, 14,  -8,  3,  -4,  -9,  7,   -3,  5,   0,
        12,  -4,  10,  8,   20, -3,  6,   2,   13, -1,  7,   0,   8,   7,   2,
        -4,  17,  13,  5,   -1, 8,   11,  6,   -8, 13,  1,   6,   0,   -3,  -4,
        7,   -3,  -8,  -5,  -2, -8,  -1,  -12, -5, -3,  5,   -4,  -10, -12, -2,
        -3,  -16, -16, -5,  -6, -14, -9,  1,   -3, -6,  -17, -1}},
      {Axis::WIDTH,
       {-6, -9, -9, 3,  0,  -6, -7, 6,  -3, 5,  -2, 0,  -6, -7, -1, 0,  6,  -7,
        5,  -7, -1, 0,  9,  2,  -9, 7,  -9, 1,  -8, 1,  -4, -8, -4, -7, -5, 0,
        -2, 0,  9,  -7, 2,  7,  4,  1,  0,  0,  -7, 7,  2,  1,  6,  3,  7,  -4,
        -5, -9, 5,  9,  -5, 3,  4,  5,  -2, -1, -5, 9,  -3, 6,  3,  -1, -2, 3,
        -8, -4, -8, -4, -2, 8,  -5, 2,  -1, 8,  5,  1,  8,  5,  7,  -6, -8, -3,
        9,  3,  -6, -9, -6, 4,  0,  0,  1,  -5, 3,  6,  -6, 2,  -2, 1,  9,  2,
        -5, -8, -8, -4, 8,  -1, 7,  -3, 2,  -9, 2,  -2, -1, -6, 7,  2,  -5, -3,
        2,  0,  3,  -4, -9, 7,  -6, 9,  9,  5,  -1, 5,  8,  8,  1,  -4, -6, -7,
        2,  1,  -2, -5, 8,  -5, -4, 9,  6,  3,  3,  -9, -2, 1,  -7, 5,  1,  6,
        0,  -3, -5, 1,  -3, -5, -1, -9, -5, 7,  -7, -3, 5,  6,  8,  -5, -9, -7,
        1,  -6, -4, -3, -3, 2,  7,  6,  3,  8,  -8, -2}},
      {Axis::CHANNELS,
       {-6,  -15, -24, -21, 0,   -6,  -13, -7, -3,  2,  0,   0,   -6,  -13, -14,
        -14, 6,   -1,  4,   -3,  -1,  -1,  8,  10,  -9, -2,  -11, -10, -8,  -7,
        -11, -19, -4,  -11, -16, -16, -2,  -2, 7,   0,  2,   9,   13,  14,  0,
        0,   -7,  0,   2,   3,   9,   12,  7,  3,   -2, -11, 5,   14,  9,   12,
        4,   9,   7,   6,   -5,  4,   1,   7,  3,   2,  0,   3,   -8,  -12, -20,
        -24, -2,  6,   1,   3,   -1,  7,   12, 13,  8,  13,  20,  14,  -8,  -11,
        -2,  1,   -6,  -15, -21, -17, 0,   0,  1,   -4, 3,   9,   3,   5,   -2,
        -1,  8,   10,  -5,  -13, -21, -25, 8,  7,   14, 11,  2,   -7,  -5,  -7,
        -1,  -7,  0,   2,   -5,  -8,  -6,  -6, 3,   -1, -10, -3,  -6,  3,   12,
        17,  -1,  4,   12,  20,  1,   -3,  -9, -16, 2,  3,   1,   -4,  8,   3,
        -1,  8,   6,   9,   12,  3,   -2,  -1, -8,  -3, 1,   7,   7,   4,   -5,
        -4,  -7,  -12, -1,  -10, -15, -8,  -7, -10, -5, 1,   8,   3,   -6,  -13,
        1,   -5,  -9,  -12, -3,  -1,  6,   12, 3,   11, 3,   1}}};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.precision = precision;
      TensorDescriptor& src = op_def.src_tensors[0];
      TensorDescriptor& dst = op_def.dst_tensors[0];
      CumsumAttributes attr = {axis};
      Cumsum operation = CreateCumsum(op_def, attr);
      dst.SetBHWCShape(shape);
      src.UploadData(src_tensor);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src}, {&dst}, std::make_unique<Cumsum>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(PointWiseNear(expected[axis], dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status CumsumHWCTest(TestExecutionEnvironment* env) {
  RETURN_IF_ERROR(CumsumHWC<Axis::HEIGHT>(env));
  RETURN_IF_ERROR(CumsumHWC<Axis::WIDTH>(env));
  RETURN_IF_ERROR(CumsumHWC<Axis::CHANNELS>(env));
  return absl::OkStatus();
}

absl::Status CumsumBHWCTest(TestExecutionEnvironment* env) {
  RETURN_IF_ERROR(CumsumBHWC<Axis::BATCH>(env));
  RETURN_IF_ERROR(CumsumBHWC<Axis::HEIGHT>(env));
  RETURN_IF_ERROR(CumsumBHWC<Axis::WIDTH>(env));
  RETURN_IF_ERROR(CumsumBHWC<Axis::CHANNELS>(env));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
