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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_conversion.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_weights_converter.h"

namespace tflite {
namespace gpu {
namespace {
absl::Status ConvolutionWeightsConverterTest(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    const WeightsDescription& weight_desc, TestExecutionEnvironment* env,
    const OperationDef& op_def) {
  TensorFloat32 dst_tensor;
  const int flt_count =
      GetTotalElementsCountForLayout(weight_desc, weights.shape);
  dst_tensor.shape = BHWC(1, 1, 1, flt_count);
  dst_tensor.data.resize(flt_count);
  RearrangeWeights(
      weights, weight_desc, DataType::FLOAT32,
      absl::MakeSpan(reinterpret_cast<uint8_t*>(dst_tensor.data.data()),
                     flt_count * 4));

  // reinterpreting weights in OHWI as tensor in BHWC
  TensorFloat32 src_tensor;
  auto src_shape =
      BHWC(weights.shape.o, weights.shape.h, weights.shape.w, weights.shape.i);
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_shape.DimensionsProduct(), 2.0);
  for (int o = 0; o < weights.shape.o; ++o) {
    for (int y = 0; y < weights.shape.h; ++y) {
      for (int x = 0; x < weights.shape.w; ++x) {
        for (int i = 0; i < weights.shape.i; ++i) {
          const int f_index = weights.shape.LinearIndex({o, y, x, i});
          const int s_index = src_shape.LinearIndex({o, y, x, i});
          src_tensor.data[s_index] = weights.data[f_index];
        }
      }
    }
  }

  TensorFloat32 dst_tensor_gpu;
  auto converter = ConverterToConvWeights(op_def, weight_desc);
  RETURN_IF_ERROR(env->ExecuteGPUOperation(
      src_tensor,
      absl::make_unique<ConverterToConvWeights>(std::move(converter)),
      dst_tensor.shape, &dst_tensor_gpu));
  RETURN_IF_ERROR(PointWiseNear(dst_tensor.data, dst_tensor_gpu.data, 0.0f));
  return absl::OkStatus();
}

}  // namespace

absl::Status ConverterToConvWeights1x1OutX4Test(TestExecutionEnvironment* env) {
  const int kSrcChannels = 8;
  const int kDstChannels = 32;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      for (auto weights_layout :
           {WeightsLayout::kOHWIOGroupI4O4, WeightsLayout::kOHWIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        auto data_type = DeduceDataTypeFromPrecision(precision);
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX4UnalignedTest(
    TestExecutionEnvironment* env) {
  const int kSrcChannels = 8;
  const int kDstChannels = 17;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 4;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      for (auto weights_layout :
           {WeightsLayout::kOHWIOGroupI4O4, WeightsLayout::kOHWIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        auto data_type = DeduceDataTypeFromPrecision(precision);
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeights1x1OutX2Test(TestExecutionEnvironment* env) {
  const int kSrcChannels = 7;
  const int kDstChannels = 37;
  auto weights_shape = OHWI(kDstChannels, 1, 1, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      for (auto weights_layout :
           {WeightsLayout::kOHWIOGroupI4O4, WeightsLayout::kOHWIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        auto data_type = DeduceDataTypeFromPrecision(precision);
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvWeightsOutX2Test(TestExecutionEnvironment* env) {
  const int kSrcChannels = 8;
  const int kDstChannels = 38;
  auto weights_shape = OHWI(kDstChannels, 3, 4, kSrcChannels);
  WeightsDescription conv_weight_desc;
  conv_weight_desc.output_group_size = 2;

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      for (auto weights_layout :
           {WeightsLayout::kOHWIOGroupI4O4, WeightsLayout::kOHWIOGroupO4I4}) {
        conv_weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        auto data_type = DeduceDataTypeFromPrecision(precision);
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(ConvolutionWeightsConverterTest(
            weights, conv_weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConverterToConvTransposedWeights4x4Test(
    TestExecutionEnvironment* env) {
  const int kSrcChannels = 7;
  const int kDstChannels = 11;
  auto weights_shape = OHWI(kDstChannels, 4, 4, kSrcChannels);
  WeightsDescription weight_desc;
  weight_desc.spatial_remap = {10, 11, 14, 15, 8, 9, 12, 13,
                               2,  3,  6,  7,  0, 1, 4,  5};

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights_shape.DimensionsProduct());
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = half(static_cast<float>(i));
  }

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      for (auto weights_layout : {WeightsLayout::kOICustomSpatialI4O4,
                                  WeightsLayout::kOICustomSpatialO4I4}) {
        weight_desc.layout = weights_layout;
        OperationDef op_def;
        op_def.precision = precision;
        auto data_type = DeduceDataTypeFromPrecision(precision);
        op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
        op_def.dst_tensors.push_back(
            {data_type, TensorStorageType::BUFFER, Layout::UNKNOWN});
        RETURN_IF_ERROR(
            ConvolutionWeightsConverterTest(weights, weight_desc, env, op_def));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
