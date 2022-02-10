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
#ifndef TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_

#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
class BaseOpModel : public SingleOpModel {
 public:
  void SetAxis(const std::vector<int>& data) { PopulateTensor(axis_, data); }

  template <class T>
  void SetInput(std::vector<T> data) {
    PopulateTensor(input_, data);
  }

  template <class T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  int Input() { return input_; }

 protected:
  void SymmetricInt16Scaling(TensorData& tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range [int_type::min+1, int_type::max] to ensure a null
    // zero-point.
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
  }

 protected:
  int input_;
  int axis_;
  int output_;
};

// Model for the tests case where axis is a const tensor.
template <BuiltinOperator op_code, bool symmetric_int16_scaling = false>
class BaseConstOpModel : public BaseOpModel {
 public:
  BaseConstOpModel(TensorData input, TensorData output,
                   std::initializer_list<int> axis_shape,
                   std::initializer_list<int> axis, bool keep_dims) {
    if (symmetric_int16_scaling) {
      SymmetricInt16Scaling(input);
      SymmetricInt16Scaling(output);
    }
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(op_code, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Model for the tests case where axis is a dynamic tensor.
template <BuiltinOperator op_code, bool symmetric_int16_scaling = false>
class BaseDynamicOpModel : public BaseOpModel {
 public:
  BaseDynamicOpModel(TensorData input, TensorData output,
                     const TensorData& axis, bool keep_dims) {
    if (symmetric_int16_scaling) {
      SymmetricInt16Scaling(input);
      SymmetricInt16Scaling(output);
    }
    input_ = AddInput(input);
    axis_ = AddInput(axis);
    output_ = AddOutput(output);
    SetBuiltinOp(op_code, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_REDUCE_TEST_COMMON_H_
