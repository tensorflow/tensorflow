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
#ifndef TENSORFLOW_LITE_KERNELS_UNSORTED_SEGMENT_TEST_H_
#define TENSORFLOW_LITE_KERNELS_UNSORTED_SEGMENT_TEST_H_

#include <limits.h>
#include <stdint.h>

#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
template <typename T>
class UnsortedSegmentModel : public SingleOpModel {
 public:
  UnsortedSegmentModel(const TensorData& data, const TensorData& segment_ids,
                       const TensorData& num_segments, const BuiltinOperator op,
                       const BuiltinOptions options) {
    data_id_ = AddInput(data);
    segment_ids_id_ = AddInput(segment_ids);
    num_segments_id_ = AddInput(num_segments);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(op, options, 0);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_),
                      GetShape(num_segments_id_)});
  }

  explicit UnsortedSegmentModel(
      const TensorData& data, const std::initializer_list<int>& segment_id_data,
      const std::initializer_list<int>& segment_id_shape,
      const std::initializer_list<int>& num_segments_data,
      const std::initializer_list<int>& num_segments_shape,
      const BuiltinOperator op, const BuiltinOptions options) {
    data_id_ = AddInput(data);
    segment_ids_id_ =
        AddConstInput(TensorType_INT32, segment_id_data, segment_id_shape);
    num_segments_id_ =
        AddConstInput(TensorType_INT32, num_segments_data, num_segments_shape);
    output_id_ = AddOutput(data.type);
    SetBuiltinOp(op, options, 0);
    BuildInterpreter({GetShape(data_id_), GetShape(segment_ids_id_),
                      GetShape(num_segments_id_)});
  }

  int data() const { return data_id_; }
  int segment_ids() const { return segment_ids_id_; }
  int num_segments() const { return num_segments_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int data_id_;
  int segment_ids_id_;
  int num_segments_id_;
  int output_id_;
};

class UnsortedSegmentTest : public ::testing::TestWithParam<BuiltinOperator> {
 public:
  UnsortedSegmentModel<int32_t> getModel(const TensorData& data,
                                         const TensorData& segment_ids,
                                         const TensorData& num_segments) {
    return UnsortedSegmentModel<int32_t>(data, segment_ids, num_segments,
                                         GetParam(), BuiltinOptions_NONE);
  }
  UnsortedSegmentModel<int32_t> getConstModel(
      const TensorData& data, const std::initializer_list<int>& segment_id_data,
      const std::initializer_list<int>& segment_id_shape,
      const std::initializer_list<int>& num_segments_data,
      const std::initializer_list<int>& num_segments_shape) {
    return UnsortedSegmentModel<int32_t>(
        data, segment_id_data, segment_id_shape, num_segments_data,
        num_segments_shape, GetParam(), BuiltinOptions_NONE);
  }
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_UNSORTED_SEGMENT_TEST_H_
