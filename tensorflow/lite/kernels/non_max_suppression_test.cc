/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

using FloatData = std::initializer_list<float>;

static constexpr FloatData k_input_boxes_data{
    1, 1,     0, 0,     // Box 0
    0, 0.1,   1, 1.1,   // Box 1
    0, .9f,   1, -0.1,  // Box 2
    0, 10,    1, 11,    // Box 3
    1, 10.1f, 0, 11.1,  // Box 4
    1, 101,   0, 100    // Box 5
};

static constexpr FloatData k_input_scores_data{0.9f,  0.75f, 0.6f,
                                               0.95f, 0.5f,  0.3f};

static constexpr FloatData k_ik_input_iou_threshold_data{0.5f};

static constexpr FloatData k_input_score_threshold_data{0.0f};

static constexpr FloatData k_input_sigma_float_data{0.5f};

using Int8Data = std::initializer_list<int8_t>;
static constexpr Int8Data k_input_sigma_int8_data{0};

using Int16Data = std::initializer_list<int16_t>;
static constexpr Int16Data k_input_sigma_int16_data{16092};

static const auto k_min_max_input =
    std::minmax_element(k_input_boxes_data.begin(), k_input_boxes_data.end());

class BaseNMSOp : public SingleOpModel {
 public:
  BaseNMSOp(TensorType tensor_type, const bool static_shaped_outputs,
            const int max_output_size = -1)
      : tensor_type_{tensor_type} {
    if (tensor_type_ == TensorType_FLOAT32) {
      input_boxes_ = AddInput({tensor_type_, {k_num_boxes_, 4}});

      input_scores_ = AddInput({tensor_type, {k_num_boxes_}});
      if (static_shaped_outputs) {
        input_max_output_size_ =
            AddConstInput(TensorType_INT32, {max_output_size});
      } else {
        input_max_output_size_ = AddInput(TensorType_INT32);
      }
      input_iou_threshold_ = AddInput({tensor_type_, {}});
      input_score_threshold_ = AddInput({tensor_type_, {}});

    } else {
      TensorData boxes_tensor{tensor_type_,
                              {k_num_boxes_, 4},
                              *k_min_max_input.first,
                              *k_min_max_input.second};
      input_boxes_ = AddQuantizedInput(boxes_tensor);

      TensorData scores_tensor{tensor_type, {k_num_boxes_}, 0.0, 1.0};
      input_scores_ = AddQuantizedInput(scores_tensor);

      if (static_shaped_outputs) {
        input_max_output_size_ =
            AddConstInput(TensorType_INT32, {max_output_size});
      } else {
        input_max_output_size_ = AddInput(TensorType_INT32);
      }

      TensorData iou_threshold_tensor = {tensor_type_, {}, 0.0, 1.0};
      input_iou_threshold_ = AddQuantizedInput(iou_threshold_tensor);

      TensorData score_threshold_tensor{tensor_type_, {}, 0.0, 1.0};
      input_score_threshold_ = AddQuantizedInput(score_threshold_tensor);
    }

    output_selected_indices_ = AddOutput(TensorType_INT32);
  }

  void SetData(int index, const FloatData& data) {
    if (tensor_type_ == TensorType_INT8) {
      QuantizeAndPopulate<int8_t>(index, data);
    } else if (tensor_type_ == TensorType_INT16) {
      QuantizeAndPopulate<int16_t>(index, data);
    } else {
      PopulateTensor(index, data);
    }
  }

  void SetScores(const FloatData& scores) { SetData(input_scores_, scores); }

  void SetMaxOutputSize(int max_output_size) {
    PopulateTensor(input_max_output_size_, {max_output_size});
  }

  void SetScoreThreshold(float score_threshold) {
    SetData(input_score_threshold_, {score_threshold});
  }

  int input_boxes() { return input_boxes_; }
  int input_scores() { return input_scores_; }
  int input_iou_threshold() { return input_iou_threshold_; }
  int input_score_threshold() { return input_score_threshold_; }
  int input_sigma() { return input_sigma_; }

  std::vector<int> GetSelectedIndices() {
    return ExtractVector<int>(output_selected_indices_);
  }

  std::vector<float> GetSelectedScores() {
    return ExtractVector<float>(output_selected_scores_);
  }

  template <typename T>
  std::vector<float> GetDequantizedSelectedScores() {
    return Dequantize<T>(ExtractVector<T>(output_selected_scores_),
                         GetScale(output_selected_scores_),
                         GetZeroPoint(output_selected_scores_));
  }

  std::vector<int> GetNumSelectedIndices() {
    return ExtractVector<int>(output_num_selected_indices_);
  }

 protected:
  static const int k_num_boxes_ = 6;

  TensorType tensor_type_;

  int input_boxes_;
  int input_scores_;
  int input_max_output_size_;
  int input_iou_threshold_;
  int input_score_threshold_;
  int input_sigma_;

  int output_selected_indices_;
  int output_selected_scores_;
  int output_num_selected_indices_;

  virtual void PopulateModel() {
    SetData(input_boxes_, k_input_boxes_data);
    SetData(input_scores_, k_input_scores_data);
    SetData(input_iou_threshold_, k_ik_input_iou_threshold_data);
    SetData(input_score_threshold_, k_input_score_threshold_data);
  }

  void SymmetricInt16Scaling(TensorData& tensor) {
    // Symmetric range and null zero-point is required for INT16 tensors. As
    // SingleOpModel::QuantizationParams calculates the scale on an asymmetric
    // base [int_type::min, int_type::max], manually calculate the scale on a
    // symmetric range to ensure a null zero-point.
    if (tensor.type == TensorType_INT16) {
      tensor.scale =
          (tensor.max - tensor.min) / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
  }

  int AddQuantizedInput(TensorData& tensor) {
    if (tensor.type == TensorType_INT16) {
      SymmetricInt16Scaling(tensor);
    }

    return AddInput(tensor);
  }

  int AddQuantizedOutput(TensorData& tensor) {
    if (tensor.type == TensorType_INT16) {
      SymmetricInt16Scaling(tensor);
    }

    return AddOutput(tensor);
  }
};
}  // namespace

class NonMaxSuppressionV4OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV4OpModel(TensorType tensor_type,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1)
      : BaseNMSOp(tensor_type, static_shaped_outputs, max_output_size) {
    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V4,
                 BuiltinOptions_NonMaxSuppressionV4Options,
                 CreateNonMaxSuppressionV5Options(builder_).Union());

    BuildInterpreter({GetShape(input_boxes_), GetShape(input_scores_),
                      GetShape(input_max_output_size_),
                      GetShape(input_iou_threshold_),
                      GetShape(input_score_threshold_)});

    PopulateModel();
  }
};

TEST(NonMaxSuppressionV4OpModel, TestOutput) {
  NonMaxSuppressionV4OpModel nms(/**tensor_type=**/ TensorType_FLOAT32,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 0, 0, 0, 0}));

  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  // The first two indices should be zeroed-out.
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestInt8Output) {
  NonMaxSuppressionV4OpModel nms(
      /*tensor_type=*/TensorType_INT8,
      /*static_shaped_outputs=*/true,
      /*max_output_size=*/6);
  nms.SetScoreThreshold(0.4f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 0, 0, 0, 0}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestInt16Output) {
  NonMaxSuppressionV4OpModel nms(
      /*tensor_type=*/TensorType_INT16,
      /*static_shaped_outputs=*/true,
      /*max_output_size=*/6);
  nms.SetScoreThreshold(0.4f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 0, 0, 0, 0}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestDynamicOutput) {
  NonMaxSuppressionV4OpModel nms(/**tensor_type=**/ TensorType_FLOAT32,
                                 /**static_shaped_outputs=**/ false);
  nms.SetScoreThreshold(0.4f);

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));

  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestInt8DynamicOutput) {
  NonMaxSuppressionV4OpModel nms(/**tensor_type=**/ TensorType_INT8,
                                 /**static_shaped_outputs=**/ false);
  nms.SetScoreThreshold(0.4f);

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));

  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestInt16DynamicOutput) {
  NonMaxSuppressionV4OpModel nms(/**tensor_type=**/ TensorType_INT16,
                                 /**static_shaped_outputs=**/ false);
  nms.SetScoreThreshold(0.4f);

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));

  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestOutputWithZeroMaxOutput) {
  NonMaxSuppressionV4OpModel nms(/**tensor_type=**/ TensorType_FLOAT32,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 0);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
}

class NonMaxSuppressionV5OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV5OpModel(TensorType tensor_type,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1)
      : BaseNMSOp(tensor_type, static_shaped_outputs, max_output_size) {
    if (tensor_type_ == TensorType_FLOAT32) {
      input_sigma_ = AddInput({tensor_type_, {}});
      output_selected_scores_ = AddOutput({tensor_type, {}});
    } else {
      TensorData sigma_tensor{tensor_type_, {}, 0.0, 1.0};
      if (tensor_type == TensorType_INT8) {
        input_sigma_ =
            AddQuantizedConstInput(sigma_tensor, k_input_sigma_int8_data);
      } else {
        input_sigma_ =
            AddQuantizedConstInput(sigma_tensor, k_input_sigma_int16_data);
      }

      TensorData selected_scores_tensor{tensor_type_, {}, 0.0, 1.0};
      output_selected_scores_ = AddQuantizedOutput(selected_scores_tensor);
    }

    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V5,
                 BuiltinOptions_NonMaxSuppressionV5Options,
                 CreateNonMaxSuppressionV5Options(builder_).Union());

    BuildInterpreter(
        {GetShape(input_boxes_), GetShape(input_scores_),
         GetShape(input_max_output_size_), GetShape(input_iou_threshold_),
         GetShape(input_score_threshold_), GetShape(input_sigma_)});

    PopulateModel();
  }

  void PopulateModel() {
    BaseNMSOp::PopulateModel();
    SetData(input_sigma_, k_input_sigma_float_data);
  }

 protected:
  template <typename T>
  int AddQuantizedConstInput(TensorData& tensor,
                             std::initializer_list<T> data) {
    if (tensor.type == TensorType_INT16) {
      SymmetricInt16Scaling(tensor);
    }

    return AddConstInput(tensor, data);
  }
};

TEST(NonMaxSuppressionV5OpModel, TestOutput) {
  NonMaxSuppressionV5OpModel nms(/**tensor_type=**/TensorType_FLOAT32,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.0);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.95, 0.9, 0.3, 0.0, 0.0, 0.0}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST(NonMaxSuppressionV5OpModel, TestInt8Output) {
  const float kQuantizedTolerance = 0.01f;
  NonMaxSuppressionV5OpModel nms(
      /*tensor_type=*/TensorType_INT8,
      /*static_shaped_outputs=*/true,
      /*max_output_size=*/6);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5, 0, 0, 0}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3, 0.0, 0.0, 0.0},
                                              kQuantizedTolerance)));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              kQuantizedTolerance)));
}

TEST(NonMaxSuppressionV5OpModel, TestInt16Output) {
  const float kQuantizedTolerance = 0.01f;
  NonMaxSuppressionV5OpModel nms(
      /*tensor_type=*/TensorType_INT16,
      /*static_shaped_outputs=*/true,
      /*max_output_size=*/6);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5, 0, 0, 0}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int16_t>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3, 0.0, 0.0, 0.0},
                                              kQuantizedTolerance)));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int16_t>(),
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                              kQuantizedTolerance)));
}

TEST(NonMaxSuppressionV5OpModel, TestDynamicOutput) {
  NonMaxSuppressionV5OpModel nms(
      /*tensor_type=*/TensorType_FLOAT32,
      /*static_shaped_outputs=*/false,
      /*max_output_size=*/6);

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9}));

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95}));

  nms.SetMaxOutputSize(3);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9, 0.3}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.0, 0.0, 0.0}));
}

TEST(NonMaxSuppressionV5OpModel, TestInt8DynamicOutput) {
  const float kQuantizedTolerance = 0.01f;

  NonMaxSuppressionV5OpModel nms(
      /*tensor_type=*/TensorType_INT8,
      /*static_shaped_outputs=*/false,
      /*max_output_size=*/6);

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.95, 0.9}, kQuantizedTolerance)));

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int8_t>(),
              ElementsAreArray(ArrayFloatNear({0.95}, kQuantizedTolerance)));

  nms.SetMaxOutputSize(3);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, kQuantizedTolerance)));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int8_t>(),
      ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0}, kQuantizedTolerance)));
}

TEST(NonMaxSuppressionV5OpModel, TestInt16DynamicOutput) {
  const float kQuantizedTolerance = 0.01f;

  NonMaxSuppressionV5OpModel nms(
      /*tensor_type=*/TensorType_INT16,
      /*static_shaped_outputs=*/false,
      /*max_output_size=*/6);

  nms.SetMaxOutputSize(2);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.95, 0.9}, kQuantizedTolerance)));

  nms.SetMaxOutputSize(1);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetDequantizedSelectedScores<int16_t>(),
              ElementsAreArray(ArrayFloatNear({0.95}, kQuantizedTolerance)));

  nms.SetMaxOutputSize(3);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, kQuantizedTolerance)));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99f);
  ASSERT_EQ(nms.Invoke(), kTfLiteOk);
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(
      nms.GetDequantizedSelectedScores<int16_t>(),
      ElementsAreArray(ArrayFloatNear({0.0, 0.0, 0.0}, kQuantizedTolerance)));
}

}  // namespace tflite
