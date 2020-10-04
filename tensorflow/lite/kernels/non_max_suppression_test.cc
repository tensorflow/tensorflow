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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseNMSOp : public SingleOpModel {
 public:
  void SetScores(std::initializer_list<float> data) {
    PopulateTensor(input_scores_, data);
  }

  void SetMaxOutputSize(int max_output_size) {
    PopulateTensor(input_max_output_size_, {max_output_size});
  }

  void SetScoreThreshold(float score_threshold) {
    PopulateTensor(input_score_threshold_, {score_threshold});
  }

  std::vector<int> GetSelectedIndices() {
    return ExtractVector<int>(output_selected_indices_);
  }

  std::vector<float> GetSelectedScores() {
    return ExtractVector<float>(output_selected_scores_);
  }

  std::vector<int> GetNumSelectedIndices() {
    return ExtractVector<int>(output_num_selected_indices_);
  }

 protected:
  int input_boxes_;
  int input_scores_;
  int input_max_output_size_;
  int input_iou_threshold_;
  int input_score_threshold_;
  int input_sigma_;

  int output_selected_indices_;
  int output_selected_scores_;
  int output_num_selected_indices_;
};

class NonMaxSuppressionV4OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV4OpModel(const float iou_threshold,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1) {
    const int num_boxes = 6;
    input_boxes_ = AddInput({TensorType_FLOAT32, {num_boxes, 4}});
    input_scores_ = AddInput({TensorType_FLOAT32, {num_boxes}});
    if (static_shaped_outputs) {
      input_max_output_size_ =
          AddConstInput(TensorType_INT32, {max_output_size});
    } else {
      input_max_output_size_ = AddInput(TensorType_INT32);
    }
    input_iou_threshold_ = AddConstInput(TensorType_FLOAT32, {iou_threshold});
    input_score_threshold_ = AddInput({TensorType_FLOAT32, {}});

    output_selected_indices_ = AddOutput(TensorType_INT32);

    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V4,
                 BuiltinOptions_NonMaxSuppressionV4Options,
                 CreateNonMaxSuppressionV4Options(builder_).Union());
    BuildInterpreter({GetShape(input_boxes_), GetShape(input_scores_),
                      GetShape(input_max_output_size_),
                      GetShape(input_iou_threshold_),
                      GetShape(input_score_threshold_)});

    // Default data.
    PopulateTensor<float>(input_boxes_, {
                                            1, 1,     0, 0,     // Box 0
                                            0, 0.1,   1, 1.1,   // Box 1
                                            0, .9f,   1, -0.1,  // Box 2
                                            0, 10,    1, 11,    // Box 3
                                            1, 10.1f, 0, 11.1,  // Box 4
                                            1, 101,   0, 100    // Box 5
                                        });
  }
};

TEST(NonMaxSuppressionV4OpModel, TestOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 0, 0, 0, 0}));

  nms.SetScoreThreshold(0.99);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  // The first two indices should be zeroed-out.
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestDynamicOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ false);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);

  nms.SetMaxOutputSize(1);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));

  nms.SetMaxOutputSize(2);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));

  nms.SetScoreThreshold(0.99);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0}));
}

TEST(NonMaxSuppressionV4OpModel, TestOutputWithZeroMaxOutput) {
  NonMaxSuppressionV4OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 0);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.4);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
}

class NonMaxSuppressionV5OpModel : public BaseNMSOp {
 public:
  explicit NonMaxSuppressionV5OpModel(const float iou_threshold,
                                      const float sigma,
                                      const bool static_shaped_outputs,
                                      const int max_output_size = -1) {
    const int num_boxes = 6;
    input_boxes_ = AddInput({TensorType_FLOAT32, {num_boxes, 4}});
    input_scores_ = AddInput({TensorType_FLOAT32, {num_boxes}});
    if (static_shaped_outputs) {
      input_max_output_size_ =
          AddConstInput(TensorType_INT32, {max_output_size});
    } else {
      input_max_output_size_ = AddInput(TensorType_INT32);
    }
    input_iou_threshold_ = AddConstInput(TensorType_FLOAT32, {iou_threshold});
    input_score_threshold_ = AddInput({TensorType_FLOAT32, {}});
    input_sigma_ = AddConstInput(TensorType_FLOAT32, {sigma});

    output_selected_indices_ = AddOutput(TensorType_INT32);
    output_selected_scores_ = AddOutput(TensorType_FLOAT32);
    output_num_selected_indices_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_NON_MAX_SUPPRESSION_V5,
                 BuiltinOptions_NonMaxSuppressionV5Options,
                 CreateNonMaxSuppressionV5Options(builder_).Union());

    BuildInterpreter(
        {GetShape(input_boxes_), GetShape(input_scores_),
         GetShape(input_max_output_size_), GetShape(input_iou_threshold_),
         GetShape(input_score_threshold_), GetShape(input_sigma_)});

    // Default data.
    PopulateTensor<float>(input_boxes_, {
                                            1, 1,     0, 0,     // Box 0
                                            0, 0.1,   1, 1.1,   // Box 1
                                            0, .9f,   1, -0.1,  // Box 2
                                            0, 10,    1, 11,    // Box 3
                                            1, 10.1f, 0, 11.1,  // Box 4
                                            1, 101,   0, 100    // Box 5
                                        });
  }
};

TEST(NonMaxSuppressionV5OpModel, TestOutput) {
  NonMaxSuppressionV5OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**sigma=**/ 0.5,
                                 /**static_shaped_outputs=**/ true,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.0);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.95, 0.9, 0.3, 0.0, 0.0, 0.0}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(),
              ElementsAreArray({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
}

TEST(NonMaxSuppressionV5OpModel, TestDynamicOutput) {
  NonMaxSuppressionV5OpModel nms(/**iou_threshold=**/ 0.5,
                                 /**sigma=**/ 0.5,
                                 /**static_shaped_outputs=**/ false,
                                 /**max_output_size=**/ 6);
  nms.SetScores({0.9, 0.75, 0.6, 0.95, 0.5, 0.3});
  nms.SetScoreThreshold(0.0);

  nms.SetMaxOutputSize(2);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({2}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9}));

  nms.SetMaxOutputSize(1);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({1}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95}));

  nms.SetMaxOutputSize(3);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({3}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({3, 0, 5}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.95, 0.9, 0.3}));

  // No candidate gets selected. But the outputs should be zeroed out.
  nms.SetScoreThreshold(0.99);
  nms.Invoke();
  EXPECT_THAT(nms.GetNumSelectedIndices(), ElementsAreArray({0}));
  EXPECT_THAT(nms.GetSelectedIndices(), ElementsAreArray({0, 0, 0}));
  EXPECT_THAT(nms.GetSelectedScores(), ElementsAreArray({0.0, 0.0, 0.0}));
}
}  // namespace
}  // namespace tflite
