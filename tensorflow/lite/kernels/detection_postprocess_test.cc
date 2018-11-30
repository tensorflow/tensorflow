/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_DETECTION_POSTPROCESS();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// Tests for scenarios where we DO NOT set use_regular_nms flag
class BaseDetectionPostprocessOpModel : public SingleOpModel {
 public:
  BaseDetectionPostprocessOpModel(const TensorData& input1,
                            const TensorData& input2,
                            const TensorData& input3,
                            const TensorData& output1,
                            const TensorData& output2,
                            const TensorData& output3,
                            const TensorData& output4) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    input3_ = AddInput(input3);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);
    output3_ = AddOutput(output3);
    output4_ = AddOutput(output4);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("max_detections", 3);
      fbb.Int("max_classes_per_detection", 1);
      fbb.Float("nms_score_threshold", 0.0);
      fbb.Float("nms_iou_threshold", 0.5);
      fbb.Int("num_classes", 2);
      fbb.Float("y_scale", 10.0);
      fbb.Float("x_scale", 10.0);
      fbb.Float("h_scale", 5.0);
      fbb.Float("w_scale", 5.0);
    });
    fbb.Finish();
    SetCustomOp("TFLite_Detection_PostProcess", fbb.GetBuffer(),
                Register_DETECTION_POSTPROCESS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int input3() { return input3_; }

  template <class T>
  void SetInput1(std::initializer_list<T> data) {
    PopulateTensor<T>(input1_, data);
  }

  template <class T>
  void SetInput2(std::initializer_list<T> data) {
    PopulateTensor<T>(input2_, data);
  }

  template <class T>
  void SetInput3(std::initializer_list<T> data) {
    PopulateTensor<T>(input3_, data);
  }

  template <class T>
  std::vector<T> GetOutput1() {
    return ExtractVector<T>(output1_);
  }

  template <class T>
  std::vector<T> GetOutput2() {
    return ExtractVector<T>(output2_);
  }

  template <class T>
  std::vector<T> GetOutput3() {
    return ExtractVector<T>(output3_);
  }

  template <class T>
  std::vector<T> GetOutput4() {
    return ExtractVector<T>(output4_);
  }

  std::vector<int> GetOutputShape1() { return GetTensorShape(output1_); }
  std::vector<int> GetOutputShape2() { return GetTensorShape(output2_); }
  std::vector<int> GetOutputShape3() { return GetTensorShape(output3_); }
  std::vector<int> GetOutputShape4() { return GetTensorShape(output4_); }

 protected:
  int input1_;
  int input2_;
  int input3_;
  int output1_;
  int output2_;
  int output3_;
  int output4_;
};

TEST(DetectionPostprocessOpTest, FloatTest) {
  BaseDetectionPostprocessOpModel m(
      {TensorType_FLOAT32, {1, 6, 4}}, {TensorType_FLOAT32, {1, 6, 3}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}});

  // six boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  });
  // class scores - two classes with background
  m.SetInput2<float>({0., .9, .8, 0., .75, .72, 0., .6, .5, 0., .93, .95, 0.,
                      .5, .4, 0., .3, .2});
  // six anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  //  anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  });
  // Same boxes in box-corner encoding:
  // { 0.0, 0.0, 1.0, 1.0,
  //   0.0, 0.1, 1.0, 1.1,
  //   0.0, -0.1, 1.0, 0.9,
  //   0.0, 10.0, 1.0, 11.0,
  //   0.0, 10.1, 1.0, 11.1,
  //   0.0, 100.0, 1.0, 101.0}
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(
      m.GetOutput1<float>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 101.0},
          1e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-1)));
}

TEST(DetectionPostprocessOpTest, QuantizedTest) {
  BaseDetectionPostprocessOpModel m(
      {TensorType_UINT8, {1, 6, 4}, -1.0, 1.0},
      {TensorType_UINT8, {1, 6, 3}, 0.0, 1.0},
      {TensorType_UINT8, {6, 4}, 0.0, 100.5}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}});
  // six boxes in center-size encoding
  std::vector<std::vector<float>> inputs1 = {{
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[0]);
  // class scores - two classes with background
  std::vector<std::vector<float>> inputs2 = {{0., .9, .8, 0., .75, .72, 0., .6,
                                              .5, 0., .93, .95, 0., .5, .4, 0.,
                                              .3, .2}};
  m.QuantizeAndPopulate<uint8_t>(m.input2(), inputs2[0]);
  // six anchors in center-size encoding
  std::vector<std::vector<float>> inputs3 = {{
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  // anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input3(), inputs3[0]);
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(
      m.GetOutput1<float>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 101.0},
          3e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-1)));
}

// Tests for scenarios where we set use_regular_nms flag
class DetectionPostprocessOpModelwithRegularNMS : public SingleOpModel {
 public:
  DetectionPostprocessOpModelwithRegularNMS(
      const TensorData& input1, const TensorData& input2,
      const TensorData& input3, const TensorData& output1,
      const TensorData& output2, const TensorData& output3,
      const TensorData& output4, bool use_regular_nms) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    input3_ = AddInput(input3);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);
    output3_ = AddOutput(output3);
    output4_ = AddOutput(output4);

    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("max_detections", 3);
      fbb.Int("max_classes_per_detection", 1);
      fbb.Int("detections_per_class", 1);
      fbb.Bool("use_regular_nms", use_regular_nms);
      fbb.Float("nms_score_threshold", 0.0);
      fbb.Float("nms_iou_threshold", 0.5);
      fbb.Int("num_classes", 2);
      fbb.Float("y_scale", 10.0);
      fbb.Float("x_scale", 10.0);
      fbb.Float("h_scale", 5.0);
      fbb.Float("w_scale", 5.0);
    });
    fbb.Finish();
    SetCustomOp("TFLite_Detection_PostProcess", fbb.GetBuffer(),
                Register_DETECTION_POSTPROCESS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int input3() { return input3_; }

  template <class T>
  void SetInput1(std::initializer_list<T> data) {
    PopulateTensor<T>(input1_, data);
  }

  template <class T>
  void SetInput2(std::initializer_list<T> data) {
    PopulateTensor<T>(input2_, data);
  }

  template <class T>
  void SetInput3(std::initializer_list<T> data) {
    PopulateTensor<T>(input3_, data);
  }

  template <class T>
  std::vector<T> GetOutput1() {
    return ExtractVector<T>(output1_);
  }

  template <class T>
  std::vector<T> GetOutput2() {
    return ExtractVector<T>(output2_);
  }

  template <class T>
  std::vector<T> GetOutput3() {
    return ExtractVector<T>(output3_);
  }

  template <class T>
  std::vector<T> GetOutput4() {
    return ExtractVector<T>(output4_);
  }

  std::vector<int> GetOutputShape1() { return GetTensorShape(output1_); }
  std::vector<int> GetOutputShape2() { return GetTensorShape(output2_); }
  std::vector<int> GetOutputShape3() { return GetTensorShape(output3_); }
  std::vector<int> GetOutputShape4() { return GetTensorShape(output4_); }

 protected:
  int input1_;
  int input2_;
  int input3_;
  int output1_;
  int output2_;
  int output3_;
  int output4_;
};

TEST(DetectionPostprocessOpTest, FloatTestFastNMS) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_FLOAT32, {1, 6, 4}}, {TensorType_FLOAT32, {1, 6, 3}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);

  // six boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  });
  // class scores - two classes with background
  m.SetInput2<float>({0., .9, .8, 0., .75, .72, 0., .6, .5, 0., .93, .95, 0.,
                      .5, .4, 0., .3, .2});
  // six anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  //  anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  });
  // Same boxes in box-corner encoding:
  // { 0.0, 0.0, 1.0, 1.0,
  //   0.0, 0.1, 1.0, 1.1,
  //   0.0, -0.1, 1.0, 0.9,
  //   0.0, 10.0, 1.0, 11.0,
  //   0.0, 10.1, 1.0, 11.1,
  //   0.0, 100.0, 1.0, 101.0}
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(
      m.GetOutput1<float>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 101.0},
          1e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-1)));
}

TEST(DetectionPostprocessOpTest, QuantizedTestFastNMS) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_UINT8, {1, 6, 4}, -1.0, 1.0},
      {TensorType_UINT8, {1, 6, 3}, 0.0, 1.0},
      {TensorType_UINT8, {6, 4}, 0.0, 100.5}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);
  // six boxes in center-size encoding
  std::vector<std::vector<float>> inputs1 = {{
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[0]);
  // class scores - two classes with background
  std::vector<std::vector<float>> inputs2 = {{0., .9, .8, 0., .75, .72, 0., .6,
                                              .5, 0., .93, .95, 0., .5, .4, 0.,
                                              .3, .2}};
  m.QuantizeAndPopulate<uint8_t>(m.input2(), inputs2[0]);
  // six anchors in center-size encoding
  std::vector<std::vector<float>> inputs3 = {{
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  // anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input3(), inputs3[0]);
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(
      m.GetOutput1<float>(),
      ElementsAreArray(ArrayFloatNear(
          {0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 1.0, 1.0, 0.0, 100.0, 1.0, 101.0},
          3e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-1)));
}

TEST(DetectionPostprocessOpTest, FloatTestRegularNMS) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_FLOAT32, {1, 6, 4}}, {TensorType_FLOAT32, {1, 6, 3}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, true);
  // six boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  });
  // class scores - two classes with background
  m.SetInput2<float>({0., .9, .8, 0., .75, .72, 0., .6, .5, 0., .93, .95, 0.,
                      .5, .4, 0., .3, .2});
  // six anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  //  anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  });
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(m.GetOutput1<float>(),
              ElementsAreArray(ArrayFloatNear({0.0, 10.0, 1.0, 11.0, 0.0, 10.0,
                                               1.0, 11.0, 0.0, 0.0, 0.0, 0.0},
                                              3e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.0}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({2.0}, 1e-1)));
}

TEST(DetectionPostprocessOpTest, QuantizedTestRegularNMS) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_UINT8, {1, 6, 4}, -1.0, 1.0},
      {TensorType_UINT8, {1, 6, 3}, 0.0, 1.0},
      {TensorType_UINT8, {6, 4}, 0.0, 100.5}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, true);
  // six boxes in center-size encoding
  std::vector<std::vector<float>> inputs1 = {{
      0.0, 0.0,  0.0, 0.0,  // box #1
      0.0, 1.0,  0.0, 0.0,  // box #2
      0.0, -1.0, 0.0, 0.0,  // box #3
      0.0, 0.0,  0.0, 0.0,  // box #4
      0.0, 1.0,  0.0, 0.0,  // box #5
      0.0, 0.0,  0.0, 0.0   // box #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[0]);
  // class scores - two classes with background
  std::vector<std::vector<float>> inputs2 = {{0., .9, .8, 0., .75, .72, 0., .6,
                                              .5, 0., .93, .95, 0., .5, .4, 0.,
                                              .3, .2}};
  m.QuantizeAndPopulate<uint8_t>(m.input2(), inputs2[0]);
  // six anchors in center-size encoding
  std::vector<std::vector<float>> inputs3 = {{
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  // anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input3(), inputs3[0]);
  m.Invoke();
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  EXPECT_THAT(m.GetOutput1<float>(),
              ElementsAreArray(ArrayFloatNear({0.0, 10.0, 1.0, 11.0, 0.0, 10.0,
                                               1.0, 11.0, 0.0, 0.0, 0.0, 0.0},
                                              3e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.0}, 1e-1)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({2.0}, 1e-1)));
}
}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
