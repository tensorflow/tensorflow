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
#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
  BaseDetectionPostprocessOpModel(
      const TensorData& input1, const TensorData& input2,
      const TensorData& input3, const TensorData& output1,
      const TensorData& output2, const TensorData& output3,
      const TensorData& output4, int max_classes_per_detection = 1) {
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
      fbb.Int("max_classes_per_detection", max_classes_per_detection);
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-4)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-4)));
}

// Tests the case when a box degenerates to a point (xmin==xmax, ymin==ymax).
TEST(DetectionPostprocessOpTest, FloatTestWithDegeneratedBox) {
  BaseDetectionPostprocessOpModel m(
      {TensorType_FLOAT32, {1, 2, 4}}, {TensorType_FLOAT32, {1, 2, 3}},
      {TensorType_FLOAT32, {2, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}});

  // two boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0, 0.0, 0.0,  // box #1
      0.0, 0.0, 0.0, 0.0,  // box #2
  });
  // class scores - two classes with background
  m.SetInput2<float>({
      /*background*/ 0., /*class 0*/ .9, /*class 1*/ .8,  // box #1
      /*background*/ 0., /*class 0*/ .2, /*class 1*/ .7   // box #2
  });
  // two anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5, 1.0, 1.0,  // anchor #1
      0.5, 0.5, 0.0, 0.0   // anchor #2 - DEGENERATED!
  });
  // Same boxes in box-corner encoding:
  // { 0.0, 0.0, 1.0, 1.0,
  //   0.5, 0.5, 0.5, 0.5} // DEGENERATED!
  // NOTE: this is used instead of `m.Invoke()` to make sure the entire test
  // gets aborted if an error occurs (which does not happen when e.g. ASSERT_EQ
  // is used in such a helper function).
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  const int num_detections = static_cast<int>(m.GetOutput4<float>()[0]);
  EXPECT_EQ(num_detections, 2);
  // detection_boxes
  std::vector<int> output_shape1 = m.GetOutputShape1();
  // NOTE: there are up to 3 detected boxes as per `max_detections` and
  // `max_classes_per_detection` parameters. But since the actual number of
  // detections is 2 (see above) only the top-2 results are tested
  // here and below.
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  std::vector<float> detection_boxes = m.GetOutput1<float>();
  detection_boxes.resize(num_detections * 4);
  EXPECT_THAT(detection_boxes,
              ElementsAreArray(ArrayFloatNear({0.0, 0.0, 1.0, 1.0,   // box #1
                                               0.5, 0.5, 0.5, 0.5},  // box #2
                                              1e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  std::vector<float> detection_classes = m.GetOutput2<float>();
  detection_classes.resize(num_detections);
  EXPECT_THAT(detection_classes,
              ElementsAreArray(ArrayFloatNear({0, 1}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  std::vector<float> detection_scores = m.GetOutput3<float>();
  detection_scores.resize(num_detections);
  EXPECT_THAT(detection_scores,
              ElementsAreArray(ArrayFloatNear({0.9, 0.7}, 1e-4)));
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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

TEST(DetectionPostprocessOpTest, MaxClass2Test) {
  BaseDetectionPostprocessOpModel m(
      {TensorType_FLOAT32, {1, 6, 4}}, {TensorType_FLOAT32, {1, 6, 3}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, /*max_classes_per_detection=*/2);

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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 6, 4));
  EXPECT_THAT(m.GetOutput1<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {0.0, 10.0,  1.0, 11.0,  0.0, 10.0,  1.0, 11.0,
                   0.0, 0.0,   1.0, 1.0,   0.0, 0.0,   1.0, 1.0,
                   0.0, 100.0, 1.0, 101.0, 0.0, 100.0, 1.0, 101.0},
                  1e-1)));
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 6));
  EXPECT_THAT(m.GetOutput2<float>(),
              ElementsAreArray(ArrayFloatNear({1, 0, 0, 1, 0, 1}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 6));
  EXPECT_THAT(
      m.GetOutput3<float>(),
      ElementsAreArray(ArrayFloatNear({0.95, .93, 0.9, 0.8, 0.3, 0.2}, 1e-4)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-4)));
}

// Tests for scenarios where we set use_regular_nms flag
class DetectionPostprocessOpModelwithRegularNMS : public SingleOpModel {
 public:
  DetectionPostprocessOpModelwithRegularNMS(
      const TensorData& input1, const TensorData& input2,
      const TensorData& input3, const TensorData& output1,
      const TensorData& output2, const TensorData& output3,
      const TensorData& output4, bool use_regular_nms, int num_threads = 1) {
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
    BuildInterpreter({GetShape(input1_), GetShape(input2_), GetShape(input3_)},
                     num_threads,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true);
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-4)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-4)));
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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

class DetectionPostprocessOpRegularTest
    : public ::testing::TestWithParam<::testing::tuple<TensorType, int>> {
 protected:
  DetectionPostprocessOpRegularTest()
      : tensor_type_(::testing::get<0>(GetParam())),
        num_threads_(::testing::get<1>(GetParam())) {}

  TensorType tensor_type_;
  int num_threads_;
};

INSTANTIATE_TEST_SUITE_P(
    DetectionPostprocessOpRegularTest, DetectionPostprocessOpRegularTest,
    ::testing::Combine(::testing::Values(TensorType_FLOAT32, TensorType_UINT8),
                       ::testing::Values(1, 2)));

TEST_P(DetectionPostprocessOpRegularTest, RegularNMS) {
  TensorData input1, input2, input3;
  if (tensor_type_ == TensorType_UINT8) {
    input1 = {tensor_type_, {1, 6, 4}, -1.0, 1.0};
    input2 = {tensor_type_, {1, 6, 3}, 0.0, 1.0};
    input3 = {tensor_type_, {6, 4}, 0.0, 100.5};
  } else {
    input1 = {tensor_type_, {1, 6, 4}};
    input2 = {tensor_type_, {1, 6, 3}};
    input3 = {tensor_type_, {6, 4}};
  }
  DetectionPostprocessOpModelwithRegularNMS m(
      input1, input2, input3, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, true, num_threads_);
  auto inputs1 = {
      0.0f, 0.0f,  0.0f, 0.0f,  // box #1
      0.0f, 1.0f,  0.0f, 0.0f,  // box #2
      0.0f, -1.0f, 0.0f, 0.0f,  // box #3
      0.0f, 0.0f,  0.0f, 0.0f,  // box #4
      0.0f, 1.0f,  0.0f, 0.0f,  // box #5
      0.0f, 0.0f,  0.0f, 0.0f   // box #6
  };
  if (tensor_type_ == TensorType_UINT8) {
    m.QuantizeAndPopulate<uint8_t>(m.input1(), std::vector<float>{inputs1});
  } else {
    m.SetInput1<float>(inputs1);
  }
  // class scores - two classes with background
  auto inputs2 = {0.f, .9f,  .8f,  0.f, .75f, .72f, 0.f, .6f, .5f,
                  0.f, .93f, .95f, 0.f, .5f,  .4f,  0.f, .3f, .2f};
  if (tensor_type_ == TensorType_UINT8) {
    m.QuantizeAndPopulate<uint8_t>(m.input2(), std::vector<float>{inputs2});
  } else {
    m.SetInput2<float>(inputs2);
  }
  // six anchors in center-size encoding
  auto inputs3 = {
      0.5f, 0.5f,   1.0f, 1.0f,  // anchor #1
      0.5f, 0.5f,   1.0f, 1.0f,  // anchor #2
      0.5f, 0.5f,   1.0f, 1.0f,  // anchor #3
      0.5f, 10.5f,  1.0f, 1.0f,  // anchor #4
      0.5f, 10.5f,  1.0f, 1.0f,  // anchor #5
      0.5f, 100.5f, 1.0f, 1.0f   // anchor #6
  };
  if (tensor_type_ == TensorType_UINT8) {
    m.QuantizeAndPopulate<uint8_t>(m.input3(), std::vector<float>{inputs3});
  } else {
    m.SetInput3<float>(inputs3);
  }
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  // detection_boxes
  // in center-size
  std::vector<int> output_shape1 = m.GetOutputShape1();
  EXPECT_THAT(output_shape1, ElementsAre(1, 3, 4));
  if (tensor_type_ == TensorType_UINT8) {
    EXPECT_THAT(
        m.GetOutput1<float>(),
        ElementsAreArray(ArrayFloatNear(
            {0.0, 10.0, 1.0, 11.0, 0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 0.0, 0.0},
            3e-1)));
  } else {
    EXPECT_THAT(
        m.GetOutput1<float>(),
        ElementsAreArray(ArrayFloatNear(
            {0.0, 10.0, 1.0, 11.0, 0.0, 10.0, 1.0, 11.0, 0.0, 0.0, 0.0, 0.0},
            3e-4)));
  }
  // detection_classes
  std::vector<int> output_shape2 = m.GetOutputShape2();
  EXPECT_THAT(output_shape2, ElementsAre(1, 3));
  if (tensor_type_ == TensorType_UINT8) {
    EXPECT_THAT(m.GetOutput2<float>(),
                ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-1)));
  } else {
    EXPECT_THAT(m.GetOutput2<float>(),
                ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-4)));
  }
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  if (tensor_type_ == TensorType_UINT8) {
    EXPECT_THAT(m.GetOutput3<float>(),
                ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.0}, 1e-1)));
  } else {
    EXPECT_THAT(m.GetOutput3<float>(),
                ElementsAreArray(ArrayFloatNear({0.95, 0.93, 0.0}, 1e-4)));
  }
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  if (tensor_type_ == TensorType_UINT8) {
    EXPECT_THAT(m.GetOutput4<float>(),
                ElementsAreArray(ArrayFloatNear({2.0}, 1e-1)));
  } else {
    EXPECT_THAT(m.GetOutput4<float>(),
                ElementsAreArray(ArrayFloatNear({2.0}, 1e-4)));
  }
}

TEST(DetectionPostprocessOpTest, FloatTestwithNoBackgroundClassAndNoKeypoints) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_FLOAT32, {1, 6, 4}}, {TensorType_FLOAT32, {1, 6, 2}},
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
  // class scores - two classes without background
  m.SetInput2<float>({.9, .8, .75, .72, .6, .5, .93, .95, .5, .4, .3, .2});
  // six anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  //  anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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

TEST(DetectionPostprocessOpTest, FloatTestwithBackgroundClassAndKeypoints) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_FLOAT32, {1, 6, 5}}, {TensorType_FLOAT32, {1, 6, 3}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);

  // six boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #1
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #2
      0.0, -1.0, 0.0, 0.0, 1.0,  // box #3
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #4
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #5
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #6
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

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-4)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-4)));
}

TEST(DetectionPostprocessOpTest,
     QuantizedTestwithNoBackgroundClassAndKeypoints) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_UINT8, {1, 6, 5}, -1.0, 1.0},
      {TensorType_UINT8, {1, 6, 2}, 0.0, 1.0},
      {TensorType_UINT8, {6, 4}, 0.0, 100.5}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);
  // six boxes in center-size encoding
  std::vector<std::vector<float>> inputs1 = {{
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #1
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #2
      0.0, -1.0, 0.0, 0.0, 1.0,  // box #3
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #4
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #5
      0.0, 0.0,  0.0, 0.0, 1.0   // box #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[0]);
  // class scores - two classes with background
  std::vector<std::vector<float>> inputs2 = {
      {.9, .8, .75, .72, .6, .5, .93, .95, .5, .4, .3, .2}};
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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

TEST(DetectionPostprocessOpTest, FloatTestwithNoBackgroundClassAndKeypoints) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_FLOAT32, {1, 6, 5}}, {TensorType_FLOAT32, {1, 6, 2}},
      {TensorType_FLOAT32, {6, 4}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);

  // six boxes in center-size encoding
  m.SetInput1<float>({
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #1
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #2
      0.0, -1.0, 0.0, 0.0, 1.0,  // box #3
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #4
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #5
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #6
  });
  // class scores - two classes with no background
  m.SetInput2<float>({.9, .8, .75, .72, .6, .5, .93, .95, .5, .4, .3, .2});
  // six anchors in center-size encoding
  m.SetInput3<float>({
      0.5, 0.5,   1.0, 1.0,  // anchor #1
      0.5, 0.5,   1.0, 1.0,  // anchor #2
      0.5, 0.5,   1.0, 1.0,  // anchor #3
      0.5, 10.5,  1.0, 1.0,  // anchor #4
      0.5, 10.5,  1.0, 1.0,  //  anchor #5
      0.5, 100.5, 1.0, 1.0   // anchor #6
  });

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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
              ElementsAreArray(ArrayFloatNear({1, 0, 0}, 1e-4)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(ArrayFloatNear({0.95, 0.9, 0.3}, 1e-4)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-4)));
}

TEST(DetectionPostprocessOpTest,
     QuantizedTestwithNoBackgroundClassAndKeypointsStableSort) {
  DetectionPostprocessOpModelwithRegularNMS m(
      {TensorType_UINT8, {1, 6, 5}, -1.0, 1.0},
      {TensorType_UINT8, {1, 6, 2}, 0.0, 1.0},
      {TensorType_UINT8, {6, 4}, 0.0, 100.5}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, false);
  // six boxes in center-size encoding
  std::vector<std::vector<float>> inputs1 = {{
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #1
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #2
      0.0, -1.0, 0.0, 0.0, 1.0,  // box #3
      0.0, 0.0,  0.0, 0.0, 1.0,  // box #4
      0.0, 1.0,  0.0, 0.0, 1.0,  // box #5
      0.0, 0.0,  0.0, 0.0, 1.0   // box #6
  }};
  m.QuantizeAndPopulate<uint8_t>(m.input1(), inputs1[0]);
  // class scores - two classes with background
  // inputs2 values taken from ssd mobilenet v1 - a stable sort is required to
  // retain order of equal elements
  std::vector<std::vector<float>> inputs2 = {
      {0.015625, 0.007812, 0.003906, 0.015625, 0.015625, 0.007812, 0.019531,
       0.019531, 0.007812, 0.003906, 0.003906, 0.003906}};
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
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
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
              ElementsAreArray(ArrayFloatNear({0, 0, 0}, 1e-1)));
  // detection_scores
  std::vector<int> output_shape3 = m.GetOutputShape3();
  EXPECT_THAT(output_shape3, ElementsAre(1, 3));
  EXPECT_THAT(m.GetOutput3<float>(),
              ElementsAreArray(
                  ArrayFloatNear({0.0196078, 0.0156863, 0.00392157}, 1e-7)));
  // num_detections
  std::vector<int> output_shape4 = m.GetOutputShape4();
  EXPECT_THAT(output_shape4, ElementsAre(1));
  EXPECT_THAT(m.GetOutput4<float>(),
              ElementsAreArray(ArrayFloatNear({3.0}, 1e-1)));
}
}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
