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

#include <cassert>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using ops::Const;
using ops::CropAndResize;
using ops::ResizeBicubic;
using ops::ResizeBilinear;
using ops::ResizeNearestNeighbor;
using ops::ScaleAndTranslate;

class ImageGradTest : public ::testing::Test {
 protected:
  ImageGradTest() : scope_(Scope::NewRootScope()) {}

  enum OpType { RESIZE_NEAREST, RESIZE_BILINEAR, RESIZE_BICUBIC };

  template <typename T>
  Tensor MakeData(const TensorShape& data_shape) {
    DataType data_type = DataTypeToEnum<T>::v();
    Tensor data(data_type, data_shape);
    auto data_flat = data.flat<T>();
    for (int i = 0; i < data_flat.size(); ++i) {
      data_flat(i) = T(i);
    }
    return data;
  }

  template <typename T>
  void MakeOp(const OpType op_type, const Tensor& x_data, const Input& y_shape,
              const bool align_corners, const bool half_pixel_centers,
              Output* x, Output* y) {
    *x = Const<T>(scope_, x_data);
    switch (op_type) {
      case RESIZE_NEAREST:
        *y = ResizeNearestNeighbor(
            scope_, *x, y_shape,
            ResizeNearestNeighbor::AlignCorners(align_corners));
        return;
      case RESIZE_BILINEAR:
        *y = ResizeBilinear(scope_, *x, y_shape,
                            ResizeBilinear::AlignCorners(align_corners)
                                .HalfPixelCenters(half_pixel_centers));
        return;
      case RESIZE_BICUBIC:
        *y = ResizeBicubic(scope_, *x, y_shape,
                           ResizeBicubic::AlignCorners(align_corners)
                               .HalfPixelCenters(half_pixel_centers));
        return;
    }
    assert(false);
  }

  template <typename T>
  void TestResizedShapeForType(const OpType op_type, const bool align_corners,
                               const bool half_pixel_centers) {
    TensorShape x_shape({1, 2, 2, 1});
    Tensor x_data = MakeData<T>(x_shape);
    Output x, y;
    MakeOp<T>(op_type, x_data, {4, 6}, align_corners, half_pixel_centers, &x,
              &y);

    ClientSession session(scope_);
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session.Run({y}, &outputs));
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].shape(), TensorShape({1, 4, 6, 1}));
  }

  void TestResizedShape(OpType op_type) {
    for (const bool half_pixel_centers : {true, false}) {
      for (const bool align_corners : {true, false}) {
        if (half_pixel_centers && align_corners) {
          continue;
        }
        TestResizedShapeForType<Eigen::half>(op_type, align_corners,
                                             half_pixel_centers);
        TestResizedShapeForType<float>(op_type, align_corners,
                                       half_pixel_centers);
        TestResizedShapeForType<double>(op_type, align_corners,
                                        half_pixel_centers);
      }
    }
  }

  template <typename X_T, typename Y_T, typename JAC_T>
  void TestResizeToSmallerAndAlign(const OpType op_type,
                                   const bool align_corners,
                                   const bool half_pixel_centers) {
    TensorShape x_shape({1, 4, 6, 1});
    Tensor x_data = MakeData<X_T>(x_shape);
    Output x, y;
    MakeOp<X_T>(op_type, x_data, {2, 3}, align_corners, half_pixel_centers, &x,
                &y);
    JAC_T max_error;
    TF_ASSERT_OK((ComputeGradientError<X_T, Y_T, JAC_T>(
        scope_, x, x_data, y, {1, 2, 3, 1}, &max_error)));
    EXPECT_LT(max_error, 1.5e-3);
  }

  template <typename X_T, typename Y_T, typename JAC_T>
  void TestResizeToLargerAndAlign(const OpType op_type,
                                  const bool align_corners,
                                  const bool half_pixel_centers) {
    TensorShape x_shape({1, 2, 3, 1});
    Tensor x_data = MakeData<X_T>(x_shape);
    Output x, y;
    MakeOp<X_T>(op_type, x_data, {4, 6}, align_corners, half_pixel_centers, &x,
                &y);
    JAC_T max_error;
    TF_ASSERT_OK((ComputeGradientError<X_T, Y_T, JAC_T>(
        scope_, x, x_data, y, {1, 4, 6, 1}, &max_error)));
    EXPECT_LT(max_error, 1.5e-3);
  }

  template <typename X_T, typename Y_T, typename JAC_T>
  void TestResize(OpType op_type) {
    for (const bool half_pixel_centers : {true, false}) {
      for (const bool align_corners : {true, false}) {
        // if (!half_pixel_centers) continue;
        if (half_pixel_centers && align_corners) {
          continue;
        }
        TestResizeToSmallerAndAlign<X_T, Y_T, JAC_T>(op_type, align_corners,
                                                     half_pixel_centers);
        TestResizeToLargerAndAlign<X_T, Y_T, JAC_T>(op_type, align_corners,
                                                    half_pixel_centers);
      }
    }
  }

  Scope scope_;
};

TEST_F(ImageGradTest, TestNearestNeighbor) {
  TestResizedShape(RESIZE_NEAREST);
  TestResize<float, float, float>(RESIZE_NEAREST);
  TestResize<double, double, double>(RESIZE_NEAREST);
}

TEST_F(ImageGradTest, TestBilinear) {
  TestResizedShape(RESIZE_BILINEAR);
  TestResize<float, float, float>(RESIZE_BILINEAR);
  // Note that Y_T is always float for this op. We choose
  // double for the jacobian to capture the higher precision
  // between X_T and Y_T.
  TestResize<double, float, double>(RESIZE_BILINEAR);
}

TEST_F(ImageGradTest, TestBicubic) {
  TestResizedShape(RESIZE_BICUBIC);
  TestResize<float, float, float>(RESIZE_BICUBIC);
  // Note that Y_T is always float for this op. We choose
  // double for the jacobian to capture the higher precision
  // between X_T and Y_T.
  TestResize<double, float, double>(RESIZE_BICUBIC);
}

class ScaleAndTranslateGradTest : public ::testing::Test {
 protected:
  ScaleAndTranslateGradTest() : scope_(Scope::NewRootScope()) {}

  template <typename T>
  Tensor MakeData(const TensorShape& data_shape) {
    DataType data_type = DataTypeToEnum<T>::v();
    Tensor data(data_type, data_shape);
    auto data_flat = data.flat<T>();
    for (int i = 0; i < data_flat.size(); ++i) {
      data_flat(i) = T(i);
    }
    return data;
  }

  template <typename T>
  void MakeOp(const Tensor& x_data, const Input& y_shape, Input scale,
              Input translation, const string& kernel_type, bool antialias,
              Output* x, Output* y) {
    *x = Const<T>(scope_, x_data);
    *y = ScaleAndTranslate(scope_, *x, y_shape, scale, translation,
                           ScaleAndTranslate::KernelType(kernel_type)
                               .Antialias(antialias)
                               .Antialias(antialias));
    TF_ASSERT_OK(scope_.status());
  }

  template <typename X_T, typename Y_T, typename JAC_T>
  void TestScaleAndTranslate(const TensorShape x_shape, const int out_height,
                             const int out_width, Input scale,
                             Input translation, const string& kernel_type,
                             bool antialias) {
    Tensor x_data = MakeData<X_T>(x_shape);
    Output x, y;
    MakeOp<X_T>(x_data, {out_height, out_width}, scale, translation,
                kernel_type, antialias, &x, &y);
    JAC_T max_error;
    TF_ASSERT_OK((ComputeGradientError<X_T, Y_T, JAC_T>(
        scope_, x, x_data, y, {1, out_height, out_width, 1}, &max_error)));
    EXPECT_LT(max_error, 2e-3);
  }

  const std::vector<Input> kScales = {Input{1.0f, 1.0f}, Input{0.37f, 0.47f},
                                      Input{2.1f, 2.1f}};
  const std::vector<Input> kTranslations = {
      Input{0.0f, 0.0f}, Input{3.14f, 1.19f}, Input{2.1f, 3.1f},
      Input{100.0f, 200.0f}};
  Scope scope_;
};

TEST_F(ScaleAndTranslateGradTest, TestGrads) {
  const std::vector<std::string> kKernelTypes = {"lanczos1", "lanczos3",
                                                 "lanczos5", "gaussian"};
  constexpr int kOutHeight = 4;
  constexpr int kOutWidth = 6;

  const TensorShape kXShape = TensorShape({1, 2, 3, 1});
  for (const Input scale : kScales) {
    for (const Input translation : kTranslations) {
      for (const std::string& kernel_type : kKernelTypes) {
        TestScaleAndTranslate<float, float, float>(
            kXShape, kOutHeight, kOutWidth, scale, translation, kernel_type,
            true);
      }
    }
  }
}

TEST_F(ScaleAndTranslateGradTest, TestGradsWithoutAntialias) {
  constexpr int kOutHeight = 4;
  constexpr int kOutWidth = 6;

  const TensorShape kXShape = TensorShape({1, 2, 3, 1});
  for (const Input scale : kScales) {
    for (const Input translation : kTranslations) {
      TestScaleAndTranslate<float, float, float>(kXShape, kOutHeight, kOutWidth,
                                                 scale, translation, "lanczos3",
                                                 false);
    }
  }
}

TEST_F(ScaleAndTranslateGradTest, TestGradsWithSameShape) {
  const std::vector<std::string> kKernelTypes = {"lanczos3", "gaussian"};

  constexpr int kOutHeight = 2;
  constexpr int kOutWidth = 3;

  const TensorShape kXShape = TensorShape({1, 2, 3, 1});
  for (const Input scale : kScales) {
    for (const Input translation : kTranslations) {
      for (const std::string& kernel_type : kKernelTypes) {
        TestScaleAndTranslate<float, float, float>(
            kXShape, kOutHeight, kOutWidth, scale, translation, kernel_type,
            true);
      }
    }
  }
}

TEST_F(ScaleAndTranslateGradTest, TestGradsWithSmallerShape) {
  const std::vector<std::string> kKernelTypes = {"lanczos3", "gaussian"};
  constexpr int kOutHeight = 2;
  constexpr int kOutWidth = 3;

  const TensorShape kXShape = TensorShape({1, 4, 6, 1});
  for (const Input scale : kScales) {
    for (const Input translation : kTranslations) {
      for (const std::string& kernel_type : kKernelTypes) {
        TestScaleAndTranslate<float, float, float>(
            kXShape, kOutHeight, kOutWidth, scale, translation, kernel_type,
            true);
      }
    }
  }
}

class CropAndResizeGradTest : public ::testing::Test {
 protected:
  CropAndResizeGradTest() : scope_(Scope::NewRootScope()) {}

  template <typename T>
  Tensor MakeData(const TensorShape& data_shape) {
    DataType data_type = DataTypeToEnum<T>::v();
    Tensor data(data_type, data_shape);
    auto data_flat = data.flat<T>();
    for (int i = 0; i < data_flat.size(); ++i) {
      data_flat(i) = T(i);
    }
    return data;
  }

  template <typename T>
  void MakeOp(const Tensor& x_data, const Input& boxes, const Input& box_ind,
              const Input& crop_size, Output* x, Output* y) {
    *x = Const<T>(scope_, x_data);
    *y = CropAndResize(scope_, *x, boxes, box_ind, crop_size,
                       CropAndResize::Method("bilinear"));
    TF_ASSERT_OK(scope_.status());
  }

  template <typename X_T, typename Y_T, typename JAC_T>
  void TestCropAndResize() {
    TensorShape x_shape({1, 4, 2, 1});
    Tensor x_data = MakeData<X_T>(x_shape);
    TensorShape box_shape({1, 4});
    Tensor boxes = MakeData<X_T>(box_shape);
    Output x, y;
    MakeOp<X_T>(x_data, boxes, {0}, {1, 1}, &x, &y);
    JAC_T max_error;
    TF_ASSERT_OK((ComputeGradientError<X_T, Y_T, JAC_T>(
        scope_, x, x_data, y, {1, 1, 1, 1}, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  Scope scope_;
};

TEST_F(CropAndResizeGradTest, TestCrop) {
  TestCropAndResize<float, float, float>();
}

}  // namespace
}  // namespace tensorflow
