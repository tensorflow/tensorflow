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
#include <cstdarg>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class TransposeConvOpModel : public SingleOpModel {
 public:
  TransposeConvOpModel(std::initializer_list<int> input_shape,
                       std::initializer_list<int> filter_shape, Padding padding,
                       int stride_w, int stride_h) {
    output_shape_ = AddInput(TensorType_INT32);
    filter_ = AddInput(TensorType_FLOAT32);
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_TRANSPOSE_CONV, BuiltinOptions_TransposeConvOptions,
        CreateTransposeConvOptions(builder_, padding, stride_w, stride_h)
            .Union());
    BuildInterpreter({{4}, filter_shape, input_shape});
  }

  int output_shape() { return output_shape_; }
  int filter() { return filter_; }
  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int output_shape_;
  int filter_;
  int input_;
  int output_;
};

// Test case:
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     tf.constant(np.arange(1, 10), shape=[ 3, 3, 1, 1 ], dtype=tf.float32),
//     tf.constant(np.arange(1, 17), shape=[ 1, 4, 4, 1 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
TEST(TransposeConvOpModelTest, SimpleTest) {
  TransposeConvOpModel m({1, 4, 4, 1}, {1, 3, 3, 1}, Padding_SAME, 1, 1);
  m.PopulateTensor<int>(m.output_shape(), {1, 4, 4, 1});
  m.PopulateTensor<float>(m.filter(), {1, 2, 3, 4, 5, 6, 7, 8, 9});
  m.PopulateTensor<float>(
      m.input(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({29, 62, 83, 75, 99, 192, 237, 198, 207, 372,
                                417, 330, 263, 446, 485, 365}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 4, 4, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[18, 1])
TEST(TransposeConvOpModelTest, TwoFiltersTest) {
  TransposeConvOpModel m({1, 4, 4, 2}, {2, 3, 3, 1}, Padding_SAME, 1, 1);
  m.PopulateTensor<int>(m.output_shape(), {1, 4, 4, 1});
  m.PopulateTensor<float>(m.filter(), {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6,
                                       8, 10, 12, 14, 16, 18});
  m.PopulateTensor<float>(
      m.input(),
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({184, 412, 568, 528, 678, 1347, 1689, 1434, 1494,
                                2715, 3057, 2442, 1968, 3352, 3652, 2760}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 1, 2 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 6, 6, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 33), shape=[ 1, 4, 4, 2 ], dtype=tf.float32),
//     [1, 1, 1, 1 ],
//     "VALID")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[1, 18])
TEST(TransposeConvOpModelTest, PaddingValidTest) {
  TransposeConvOpModel m({1, 4, 4, 2}, {2, 3, 3, 1}, Padding_VALID, 1, 1);
  m.PopulateTensor<int>(m.output_shape(), {1, 6, 6, 1});
  m.PopulateTensor<float>(m.filter(), {1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6,
                                       8, 10, 12, 14, 16, 18});
  m.PopulateTensor<float>(
      m.input(),
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({5,    22,   59,   101,  114,  83,   52,   184,  412,
                        568,  528,  344,  237,  678,  1347, 1689, 1434, 879,
                        597,  1494, 2715, 3057, 2442, 1431, 856,  1968, 3352,
                        3652, 2760, 1548, 689,  1534, 2543, 2729, 2010, 1103}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 6, 6, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 10),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 1 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST(TransposeConvOpModelTest, StrideValidTest) {
  TransposeConvOpModel m({1, 2, 2, 1}, {1, 3, 3, 1}, Padding_VALID, 2, 2);
  m.PopulateTensor<int>(m.output_shape(), {1, 5, 5, 1});
  m.PopulateTensor<float>(m.filter(), {1, 2, 3, 4, 5, 6, 7, 8, 9});
  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  5,  4,  6,  4,  5,  14, 10, 12, 10, 14, 36,
                        24, 30, 12, 15, 34, 20, 24, 21, 24, 55, 32, 36}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 5, 5, 1}));
}

// Test case:
// filter = tf.constant(np.arange(1, 19),
//                      shape=[ 3, 3, 2, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 5, 5, 2 ]),
//     filter,
//     tf.constant(np.arange(1, 5), shape=[ 1, 2, 2, 1 ], dtype=tf.float32),
//     [1, 2, 2, 1 ],
//     "VALID")
TEST(TransposeConvOpModelTest, MultiChannelTest) {
  TransposeConvOpModel m({1, 2, 2, 1}, {1, 3, 3, 2}, Padding_VALID, 2, 2);
  m.PopulateTensor<int>(m.output_shape(), {1, 5, 5, 2});
  m.PopulateTensor<float>(m.filter(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18});
  m.PopulateTensor<float>(m.input(), {1, 2, 3, 4});
  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  7,  10,  6,   8,  10, 12, 7,  8,  9,
                        10, 25, 28, 18, 20, 22,  24,  16, 20, 24, 28, 62, 72,
                        42, 48, 54, 60, 21, 24,  27,  30, 61, 68, 36, 40, 44,
                        48, 39, 42, 45, 48, 103, 110, 60, 64, 68, 72}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 5, 5, 2}));
}

// Test case:
// filter = tf.constant(np.random.randint(1, 10, size=9),
//                      shape=[ 3, 3, 1, 1 ],
//                      dtype=tf.float32)
// output = tf.nn.conv2d_backprop_input(
//     tf.constant([ 1, 3, 4, 1 ]),
//     filter,
//     tf.constant([323, 521], shape=[ 1, 1, 2, 1], dtype=tf.float32),
//     [1, 3, 3, 1 ],
//     "SAME")
// And filter value is derived by:
// filter = tf.reshape(tf.transpose(filter, perm=[3, 0, 1, 2]), shape=[-1])
TEST(TransposeConvOpModelTest, AccuracyTest) {
  TransposeConvOpModel m({1, 1, 2, 1}, {1, 3, 3, 1}, Padding_SAME, 3, 3);
  m.PopulateTensor<int>(m.output_shape(), {1, 3, 4, 1});
  m.PopulateTensor<float>(m.filter(), {9, 5, 6, 9, 8, 5, 3, 1, 4});
  m.PopulateTensor<float>(m.input(), {323, 521});
  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {1615., 1938., 4689., 2605., 2584., 1615.,
                                  4689., 4168., 323., 1292., 1563., 521.})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 4, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
