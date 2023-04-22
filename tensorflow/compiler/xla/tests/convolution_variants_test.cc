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

// Tests of convolution variants -- kernel sizes, padding, and strides --
// in small sized data.

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConvolutionVariantsTest : public ClientLibraryTestBase {
 protected:
#if XLA_TEST_BACKEND_GPU
  // XLA:GPU sometimes uses FFT convolution which isn't as precise as spatial
  // convolution. So relax the absolute error threshold.
  ErrorSpec error_spec_ = ErrorSpec(1e-1, 1e-5);
#else
  ErrorSpec error_spec_ = ErrorSpec(1e-4, 1e-2);
#endif
};

XLA_TEST_F(ConvolutionVariantsTest, Minimal) {
  XlaBuilder builder(TestName());

  const Array4D<float> input_array(1, 1, 1, 1, {2});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {3});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  const Array4D<float> expected(1, 1, 1, 1, {6});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, MinimalWithBatch) {
  XlaBuilder builder(TestName());

  const Array4D<float> input_array(5, 1, 1, 1, {1, 2, 3, 4, 5});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {2});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  const Array4D<float> expected(5, 1, 1, 1, {2, 4, 6, 8, 10});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Flat1x1) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(2, 1, 3, 4);
  input_array.FillWithMultiples(1);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {2.3});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(2, 1, 3, 4);
  expected.FillWithMultiples(2.3);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Deep1x1) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 2, 1, 1, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(3, 2, 1, 1, {1, 2, 3, 4, 5, 6});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 3, 1, 1, {12, 34, 56});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2in1x2) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 2, {1, 2});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {12});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2in1x3) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {12, 23});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2in2x2) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 1, {12, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x1in2x2) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 2, 1, {10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {13, 24});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x2in2x2) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 2, 2, {1000, 100, 10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {1234});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2in2x3WithDepthAndBatch) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(
      2, 2, 2, 3, {0, 1, 2, 3, 4, 5,  6,  7,  8,  9,  0, 0,    // plane 0
                   0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 0});  // plane 1
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(
      2, 2, 1, 2, {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(
      2, 2, 2, 2,
      {167, 1278, 3490, 4500, 0.0167, 0.1278, 0.3490, 0.4500,    // plane 0
       334, 2556, 6980, 9000, 0.0334, 0.2556, 0.6980, 0.9000});  // plane 1
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1stride1x2in1x4) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 4, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {10, 30});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1stride1x2in1x5) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 5, {1, 2, 3, 4, 5});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 3, {10, 30, 50});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x3stride1x2in1x4) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 4, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {100, 10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 1, {123});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x3stride1x2in1x5) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 5, {1, 2, 3, 4, 5});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {100, 10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {123, 345});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1stride2x2in3x3) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 1, {10});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {2, 2}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 2, {10, 30, 70, 90});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter3x1in1x1Padded) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 1, {1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 3, {10, 20, 30});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 1, {20});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter5x1in3x1Padded) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 5, {10000, 1000, 100, 10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 3, {123, 1230, 12300});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter3x3in2x2Padded) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 2, 2, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 3, 3,
                                    {10000, 0, 1000,  // row 0
                                     0, 100, 0,       // row 1
                                     10, 0, 1});      // row 2
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 2, 2, {104, 230, 2300, 10400});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1in2x1WithPaddingAndDepth) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 2, 1, 2, {1, 2, 3, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 2, 1, 1, {10, 1});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kSame);

  Array4D<float> expected(1, 1, 1, 2, {13, 24});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x2Stride1x1Input3x3) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 2, 2, {7, 13, 17, 23});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 2, 2, {216, 276, 396, 456});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2Stride1x1Input1x3) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(1, 1, 1, 3, {1, 2, 3});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  const Array4D<float> filter_array(1, 1, 1, 2, {7, 13});
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 1, 1, 2, {33, 53});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x1x8x8Input1x1x8x8) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(64);
  std::iota(input_data.begin(), input_data.end(), 0.0);
  Array4D<float> input_array(1, 1, 8, 8, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(128);
  std::fill(filter_data.begin(), filter_data.begin() + 64, 1.0);
  std::fill(filter_data.begin() + 64, filter_data.begin() + 128, 2.0);
  const Array4D<float> filter_array(2, 1, 8, 8, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 2, 1, 1, {2016, 4032});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x1x1Input16x1x1x1) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(16 * 1 * 1 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(16, 1, 1, 1, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 1.0);
  const Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                      9, 10, 11, 12, 13, 14, 15, 16};
  Array4D<float> expected(16, 1, 1, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x2x2Input16x1x2x2) {
  XlaBuilder builder(TestName());

  constexpr int bs = 16;
  constexpr int kx = 2;
  constexpr int ky = 2;
  Array4D<float> input_array(bs, 1, ky, kx);
  for (int i0 = 0; i0 < bs; ++i0) {
    for (int i2 = 0; i2 < ky; ++i2) {
      for (int i3 = 0; i3 < kx; ++i3) {
        input_array(i0, 0, i2, i3) = i0 + 1;
      }
    }
  }
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * ky * kx);
  std::iota(filter_data.begin(), filter_data.end(), 1.0);
  const Array4D<float> filter_array(1, 1, ky, kx, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data(bs);
  for (int i = 0; i < bs; ++i) {
    expected_data[i] = 10 * (i + 1);
  }
  Array4D<float> expected(bs, 1, 1, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x2x2Input3x1x2x2) {
  XlaBuilder builder(TestName());

  constexpr int kx = 2;
  constexpr int ky = 2;
  constexpr int bs = 3;
  Array4D<float> input_array(bs, 1, ky, kx);
  for (int i0 = 0; i0 < bs; ++i0) {
    for (int i2 = 0; i2 < ky; ++i2) {
      for (int i3 = 0; i3 < kx; ++i3) {
        input_array(i0, 0, i2, i3) = i0 + i2 + i3 + 1;
      }
    }
  }
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * ky * kx);
  std::iota(filter_data.begin(), filter_data.end(), 1.0);
  const Array4D<float> filter_array(1, 1, ky, kx, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      23,
      33,
      43,
  };
  Array4D<float> expected(bs, 1, 1, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x8x8Input16x1x8x8) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(16, 1, 8, 8);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i2 = 0; i2 < 8; ++i2) {
      for (int i3 = 0; i3 < 8; ++i3) {
        input_array(i0, 0, i2, i3) = i0 + i2 + i3 + 1;
      }
    }
  }
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * 8 * 8);
  std::iota(filter_data.begin(), filter_data.end(), 1.0);
  const Array4D<float> filter_array(1, 1, 8, 8, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      19664, 21744, 23824, 25904, 27984, 30064, 32144, 34224,
      36304, 38384, 40464, 42544, 44624, 46704, 48784, 50864,
  };
  Array4D<float> expected(16, 1, 1, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x2x8x8Input1x2x8x8) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.0);
  Array4D<float> input_array(1, 2, 8, 8, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.0);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.0);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.0);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.0);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(1, 2, 1, 1, {14240, 30496});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x2x8x8Input2x2x8x8) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(2 * 2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.0);
  Array4D<float> input_array(2, 2, 8, 8, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.0);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.0);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.0);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.0);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(2, 2, 1, 1, {14240, 30496, 38816, 87840});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter2x2x8x8Input32x2x8x8) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(32 * 2 * 8 * 8);
  std::iota(input_data.begin(), input_data.end(), 0.0);
  Array4D<float> input_array(32, 2, 8, 8, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(2 * 2 * 8 * 8);
  std::fill(filter_data.begin(), filter_data.begin() + filter_data.size() / 4,
            1.0);
  std::fill(filter_data.begin() + filter_data.size() / 4,
            filter_data.begin() + filter_data.size() / 2, 2.0);
  std::fill(filter_data.begin() + filter_data.size() / 2,
            filter_data.begin() + 3 * filter_data.size() / 4, 3.0);
  std::fill(filter_data.begin() + 3 * filter_data.size() / 4, filter_data.end(),
            4.0);
  const Array4D<float> filter_array(2, 2, 8, 8, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  Conv(input, filter, {1, 1}, Padding::kValid);

  std::vector<float> expected_data = {
      14240,       30496,       38816,   87840,   63392,       145184,  87968,
      202528,      112544,      259872,  137120,  317216,      161696,  374560,
      186272,      431904,      210848,  489248,  235424,      546592,  260000,
      603936,      284576,      661280,  309152,  718624,      333728,  775968,
      358304,      833312,      382880,  890656,  407456,      948000,  432032,
      1005344,     456608,      1062688, 481184,  1120032,     505760,  1177376,
      530336,      1.23472e+06, 554912,  1292064, 579488,      1349408, 604064,
      1406752,     628640,      1464096, 653216,  1.52144e+06, 677792,  1578784,
      702368,      1636128,     726944,  1693472, 751520,      1750816, 776096,
      1.80816e+06,
  };
  Array4D<float> expected(32, 2, 1, 1, expected_data);
  // The output elements can be larger than 1e+5, making the absolute error
  // large sometimes. So, we focus on relative errors for this test case.
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter16x16x1x1Input16x16x1x1) {
  XlaBuilder builder(TestName());

  Array4D<float> input_array(16, 16, 1, 1);
  Array4D<float> filter_array(16, 16, 1, 1);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i1 = 0; i1 < 16; ++i1) {
      input_array(i0, i1, 0, 0) = 1000 * i0 + i1;
      filter_array(i0, i1, 0, 0) = 1;
    }
  }

  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  Array4D<float> expected(16, 16, 1, 1);
  for (int i0 = 0; i0 < 16; ++i0) {
    for (int i1 = 0; i1 < 16; ++i1) {
      expected(i0, i1, 0, 0) = 16000 * i0 + 120;
    }
  }

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, FlatRhsDilation) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 4 * 6);
  std::iota(input_data.begin(), input_data.end(), 0.0);
  Array4D<float> input_array(1, 1, 4, 6, input_data);

  Array4D<float> filter_array(1, 1, 2, 3, {1, 10, 100, 2, 20, 200});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{}, /*padding=*/{},
      /*lhs_dilation=*/{}, /*rhs_dilation=*/{2, 2},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 2, 2, {3924, 4257, 5922, 6255});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, FlatLhsDilation1D) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{}, /*padding=*/{},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 8, {10, 2, 20, 3, 30, 4, 40, 5});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, FlatLhsDilation) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 3 * 4);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 3, 4, input_data);

  Array4D<float> filter_array(1, 1, 4, 3,
                              {100, 10, 1,  //
                               200, 20, 2,  //
                               300, 30, 3,  //
                               400, 40, 4});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{2, 1},
      /*padding=*/{{1, 0}, {0, 0}}, /*lhs_dilation=*/{3, 2},
      /*rhs_dilation=*/{}, XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 3, 5,
                          {204, 40, 406, 60, 608,       //
                           1518, 180, 1821, 210, 2124,  //
                           4146, 460, 4651, 510, 5156});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, NegativePaddingOnBothEnds) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-1, -1}},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 2, {23, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, NegativePaddingLowAndPositivePaddingHigh) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-1, 2}},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 5, {23, 34, 45, 50, 0});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, PositivePaddingLowAndNegativePaddingHigh) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneral(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {2, -1}},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  Array4D<float> expected(1, 1, 1, 5, {0, 1, 12, 23, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, PositivePaddingAndDilation) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {3, 2}},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{1, 2},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  // input:
  //   [1, 2, 3, 4, 5] --dilate-> [1, 0, 2, 0, 3, 0, 4, 0, 5]
  //                   ---pad---> [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 0]
  // filter:
  //   [10, 1] --dilate-> [10, 0, 1]
  Array4D<float> expected(1, 1, 1, 12,
                          {0, 1, 0, 12, 0, 23, 0, 34, 0, 45, 0, 50});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}
XLA_TEST_F(ConvolutionVariantsTest, NegativePaddingAndDilation) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 1 * 1 * 5);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 1, 1, 5, input_data);

  Array4D<float> filter_array(1, 1, 1, 2, {10, 1});
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  ConvGeneralDilated(
      /*lhs=*/input, /*rhs=*/filter, /*window_strides=*/{},
      /*padding=*/{{0, 0}, {-3, -2}},
      /*lhs_dilation=*/{1, 2}, /*rhs_dilation=*/{1, 2},
      XlaBuilder::CreateDefaultConvDimensionNumbers());

  // input:
  //   [1, 2, 3, 4, 5] --dilate-> [1, 0, 2, 0, 3, 0, 4, 0, 5]
  //                   ---pad---> [0, 3, 0, 4]
  // filter:
  //   [10, 1] --dilate-> [10, 0, 1]
  Array4D<float> expected(1, 1, 1, 2, {0, 34});
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, RandomData_Input1x1x2x3_Filter2x1x1x2) {
  constexpr int bs = 1;
  constexpr int iz = 1;
  constexpr int oz = 2;
  constexpr int iy = 2;
  constexpr int ix = 3;
  constexpr int ky = 1;
  constexpr int kx = 2;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, RandomData_Input1x16x1x1_Filter1x16x1x1) {
  constexpr int bs = 1;
  constexpr int iz = 16;
  constexpr int oz = 1;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, RandomData_Input16x16x1x1_Filter1x16x1x1) {
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 1;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, RandomData_Input16x16x1x1_Filter16x16x1x1) {
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 16;
  constexpr int iy = 1;
  constexpr int ix = 1;
  constexpr int ky = 1;
  constexpr int kx = 1;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest,
           RandomData_Input16x16x16x16_Filter16x16x16x16) {
  constexpr int bs = 16;
  constexpr int iz = 16;
  constexpr int oz = 16;
  constexpr int iy = 16;
  constexpr int ix = 16;
  constexpr int ky = 16;
  constexpr int kx = 16;
  std::mt19937 rng;
  std::uniform_real_distribution<float> distribution;
  std::vector<float> input_data(bs * iz * iy * ix);
  for (float& f : input_data) {
    f = distribution(rng);
  }
  std::vector<float> kernel_data(oz * iz * ky * kx);
  for (float& f : kernel_data) {
    f = distribution(rng);
  }

  Array4D<float> input_array(bs, iz, iy, ix, input_data);
  Array4D<float> filter_array(oz, iz, ky, kx, kernel_data);

  XlaBuilder builder(TestName());
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);
  Conv(input, filter, {1, 1}, Padding::kValid);

  std::unique_ptr<Array4D<float>> expected = ReferenceUtil::ConvArray4D(
      input_array, filter_array, {1, 1}, Padding::kValid);

  ComputeAndCompareR4<float>(&builder, *expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x2x1x1Input1x2x3x1GeneralPadding) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 2 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 1.0);
  Array4D<float> filter_array(1, 2, 1, 1, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests padding sizes that don't correspond either to SAME or VALID padding.
  ConvGeneral(input, filter, {1, 1}, {{2, 1}, {2, 3}}, dnums);

  std::vector<float> expected_data = {
      0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 0,  0,  0, 0, 0,  //
      0, 2, 5,  8,  3, 0, 0,  //
      0, 8, 14, 17, 6, 0, 0,  //
      0, 0, 0,  0,  0, 0, 0   //
  };
  Array4D<float> expected(1, 5, 7, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x1x1Input1x2x3x1GeneralPadding) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 2.0);
  Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests padding sizes that don't correspond either to SAME or VALID padding.
  ConvGeneral(input, filter, {1, 1}, {{2, 1}, {2, 3}}, dnums);

  std::vector<float> expected_data = {
      0, 0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 0, 0,  0,  0, 0, 0,  //
      0, 0, 2, 4,  6,  0, 0, 0,  //
      0, 0, 8, 10, 12, 0, 0, 0,  //
      0, 0, 0, 0,  0,  0, 0, 0   //
  };
  Array4D<float> expected(1, 5, 8, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x1x1Input1x2x3x1NoPadding) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 1);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 2, 3, 1, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * 1 * 1);
  std::iota(filter_data.begin(), filter_data.end(), 2.0);
  Array4D<float> filter_array(1, 1, 1, 1, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests zero padding sizes. This can use matmul for computation.
  ConvGeneral(input, filter, {1, 1}, {{0, 0}, {0, 0}}, dnums);

  std::vector<float> expected_data = {
      2, 4,  6,  //
      8, 10, 12,
  };
  Array4D<float> expected(1, 2, 3, 1, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, Filter1x1x2x3Input1x2x3x2NoPadding) {
  XlaBuilder builder(TestName());

  std::vector<float> input_data(1 * 2 * 3 * 2);
  std::iota(input_data.begin(), input_data.end(), 1.0);
  Array4D<float> input_array(1, 2, 3, 2, input_data);
  auto input = ConstantR4FromArray4D<float>(&builder, input_array);

  std::vector<float> filter_data(1 * 1 * 2 * 3);
  std::iota(filter_data.begin(), filter_data.end(), 2.0);
  Array4D<float> filter_array(1, 1, 2, 3, filter_data);
  auto filter = ConstantR4FromArray4D<float>(&builder, filter_array);

  ConvolutionDimensionNumbers dnums;
  // NHWC input format.
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);

  // Tensorflow filter shape: [ H, W, inC, outC ]
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);

  // Tests zero padding sizes. This can use matmul for computation.
  ConvGeneral(input, filter, {1, 1}, {{0, 0}, {0, 0}}, dnums);

  std::vector<float> expected_data = {
      12, 15,  18,   //
      26, 33,  40,   //
      40, 51,  62,   //
      54, 69,  84,   //
      68, 87,  106,  //
      82, 105, 128,  //
  };
  Array4D<float> expected(1, 2, 3, 3, expected_data);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

// Regression test for b/32034796.
//
// XLA:GPU fuses
//   Conv([1,2,3], Reverse([5,6]), padding_low=1)
// into
//   BackwardInputConv([1,2,3], [5,6], padding_low=0, padding_high=1)
XLA_TEST_F(ConvolutionVariantsTest,
           BackwardInputLowPaddingLessThanHighPadding) {
  XlaBuilder builder(TestName());

  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{1, 2, 3}));
  auto weights = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 2, /*values=*/{5, 6}));
  auto mirrored_weights = Rev(weights, {2, 3});
  ConvWithGeneralPadding(gradients, mirrored_weights,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {1, 0}});
  ComputeAndCompareR4<float>(&builder, {{{{5, 16, 27}}}}, {}, error_spec_);
}

// XLA:GPU fuses
//   Conv([1], Reverse([1,10,100]), padding_high=3, base_dilation=3)
// into
//   BackwardInputConv([1], [1,10,100], stride=3, padding=(2,1))
XLA_TEST_F(ConvolutionVariantsTest,
           BackwardInputLowPaddingGreaterThanHighPadding) {
  XlaBuilder builder(TestName());

  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 1, /*values=*/{1}));
  auto weights = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{1, 10, 100}));
  auto mirrored_weights = Rev(weights, {2, 3});
  ConvGeneralDilated(gradients, mirrored_weights,
                     /*window_strides=*/{1, 1},
                     /*padding=*/{{0, 0}, {0, 3}},
                     /*lhs_dilation=*/{1, 3}, /*rhs_dilation=*/{},
                     XlaBuilder::CreateDefaultConvDimensionNumbers());
  ComputeAndCompareR4<float>(&builder, {{{{100, 0}}}}, {}, error_spec_);
}

// XLA:GPU fuses
//   Conv([1], Reverse([1,10,100]), padding=(1,1))
// into
//   BackwardInputConv([1], [1,10,100], padding=(1,1))
XLA_TEST_F(ConvolutionVariantsTest, BackwardInputEvenPadding) {
  XlaBuilder builder(TestName());

  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 1, /*values=*/{1}));
  auto weights = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{1, 10, 100}));
  auto mirrored_weights = Rev(weights, {2, 3});
  ConvWithGeneralPadding(gradients, mirrored_weights,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {1, 1}});
  ComputeAndCompareR4<float>(&builder, {{{{10}}}}, {}, error_spec_);
}

// HLO pattern
//   Conv([1,2,3], Reverse([1,10], padding_high=2)
// could be fused to
//   BackwardInputConv([1,2,3], [1,10], padding_low=1, padding_high=-1)
//
// However, XLA:GPU doesn't actually fuse it because PadInsertion doesn't
// support negative padding on backward convolution yet (b/32744257).
XLA_TEST_F(ConvolutionVariantsTest, BackwardInputWithNegativePaddingHigh) {
  XlaBuilder builder(TestName());

  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{1, 2, 3}));
  auto weights = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 2, /*values=*/{1, 10}));
  auto mirrored_weights = Rev(weights, {2, 3});
  ConvWithGeneralPadding(gradients, mirrored_weights,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {0, 2}});

  ComputeAndCompareR4<float>(&builder, {{{{12, 23, 30, 0}}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest,
           BackwardFilterLowPaddingLessThanHighPadding) {
  XlaBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,1,2,3,4,0,0
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 24,130,240
  //
  // This pattern will be fused to backward convolution with padding=(1,2).
  auto activations = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv =
      ConvGeneralDilated(activations, gradients,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {1, 2}},
                         /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
                         XlaBuilder::CreateDefaultConvDimensionNumbers());
  Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{24, 130, 240}}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest,
           BackwardFilterLowPaddingGreaterThanHighPadding) {
  XlaBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,0,1,2,3,4
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 13,24
  //
  // This pattern will be fused to backward convolution with padding=(2,1).
  // Note: both (2,1) and (2,0) are valid padding for the backward convolution
  // because the stride is 2.
  auto activations = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv =
      ConvGeneralDilated(activations, gradients,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {2, 0}},
                         /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
                         XlaBuilder::CreateDefaultConvDimensionNumbers());
  Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{13, 24}}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, BackwardFilterEvenPadding) {
  XlaBuilder builder(TestName());

  // activations:      1,2,3,4  ---pad--> 0,0,1,2,3,4,0
  // gradients:        100,10,1 -dilate-> 100,0,10,0,1
  // weight gradients: 13,24,130
  //
  // This pattern will be fused to backward convolution with padding=(2,2).
  // Note: both (2,1) and (2,2) are valid padding for the backward convolution
  // because the stride is 2. ConvolutionFolding prefers (2,2) because cuDNN
  // supports even padding only -- using (2,1) would need extra effort of
  // canonicalization.
  auto activations = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 4, /*values=*/{1, 2, 3, 4}));
  auto gradients = ConstantR4FromArray4D<float>(
      &builder, Array4D<float>(1, 1, 1, 3, /*values=*/{100, 10, 1}));
  auto forward_conv =
      ConvGeneralDilated(activations, gradients,
                         /*window_strides=*/{1, 1},
                         /*padding=*/{{0, 0}, {2, 1}},
                         /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 2},
                         XlaBuilder::CreateDefaultConvDimensionNumbers());
  Transpose(forward_conv, {0, 1, 2, 3});

  ComputeAndCompareR4<float>(&builder, {{{{13, 24, 130}}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, BackwardInputEvenPadding1D) {
  XlaBuilder builder(TestName());

  auto gradients = ConstantR3FromArray3D<float>(
      &builder, Array3D<float>(1, 1, 1, /*value=*/1));
  auto weights =
      ConstantR3FromArray3D<float>(&builder, Array3D<float>({{{1, 10, 100}}}));
  auto mirrored_weights = Rev(weights, {2});
  ConvWithGeneralPadding(gradients, mirrored_weights,
                         /*window_strides=*/{1},
                         /*padding=*/{{1, 1}});
  ComputeAndCompareR3<float>(&builder, {{{10}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, BackwardFilterEvenPadding1D) {
  XlaBuilder builder(TestName());

  auto activations =
      ConstantR3FromArray3D<float>(&builder, Array3D<float>({{{1, 2, 3, 4}}}));
  auto gradients =
      ConstantR3FromArray3D<float>(&builder, Array3D<float>({{{100, 10, 1}}}));
  auto forward_conv =
      ConvGeneralDilated(activations, gradients,
                         /*window_strides=*/{1},
                         /*padding=*/{{2, 1}},
                         /*lhs_dilation=*/{}, /*rhs_dilation=*/{2},
                         XlaBuilder::CreateDefaultConvDimensionNumbers(
                             /*num_spatial_dims=*/1));
  Transpose(forward_conv, {0, 1, 2});

  ComputeAndCompareR3<float>(&builder, {{{13, 24, 130}}}, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, BackwardInputEvenPadding3D) {
  XlaBuilder builder(TestName());

  auto gradients_flat = LiteralUtil::CreateR1<float>({1});
  auto gradients_literal =
      gradients_flat.Reshape({1, 1, 1, 1, 1}).ConsumeValueOrDie();
  auto gradients = ConstantLiteral(&builder, gradients_literal);

  auto weights_flat = LiteralUtil::CreateR1<float>({1, 10, 100});
  auto weights_literal =
      weights_flat.Reshape({1, 1, 1, 1, 3}).ConsumeValueOrDie();
  auto weights = ConstantLiteral(&builder, weights_literal);

  auto expected_flat = LiteralUtil::CreateR1<float>({10});
  auto expected_literal =
      expected_flat.Reshape({1, 1, 1, 1, 1}).ConsumeValueOrDie();

  auto mirrored_weights = Rev(weights, {2, 3, 4});
  ConvWithGeneralPadding(gradients, mirrored_weights,
                         /*window_strides=*/{1, 1, 1},
                         /*padding=*/{{0, 0}, {0, 0}, {1, 1}});
  ComputeAndCompareLiteral(&builder, expected_literal, {}, error_spec_);
}

XLA_TEST_F(ConvolutionVariantsTest, BackwardFilterEvenPadding3D) {
  XlaBuilder builder(TestName());

  auto activations_flat = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  auto activations_literal =
      activations_flat.Reshape({1, 1, 1, 1, 4}).ConsumeValueOrDie();
  auto activations = ConstantLiteral(&builder, activations_literal);

  auto gradients_flat = LiteralUtil::CreateR1<float>({100, 10, 1});
  auto gradients_literal =
      gradients_flat.Reshape({1, 1, 1, 1, 3}).ConsumeValueOrDie();
  auto gradients = ConstantLiteral(&builder, gradients_literal);

  auto expected_flat = LiteralUtil::CreateR1<float>({13, 24, 130});
  auto expected_literal =
      expected_flat.Reshape({1, 1, 1, 1, 3}).ConsumeValueOrDie();

  auto forward_conv =
      ConvGeneralDilated(activations, gradients,
                         /*window_strides=*/{1, 1, 1},
                         /*padding=*/{{0, 0}, {0, 0}, {2, 1}},
                         /*lhs_dilation=*/{}, /*rhs_dilation=*/{1, 1, 2},
                         XlaBuilder::CreateDefaultConvDimensionNumbers(
                             /*num_spatial_dims=*/3));
  Transpose(forward_conv, {0, 1, 2, 3, 4});
  ComputeAndCompareLiteral(&builder, expected_literal, {}, error_spec_);
}

}  // namespace
}  // namespace xla
