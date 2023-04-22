/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}
}  // namespace

TEST(EigenPoolingTest, Simple) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> result(depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(0), depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(expected, input(d, r + i, c + j, b));
            }
          }
          if (result(d, i, j, b) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(d, i, j, b) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SimpleRowMajor) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows, depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.f);

  // Max pooling using a 4x4 window and a stride of 1.
  const int stride = 1;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(expected, input(b, c + j, r + i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, Cuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected =
                      (std::max)(expected, input(d, p + i, r + j, c + k, b));
                }
              }
            }
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, CuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected =
                      (std::max)(expected, input(b, c + k, r + j, p + i, d));
                }
              }
            }
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, ValidCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected_sum += input(d, p + i, r + j, c + k, b);
                  expected_count++;
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, ValidCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = 2;
  const int output_cols = 3;
  const int output_planes = 4;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected_sum += input(b, c + k, r + j, p + i, d);
                  expected_count++;
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SameCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = input_rows;
  const int output_cols = input_cols;
  const int output_planes = input_planes;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  const int pad_p = output_planes - input_planes + patch_planes - 1;
  const int pad_r = output_rows - input_rows + patch_rows - 1;
  const int pad_c = output_cols - input_cols + patch_cols - 1;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  const int in_p = p + i - dp;
                  const int in_r = r + j - dr;
                  const int in_c = c + k - dc;
                  if (in_p >= 0 && in_p < input_planes && in_r >= 0 &&
                      in_r < input_rows && in_c >= 0 && in_c < input_cols) {
                    expected_sum += input(d, in_p, in_r, in_c, b);
                    expected_count++;
                  }
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, SameCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 4;
  const int patch_cols = 3;
  const int patch_planes = 2;
  const int output_rows = input_rows;
  const int output_cols = input_cols;
  const int output_planes = input_planes;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();
  result = result.constant(-1000.0f);

  // Max pooling using a 4x3x2 window and a stride of 1.
  const int stride = 1;
  result = CuboidAvgPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  const int pad_p = output_planes - input_planes + patch_planes - 1;
  const int pad_r = output_rows - input_rows + patch_rows - 1;
  const int pad_c = output_cols - input_cols + patch_cols - 1;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected_sum = 0.0f;
            int expected_count = 0;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  const int in_p = p + i - dp;
                  const int in_r = r + j - dr;
                  const int in_c = c + k - dc;
                  if (in_p >= 0 && in_p < input_planes && in_r >= 0 &&
                      in_r < input_rows && in_c >= 0 && in_c < input_cols) {
                    expected_sum += input(b, in_c, in_r, in_p, d);
                    expected_count++;
                  }
                }
              }
            }
            const float expected = expected_sum / expected_count;
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " k=" << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, Strided) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> result(depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(0), depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(d, r + stride * i, c + stride * j, b));
            }
          }
          if (result(d, i, j, b) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(d, i, j, b) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(d, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedRowMajor) {
  const int depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows, depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    depth);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3 window and a stride of 2.
  int stride = 2;
  result = SpatialMaxPooling(input, patch_rows, patch_cols, stride, stride,
                             PADDING_VALID);

  EXPECT_EQ(result.dimension(3), depth);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = -10000.f;
          for (int r = 0; r < patch_rows; ++r) {
            for (int c = 0; c < patch_cols; ++c) {
              expected = (std::max)(
                  expected, input(b, c + stride * j, r + stride * i, d));
            }
          }
          if (result(b, j, i, d) != expected) {
            std::cout << "at d=" << d << " b=" << b << " i=" << i << " j=" << j
                      << " " << result(b, j, i, d) << " vs " << expected
                      << std::endl;
          }
          EigenApprox(result(b, j, i, d), expected);
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedCuboid) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_planes = 3;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_planes = 2;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 5> input(channels, input_planes, input_rows, input_cols,
                         num_batches);
  Tensor<float, 5> result(channels, output_planes, output_rows, output_cols,
                          num_batches);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3x3 window and a stride of 2.
  int stride = 2;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), channels);
  EXPECT_EQ(result.dimension(1), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_cols);
  EXPECT_EQ(result.dimension(4), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected = (std::max)(expected,
                                        input(d, p + stride * i, r + stride * j,
                                              c + stride * k, b));
                }
              }
            }
            if (result(d, i, j, k, b) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " " << k << " "
                        << result(d, i, j, k, b) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(d, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenPoolingTest, StridedCuboidRowMajor) {
  const int channels = 10;
  const int input_planes = 5;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int patch_planes = 3;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_planes = 2;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 5, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_planes, channels);
  Tensor<float, 5, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_planes, channels);
  input = input.constant(11.0f) + input.random();
  result.setRandom();

  // Max pooling using a 3x3x3 window and a stride of 2.
  int stride = 2;
  result = CuboidMaxPooling(input, patch_planes, patch_rows, patch_cols, stride,
                            stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(4), channels);
  EXPECT_EQ(result.dimension(3), output_planes);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(0), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int d = 0; d < channels; ++d) {
      for (int i = 0; i < output_planes; ++i) {
        for (int j = 0; j < output_rows; ++j) {
          for (int k = 0; k < output_cols; ++k) {
            float expected = -10000.f;
            for (int p = 0; p < patch_planes; ++p) {
              for (int r = 0; r < patch_rows; ++r) {
                for (int c = 0; c < patch_cols; ++c) {
                  expected = (std::max)(expected,
                                        input(b, c + stride * k, r + stride * j,
                                              p + stride * i, d));
                }
              }
            }
            if (result(b, k, j, i, d) != expected) {
              std::cout << "at d=" << d << " b=" << b << " i=" << i
                        << " j=" << j << " " << k << " "
                        << result(b, k, j, i, d) << " vs " << expected
                        << std::endl;
            }
            EigenApprox(result(b, k, j, i, d), expected);
          }
        }
      }
    }
  }
}

}  // namespace Eigen
