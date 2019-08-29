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

#include "tensorflow/core/kernels/eigen_spatial_convolutions.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace Eigen {

#define EigenApprox(a, b) \
  { ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3); }
static int ceil_div(int a, int b) { return (a + b - 1) / b; }

TEST(EigenSpatialConvolutionsTest, Simple) {
  const int input_depth = 7;
  const int input_rows = 4;
  const int input_cols = 5;
  const int output_depth = 10;
  const int patch_rows = 3;
  const int patch_cols = 4;
  const int output_rows = input_rows;
  const int output_cols = input_cols;

  Tensor<float, 3> input(input_depth, input_rows, input_cols);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 3> result(output_depth, output_rows, output_cols);

  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = SpatialConvolution(input, kernel);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);

  for (int od = 0; od < output_depth; ++od) {
    for (int i = 0; i < output_rows; ++i) {
      for (int j = 0; j < output_cols; ++j) {
        float expected = 0.0f;
        for (int c = 0; c < patch_cols; ++c) {
          for (int r = 0; r < patch_rows; ++r) {
            for (int id = 0; id < input_depth; ++id) {
              if (r - 1 + i >= 0 && c - 1 + j >= 0 && r - 1 + i < output_rows &&
                  c - 1 + j < output_cols) {
                expected +=
                    input(id, r - 1 + i, c - 1 + j) * kernel(od, id, r, c);
              }
            }
          }
        }
        EigenApprox(result(od, i, j), expected);
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, SimpleRowMajor) {
  const int input_depth = 7;
  const int input_rows = 4;
  const int input_cols = 5;
  const int output_depth = 10;
  const int patch_rows = 3;
  const int patch_cols = 4;
  const int output_rows = input_rows;
  const int output_cols = input_cols;

  Tensor<float, 3, RowMajor> input(input_cols, input_rows, input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth,
                                    output_depth);
  Tensor<float, 3, RowMajor> result(output_cols, output_rows, output_depth);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = SpatialConvolution(input, kernel);

  EXPECT_EQ(result.dimension(0), output_cols);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_depth);

  for (int od = 0; od < output_depth; ++od) {
    for (int i = 0; i < output_rows; ++i) {
      for (int j = 0; j < output_cols; ++j) {
        float expected = 0.0f;
        for (int c = 0; c < patch_cols; ++c) {
          for (int r = 0; r < patch_rows; ++r) {
            for (int id = 0; id < input_depth; ++id) {
              if (r - 1 + i >= 0 && c - 1 + j >= 0 && r - 1 + i < output_rows &&
                  c - 1 + j < output_cols) {
                expected +=
                    input(c - 1 + j, r - 1 + i, id) * kernel(c, r, id, od);
              }
            }
          }
        }
        EigenApprox(result(j, i, od), expected);
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, BatchedSpatialConvolution) {
  Tensor<float, 4> input(10, 5, 5, 13);
  Tensor<float, 4> kernel(7, 10, 3, 3);
  Tensor<float, 4> result(7, 5, 5, 13);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = SpatialConvolution(input, kernel);

  EXPECT_EQ(result.dimension(0), 7);
  EXPECT_EQ(result.dimension(1), 5);
  EXPECT_EQ(result.dimension(2), 5);

  for (int b = 0; b < 13; ++b) {
    for (int od = 0; od < 7; ++od) {
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
              for (int id = 0; id < 10; ++id) {
                if (r - 1 + i >= 0 && c - 1 + j >= 0 && r - 1 + i < 5 &&
                    c - 1 + j < 5) {
                  expected +=
                      input(id, r - 1 + i, c - 1 + j, b) * kernel(od, id, r, c);
                }
              }
            }
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, BatchedSpatialConvolutionRowMajor) {
  Tensor<float, 4, RowMajor> input(13, 5, 5, 10);
  Tensor<float, 4, RowMajor> kernel(3, 3, 10, 7);
  Tensor<float, 4, RowMajor> result(13, 5, 5, 7);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = SpatialConvolution(input, kernel);

  EXPECT_EQ(result.dimension(1), 5);
  EXPECT_EQ(result.dimension(2), 5);
  EXPECT_EQ(result.dimension(3), 7);

  for (int b = 0; b < 13; ++b) {
    for (int od = 0; od < 7; ++od) {
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
              for (int id = 0; id < 10; ++id) {
                if (r - 1 + i >= 0 && c - 1 + j >= 0 && r - 1 + i < 5 &&
                    c - 1 + j < 5) {
                  expected +=
                      input(b, c - 1 + j, r - 1 + i, id) * kernel(c, r, id, od);
                }
              }
            }
          }
          EigenApprox(result(b, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, ValidSpatialConvolution) {
  const int input_depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = input_rows - patch_rows + 1;
  const int output_cols = input_cols - patch_cols + 1;

  Tensor<float, 4> input(input_depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 4> result(output_depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 4x4 kernel, valid padding, and a stride
  // of 1.
  const int stride = 1;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(id, r + i, c + j, b) * kernel(od, id, r, c);
              }
            }
          }
          if (result(od, i, j, b) != expected) {
            std::cout << "at od=" << od << " b=" << b << " i=" << i
                      << " j=" << j << " " << result(od, i, j, b) << " vs "
                      << expected << std::endl;
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, ValidSpatialConvolutionUnequalStrides) {
  const int input_depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 4;
  const int patch_cols = 4;

  const int row_stride = 1;
  const int col_stride = 2;
  const int output_rows = 2;
  const int output_cols = 1;

  Tensor<float, 4> input(input_depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 4> result(output_depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 4x4 kernel, valid padding, and a stride
  // of 1.
  result =
      SpatialConvolution(input, kernel, row_stride, col_stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);
  if (true) return;

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected +=
                    input(id, r + row_stride * i, c + col_stride * j, b) *
                    kernel(od, id, r, c);
              }
            }
          }
          if (result(od, i, j, b) != expected) {
            std::cout << "at od=" << od << " b=" << b << " i=" << i
                      << " j=" << j << " " << result(od, i, j, b) << " vs "
                      << expected << std::endl;
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, ValidSpatialConvolutionRowMajor) {
  const int input_depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 4;
  const int patch_cols = 4;
  const int output_rows = input_rows - patch_rows + 1;
  const int output_cols = input_cols - patch_cols + 1;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth,
                                    output_depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_depth);

  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 4x4 kernel, valid padding, and a stride
  // of 1.
  const int stride = 1;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), num_batches);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_depth);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_rows; ++c) {
            for (int r = 0; r < patch_cols; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(b, c + j, r + i, id) * kernel(c, r, id, od);
              }
            }
          }
          if (result(b, j, i, od) != expected) {
            std::cout << "at od=" << od << " b=" << b << " i=" << i
                      << " j=" << j << " " << result(b, j, i, od) << " vs "
                      << expected << std::endl;
          }
          EigenApprox(result(b, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedSpatialConvolution) {
  const int input_depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(input_depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 4> result(output_depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 3x3 kernel, valid padding, and a stride
  // of 2.
  int stride = 2;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(id, r + stride * i, c + stride * j, b) *
                            kernel(od, id, r, c);
              }
            }
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, KernelSmallerThanStride) {
  const int input_depth = 2;
  const int input_rows = 3;
  const int input_cols = 3;
  const int num_batches = 5;
  const int output_depth = 6;
  const int patch_rows = 1;
  const int patch_cols = 1;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4> input(input_depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 4> result(output_depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 1x1 kernel, valid padding, and a stride
  // of 2.
  int stride = 2;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(id, r + stride * i, c + stride * j, b) *
                            kernel(od, id, r, c);
              }
            }
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedSpatialConvolutionRowMajor) {
  const int input_depth = 10;
  const int input_rows = 5;
  const int input_cols = 5;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 2;
  const int output_cols = 2;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth,
                                    output_depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_depth);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 3x3 kernel, valid padding, and a stride
  // of 2.
  int stride = 2;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), num_batches);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_depth);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(b, c + stride * j, r + stride * i, id) *
                            kernel(c, r, id, od);
              }
            }
          }
          EigenApprox(result(b, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, AtrousSpatial) {
  const int input_depth = 10;
  const int input_rows = 7;
  const int input_cols = 7;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 3;
  const int output_cols = 3;

  Tensor<float, 4> input(input_depth, input_rows, input_cols, num_batches);
  Tensor<float, 4> kernel(output_depth, input_depth, patch_rows, patch_cols);
  Tensor<float, 4> result(output_depth, output_rows, output_cols, num_batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 3x3 kernel, valid padding
  // output (standard) stride 1, and input (atrous) stride of 2.
  int stride = 1;
  int in_stride = 2;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID,
                              in_stride, in_stride);

  EXPECT_EQ(result.dimension(0), output_depth);
  EXPECT_EQ(result.dimension(1), output_rows);
  EXPECT_EQ(result.dimension(2), output_cols);
  EXPECT_EQ(result.dimension(3), num_batches);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(id, in_stride * r + stride * i,
                                  in_stride * c + stride * j, b) *
                            kernel(od, id, r, c);
              }
            }
          }
          EigenApprox(result(od, i, j, b), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, AtrousSpatialRowMajor) {
  const int input_depth = 10;
  const int input_rows = 7;
  const int input_cols = 7;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 3;
  const int output_cols = 3;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth,
                                    output_depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_depth);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 3x3 kernel, valid padding
  // output (standard) stride 1, and input (atrous) stride of 2.
  int stride = 1;
  int in_stride = 2;
  result = SpatialConvolution(input, kernel, stride, stride, PADDING_VALID,
                              in_stride, in_stride);

  EXPECT_EQ(result.dimension(0), num_batches);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_depth);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(b, in_stride * c + stride * j,
                                  in_stride * r + stride * i, id) *
                            kernel(c, r, id, od);
              }
            }
          }
          EigenApprox(result(b, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, AtrousSpatialRowMajorUnequalStrides) {
  const int input_depth = 10;
  const int input_rows = 7;
  const int input_cols = 7;
  const int num_batches = 13;
  const int output_depth = 7;
  const int patch_rows = 3;
  const int patch_cols = 3;
  const int output_rows = 1;
  const int output_cols = 3;

  Tensor<float, 4, RowMajor> input(num_batches, input_cols, input_rows,
                                   input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth,
                                    output_depth);
  Tensor<float, 4, RowMajor> result(num_batches, output_cols, output_rows,
                                    output_depth);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  // Apply a spatial convolution using a 3x3 kernel, valid padding
  // output (standard) stride 1, and input (atrous) stride of 2.
  int row_stride = 1;
  int col_stride = 2;
  int row_in_stride = 3;
  int col_in_stride = 1;
  result = SpatialConvolution(input, kernel, row_stride, col_stride,
                              PADDING_VALID, row_in_stride, col_in_stride);

  EXPECT_EQ(result.dimension(0), num_batches);
  EXPECT_EQ(result.dimension(1), output_cols);
  EXPECT_EQ(result.dimension(2), output_rows);
  EXPECT_EQ(result.dimension(3), output_depth);

  for (int b = 0; b < num_batches; ++b) {
    for (int od = 0; od < output_depth; ++od) {
      for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
          float expected = 0.0f;
          for (int c = 0; c < patch_cols; ++c) {
            for (int r = 0; r < patch_rows; ++r) {
              for (int id = 0; id < input_depth; ++id) {
                expected += input(b, col_in_stride * c + col_stride * j,
                                  row_in_stride * r + row_stride * i, id) *
                            kernel(c, r, id, od);
              }
            }
          }
          EigenApprox(result(b, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, Cuboid) {
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 8;
  const int in_cols = 7;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 4;
  const int kern_height = 4;

  const int out_depth = in_depth;
  const int out_height = in_rows;
  const int out_width = in_cols;

  Tensor<float, 4> input(in_channels, in_depth, in_rows, in_cols);
  Tensor<float, 5> kernel(kern_filters, in_channels, kern_depth, kern_height,
                          kern_width);
  Tensor<float, 4> result(kern_filters, out_depth, out_height, out_width);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel);

  EXPECT_EQ(result.dimension(0), kern_filters);
  EXPECT_EQ(result.dimension(1), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(3), out_width);

  const int off_p = (kern_depth - 1) / 2;
  const int off_r = (kern_height - 1) / 2;
  const int off_c = (kern_width - 1) / 2;

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  if (p - off_p + i >= 0 && r - off_r + j >= 0 &&
                      c - off_c + k >= 0 && p - off_p + i < in_depth &&
                      r - off_r + j < in_rows && c - off_c + k < in_cols) {
                    expected +=
                        input(id, p - off_p + i, r - off_r + j, c - off_c + k) *
                        kernel(od, id, p, r, c);
                  }
                }
              }
            }
          }
          EigenApprox(result(od, i, j, k), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, CuboidRowMajor) {
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 8;
  const int in_cols = 7;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 4;
  const int kern_height = 4;

  const int out_depth = in_depth;
  const int out_height = in_rows;
  const int out_width = in_cols;

  Tensor<float, 4, RowMajor> input(in_cols, in_rows, in_depth, in_channels);
  Tensor<float, 5, RowMajor> kernel(kern_width, kern_height, kern_depth,
                                    in_channels, kern_filters);
  Tensor<float, 4, RowMajor> result(out_width, out_height, out_depth,
                                    kern_filters);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel);

  EXPECT_EQ(result.dimension(3), kern_filters);
  EXPECT_EQ(result.dimension(2), out_depth);
  EXPECT_EQ(result.dimension(1), out_height);
  EXPECT_EQ(result.dimension(0), out_width);

  const int off_p = (kern_depth - 1) / 2;
  const int off_r = (kern_height - 1) / 2;
  const int off_c = (kern_width - 1) / 2;

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  if (p - off_p + i >= 0 && r - off_r + j >= 0 &&
                      c - off_c + k >= 0 && p - off_p + i < in_depth &&
                      r - off_r + j < in_rows && c - off_c + k < in_cols) {
                    expected +=
                        input(c - off_c + k, r - off_r + j, p - off_p + i, id) *
                        kernel(c, r, p, id, od);
                  }
                }
              }
            }
          }
          EigenApprox(result(k, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, ValidCuboid) {
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 5;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int out_depth = 3;
  const int out_height = 3;
  const int out_width = 3;

  Tensor<float, 4> input(in_channels, in_depth, in_rows, in_cols);
  Tensor<float, 5> kernel(kern_filters, in_channels, kern_depth, kern_height,
                          kern_width);
  Tensor<float, 4> result(kern_filters, out_depth, out_height, out_width);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel, 1, 1, 1, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), kern_filters);
  EXPECT_EQ(result.dimension(1), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(3), out_width);

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  expected +=
                      input(id, p + i, r + j, c + k) * kernel(od, id, p, r, c);
                }
              }
            }
          }
          EigenApprox(result(od, i, j, k), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, ValidCuboidRowMajor) {
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 5;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int out_depth = 3;
  const int out_height = 3;
  const int out_width = 3;

  Tensor<float, 4, RowMajor> input(in_cols, in_rows, in_depth, in_channels);
  Tensor<float, 5, RowMajor> kernel(kern_width, kern_height, kern_depth,
                                    in_channels, kern_filters);
  Tensor<float, 4, RowMajor> result(out_width, out_height, out_depth,
                                    kern_filters);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel, 1, 1, 1, PADDING_VALID);

  EXPECT_EQ(result.dimension(3), kern_filters);
  EXPECT_EQ(result.dimension(2), out_depth);
  EXPECT_EQ(result.dimension(1), out_height);
  EXPECT_EQ(result.dimension(0), out_width);

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  expected +=
                      input(c + k, r + j, p + i, id) * kernel(c, r, p, id, od);
                }
              }
            }
          }
          EigenApprox(result(k, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, BatchedCuboid) {
  const int batches = 2;
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 8;
  const int in_cols = 7;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 4;
  const int kern_height = 4;

  const int out_depth = in_depth;
  const int out_height = in_rows;
  const int out_width = in_cols;

  Tensor<float, 5> input(in_channels, in_depth, in_rows, in_cols, batches);
  Tensor<float, 5> kernel(kern_filters, in_channels, kern_depth, kern_height,
                          kern_width);
  Tensor<float, 5> result(kern_filters, out_depth, out_height, out_width,
                          batches);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel);

  EXPECT_EQ(result.dimension(0), kern_filters);
  EXPECT_EQ(result.dimension(1), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(3), out_width);
  EXPECT_EQ(result.dimension(4), batches);

  const int off_p = (kern_depth - 1) / 2;
  const int off_r = (kern_height - 1) / 2;
  const int off_c = (kern_width - 1) / 2;

  for (int b = 0; b < batches; b++) {
    for (int od = 0; od < kern_filters; ++od) {
      for (int i = 0; i < out_depth; ++i) {
        for (int j = 0; j < out_height; ++j) {
          for (int k = 0; k < out_width; ++k) {
            float expected = 0.0f;
            for (int c = 0; c < kern_width; ++c) {
              for (int r = 0; r < kern_height; ++r) {
                for (int p = 0; p < kern_depth; ++p) {
                  for (int id = 0; id < in_channels; ++id) {
                    if (p - off_p + i >= 0 && r - off_r + j >= 0 &&
                        c - off_c + k >= 0 && p - off_p + i < in_depth &&
                        r - off_r + j < in_rows && c - off_c + k < in_cols) {
                      expected += input(id, p - off_p + i, r - off_r + j,
                                        c - off_c + k, b) *
                                  kernel(od, id, p, r, c);
                    }
                  }
                }
              }
            }
            EigenApprox(result(od, i, j, k, b), expected);
          }
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, BatchedCuboidRowMajor) {
  const int batches = 2;
  const int in_channels = 10;
  const int in_depth = 5;
  const int in_rows = 8;
  const int in_cols = 7;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 4;
  const int kern_height = 4;

  const int out_depth = in_depth;
  const int out_height = in_rows;
  const int out_width = in_cols;

  Tensor<float, 5, RowMajor> input(batches, in_cols, in_rows, in_depth,
                                   in_channels);
  Tensor<float, 5, RowMajor> kernel(kern_width, kern_height, kern_depth,
                                    in_channels, kern_filters);
  Tensor<float, 5, RowMajor> result(batches, out_width, out_height, out_depth,
                                    kern_filters);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result = CuboidConvolution(input, kernel);

  EXPECT_EQ(result.dimension(4), kern_filters);
  EXPECT_EQ(result.dimension(3), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(1), out_width);
  EXPECT_EQ(result.dimension(0), batches);

  const int off_p = (kern_depth - 1) / 2;
  const int off_r = (kern_height - 1) / 2;
  const int off_c = (kern_width - 1) / 2;

  for (int b = 0; b < batches; b++) {
    for (int od = 0; od < kern_filters; ++od) {
      for (int i = 0; i < out_depth; ++i) {
        for (int j = 0; j < out_height; ++j) {
          for (int k = 0; k < out_width; ++k) {
            float expected = 0.0f;
            for (int c = 0; c < kern_width; ++c) {
              for (int r = 0; r < kern_height; ++r) {
                for (int p = 0; p < kern_depth; ++p) {
                  for (int id = 0; id < in_channels; ++id) {
                    if (p - off_p + i >= 0 && r - off_r + j >= 0 &&
                        c - off_c + k >= 0 && p - off_p + i < in_depth &&
                        r - off_r + j < in_rows && c - off_c + k < in_cols) {
                      expected += input(b, c - off_c + k, r - off_r + j,
                                        p - off_p + i, id) *
                                  kernel(c, r, p, id, od);
                    }
                  }
                }
              }
            }
            EigenApprox(result(b, k, j, i, od), expected);
          }
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedValidCuboid) {
  const int in_channels = 10;
  const int in_depth = 8;
  const int in_rows = 7;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int out_depth = 3;
  const int out_height = 3;
  const int out_width = 2;

  Tensor<float, 4> input(in_channels, in_depth, in_rows, in_cols);
  Tensor<float, 5> kernel(kern_filters, in_channels, kern_depth, kern_height,
                          kern_width);
  Tensor<float, 4> result(kern_filters, out_depth, out_height, out_width);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  const int stride = 2;
  result =
      CuboidConvolution(input, kernel, stride, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(0), kern_filters);
  EXPECT_EQ(result.dimension(1), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(3), out_width);

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  expected += input(id, p + stride * i, r + stride * j,
                                    c + stride * k) *
                              kernel(od, id, p, r, c);
                }
              }
            }
          }
          EigenApprox(result(od, i, j, k), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedValidCuboidRowMajor) {
  const int in_channels = 10;
  const int in_depth = 8;
  const int in_rows = 7;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int out_depth = 3;
  const int out_height = 3;
  const int out_width = 2;

  Tensor<float, 4, RowMajor> input(in_cols, in_rows, in_depth, in_channels);
  Tensor<float, 5, RowMajor> kernel(kern_width, kern_height, kern_depth,
                                    in_channels, kern_filters);
  Tensor<float, 4, RowMajor> result(out_width, out_height, out_depth,
                                    kern_filters);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  const int stride = 2;
  result =
      CuboidConvolution(input, kernel, stride, stride, stride, PADDING_VALID);

  EXPECT_EQ(result.dimension(3), kern_filters);
  EXPECT_EQ(result.dimension(2), out_depth);
  EXPECT_EQ(result.dimension(1), out_height);
  EXPECT_EQ(result.dimension(0), out_width);

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  expected += input(c + stride * k, r + stride * j,
                                    p + stride * i, id) *
                              kernel(c, r, p, id, od);
                }
              }
            }
          }
          EigenApprox(result(k, j, i, od), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedSameCuboid) {
  const int in_channels = 10;
  const int in_depth = 8;
  const int in_rows = 7;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int stride = 2;
  const int out_depth = ceil_div(in_depth, stride);
  const int out_height = ceil_div(in_rows, stride);
  const int out_width = ceil_div(in_cols, stride);

  Tensor<float, 4> input(in_channels, in_depth, in_rows, in_cols);
  Tensor<float, 5> kernel(kern_filters, in_channels, kern_depth, kern_height,
                          kern_width);
  Tensor<float, 4> result(kern_filters, out_depth, out_height, out_width);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result =
      CuboidConvolution(input, kernel, stride, stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(0), kern_filters);
  EXPECT_EQ(result.dimension(1), out_depth);
  EXPECT_EQ(result.dimension(2), out_height);
  EXPECT_EQ(result.dimension(3), out_width);

  const int pad_p = (out_depth - 1) * stride - in_depth + kern_depth;
  const int pad_r = (out_height - 1) * stride - in_rows + kern_height;
  const int pad_c = (out_width - 1) * stride - in_cols + kern_width;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  const int in_p = p - dp + i * stride;
                  const int in_r = r - dr + j * stride;
                  const int in_c = c - dc + k * stride;
                  if (in_p >= 0 && in_r >= 0 && in_c >= 0 && in_p < in_depth &&
                      in_r < in_rows && in_c < in_cols) {
                    expected +=
                        input(id, in_p, in_r, in_c) * kernel(od, id, p, r, c);
                  }
                }
              }
            }
          }
          EigenApprox(result(od, i, j, k), expected);
        }
      }
    }
  }
}

TEST(EigenSpatialConvolutionsTest, StridedSameCuboidRowMajor) {
  const int in_channels = 10;
  const int in_depth = 8;
  const int in_rows = 7;
  const int in_cols = 5;

  const int kern_filters = 7;
  const int kern_depth = 3;
  const int kern_width = 3;
  const int kern_height = 3;

  const int stride = 2;
  const int out_depth = ceil_div(in_depth, stride);
  const int out_height = ceil_div(in_rows, stride);
  const int out_width = ceil_div(in_cols, stride);

  Tensor<float, 4, RowMajor> input(in_cols, in_rows, in_depth, in_channels);
  Tensor<float, 5, RowMajor> kernel(kern_width, kern_height, kern_depth,
                                    in_channels, kern_filters);
  Tensor<float, 4, RowMajor> result(out_width, out_height, out_depth,
                                    kern_filters);
  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();
  result.setRandom();

  result =
      CuboidConvolution(input, kernel, stride, stride, stride, PADDING_SAME);

  EXPECT_EQ(result.dimension(3), kern_filters);
  EXPECT_EQ(result.dimension(2), out_depth);
  EXPECT_EQ(result.dimension(1), out_height);
  EXPECT_EQ(result.dimension(0), out_width);

  const int pad_p = (out_depth - 1) * stride - in_depth + kern_depth;
  const int pad_r = (out_height - 1) * stride - in_rows + kern_height;
  const int pad_c = (out_width - 1) * stride - in_cols + kern_width;

  // Number of pixels the input is extended with at the lower end in every
  // dimension.
  const int dp = pad_p / 2;
  const int dr = pad_r / 2;
  const int dc = pad_c / 2;

  for (int od = 0; od < kern_filters; ++od) {
    for (int i = 0; i < out_depth; ++i) {
      for (int j = 0; j < out_height; ++j) {
        for (int k = 0; k < out_width; ++k) {
          float expected = 0.0f;
          for (int c = 0; c < kern_width; ++c) {
            for (int r = 0; r < kern_height; ++r) {
              for (int p = 0; p < kern_depth; ++p) {
                for (int id = 0; id < in_channels; ++id) {
                  const int in_p = p - dp + i * stride;
                  const int in_r = r - dr + j * stride;
                  const int in_c = c - dc + k * stride;
                  if (in_p >= 0 && in_r >= 0 && in_c >= 0 && in_p < in_depth &&
                      in_r < in_rows && in_c < in_cols) {
                    expected +=
                        input(in_c, in_r, in_p, id) * kernel(c, r, p, id, od);
                  }
                }
              }
            }
          }
          EigenApprox(result(k, j, i, od), expected);
        }
      }
    }
  }
}

// A test case discovered when testing backward spatial convolution where the
// special tensor contraction mapper for spatial convolution contains a bug.
TEST(EigenSpatialConvolutionsTest, SpatialConvContractionMapper) {
  // We have a 3x4 input image with 2x2 patch and stride of 2.
  // The output has size 1x2.
  typedef Tensor<float, 1>::DimensionPair DimPair;
  Tensor<float, 4> out(1, 1, 2, 1);
  Tensor<float, 4> kern(1, 1, 2, 2);
  for (int i = 0; i < kern.size(); ++i) {
    kern.coeffRef(i) = static_cast<float>(i) + 1;
  }
  for (int i = 0; i < out.size(); ++i) {
    out.coeffRef(i) = static_cast<float>(i) + 1;
  }

  DSizes<ptrdiff_t, 4> strides;
  strides[0] = 1;
  strides[1] = 2;
  strides[2] = 2;
  strides[3] = 1;

  array<std::pair<ptrdiff_t, ptrdiff_t>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(1, 2);
  paddings[2] = std::make_pair(1, 1);
  paddings[3] = std::make_pair(0, 0);

  DSizes<ptrdiff_t, 3> out_dim;
  out_dim[0] = 1;
  out_dim[1] = 4;
  out_dim[2] = 12;

  array<bool, 4> kernel_reverse;
  kernel_reverse[0] = false;
  kernel_reverse[1] = false;
  kernel_reverse[2] = true;
  kernel_reverse[3] = true;

  DSizes<ptrdiff_t, 3> k_dims;
  k_dims[0] = 1;
  k_dims[1] = 1;
  k_dims[2] = 4;

  array<DimPair, 2> contract_dims;
  contract_dims[0] = DimPair(0, 0);
  contract_dims[1] = DimPair(2, 1);

  DSizes<ptrdiff_t, 4> in_dim;
  in_dim[0] = 1;
  in_dim[1] = 3;
  in_dim[2] = 4;
  in_dim[3] = 1;

  DSizes<ptrdiff_t, 2> in_dbg_dim;
  in_dbg_dim[0] = 3;
  in_dbg_dim[1] = 4;

  DSizes<ptrdiff_t, 2> out_dbg_dim;
  out_dbg_dim[0] = 4;
  out_dbg_dim[1] = 12;

  // This is the formula for computing the backward prop for input with a
  // spatial convolution.
  Tensor<float, 4> direct =
      kern.reverse(kernel_reverse)
          .reshape(k_dims)
          .contract(
              out.extract_image_patches(2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 0)
                  .reshape(out_dim),
              contract_dims)
          .reshape(in_dim);

  Tensor<float, 4> indirect =
      kern.reverse(kernel_reverse)
          .reshape(k_dims)
          .contract(
              out.inflate(strides)
                  .pad(paddings)
                  .extract_image_patches(2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
                  .reshape(out_dim),
              contract_dims)
          .reshape(in_dim);

  eigen_assert(dimensions_match(direct.dimensions(), indirect.dimensions()));
  for (size_t i = 0; i < direct.dimensions().TotalSize(); ++i) {
    EigenApprox(direct.data()[i], indirect.data()[i]);
  }
  EigenApprox(1.0f, direct(0, 0, 0, 0));
  EigenApprox(3.0f, direct(0, 0, 1, 0));
  EigenApprox(2.0f, direct(0, 0, 2, 0));
  EigenApprox(6.0f, direct(0, 0, 3, 0));

  EigenApprox(2.0f, direct(0, 1, 0, 0));
  EigenApprox(4.0f, direct(0, 1, 1, 0));
  EigenApprox(4.0f, direct(0, 1, 2, 0));
  EigenApprox(8.0f, direct(0, 1, 3, 0));
}

template <typename T>
static void PackRhsHelper(int iters,
                          /* Input dimensions: */
                          int input_batches, int input_cols, int input_rows,
                          int input_depth,
                          /* Filter (kernel) dimensions: */
                          int filter_count, int filter_cols, int filter_rows,
                          /* Input strides: */
                          int col_strides, int row_strides,
                          /* Patch inflate strides: */
                          int patch_col_inflate_stride,
                          int patch_row_inflate_stride,
                          /* Block dimensions: */
                          Index block_rows, Index block_cols) {
  // Set random seed for benchmark repeatability.
  srand(12345);

  tensorflow::testing::UseRealTime();
  tensorflow::testing::StopTiming();

  using Dimensions = Eigen::DSizes<Eigen::Index, 4>;

  // Default Eigen::Tensor layout is column major, so we configure dimensions
  // starting from the inner most (channels aka depth in this case).
  Dimensions input_dims(input_depth, input_rows, input_cols, input_batches);

  static const int packet_size = Eigen::internal::packet_traits<T>::size;

  // Reshape dimensions.
  using NewDimension = Eigen::DSizes<Index, 2>;

  // Contraction dimensions.
  using nocontract_t = Eigen::array<Eigen::Index, 1>;
  using contract_t = Eigen::array<Eigen::Index, 1>;

  // Input to the TensorImagePatchOp. It is the tensorflow TTypes<T>::Tensor
  // with ColMajor layout, instead of RowMajor. But that doesn't make any
  // difference, because TensorContraction swaps LHS with RHS for row major
  // inputs, and contraction mapper always works with column major data.
  using ArgType = TensorMap<Tensor<T, 4>, Eigen::Aligned>;

  using Evaluator = TensorEvaluator<
      const TensorReshapingOp<
          NewDimension, const TensorImagePatchOp<Dynamic, Dynamic, ArgType>>,
      Eigen::DefaultDevice>;

  using InputMapper = Eigen::internal::TensorContractionInputMapper<
      T, Index, Eigen::internal::Rhs, Evaluator,  //
      nocontract_t, contract_t,                   //
      packet_size,                                //
      /*inner_dim_contiguous*/ true,              //
      /*inner_dim_reordered*/ false,              //
      /*Alignment*/ 0>;

  using SubMapper = Eigen::internal::TensorContractionSubMapper<
      T, Index, Eigen::internal::Rhs, Evaluator,  //
      nocontract_t, contract_t,                   //
      packet_size,                                //
      /*inner_dim_contiguous*/ true,              //
      /*inner_dim_reordered*/ false,              //
      /*Alignment*/ 0>;

#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
  using PackRhsImpl =
      Eigen::internal::gemm_pack_colmajor_block<T, Eigen::Index, SubMapper,
                                                ColMajor>;
#else
  using Traits = typename Eigen::internal::gebp_traits<T, T>;
  using PackRhsImpl =
      Eigen::internal::gemm_pack_rhs<T, Eigen::Index, SubMapper,  //
                                     Traits::nr,                  //
                                     ColMajor,                    //
                                     /*Conjugate*/ false,         //
                                     /*PanelMode*/ false>;
#endif

  Eigen::DefaultDevice device;

  // Actual contract dimensions are not important.
  const Eigen::Index not_important = -1234;
  nocontract_t nocontract_dim = {not_important};
  contract_t contract_dim = {not_important};

  // We use tensor of the same dimensions to store packed data.
  Tensor<T, 4> packed(input_dims);

  // We generate multiple input tensors, around 512mb in total size to measure
  // realistic workload when input data in not in L1-L3 cache.
  size_t input_bytes = input_dims.TotalSize() * sizeof(T);
  size_t mem_size_bytes = 1024 * 1024 * 512;
  size_t num_inputs =
      std::max(static_cast<size_t>(1), mem_size_bytes / input_bytes);

  std::vector<Tensor<T, 4>> inputs;
  std::vector<Evaluator> evaluators;
  std::vector<InputMapper> input_mappers;

  inputs.reserve(num_inputs);
  evaluators.reserve(num_inputs);
  input_mappers.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    inputs.emplace_back(input_dims);
    inputs[i].setRandom();

    ArgType tensor_map(inputs[i].data(), input_dims);

    // 1. Extract image patches from input tensor. All strides are `1`.
    const auto image_patch_op = TensorImagePatchOp<Dynamic, Dynamic, ArgType>(
        tensor_map,                                          //
        filter_rows, filter_cols,                            //
        row_strides, col_strides,                            //
        /*in_row_strides=*/1, /*in_col_strides=*/1,          //
        patch_row_inflate_stride, patch_col_inflate_stride,  //
        Eigen::PADDING_SAME, /*padding_value=*/0.0);

    // 2. Reshape extracted patches into "virtual" 2d tensor.
    // NOTE: This is valid for PADDING_SAME only.
    Index input_rows_eff = (input_rows - 1) * patch_row_inflate_stride + 1;
    Index input_cols_eff = (input_cols - 1) * patch_col_inflate_stride + 1;
    Index output_rows = input_rows_eff / row_strides;
    Index output_cols = input_cols_eff / col_strides;
    NewDimension reshape_dims;
    reshape_dims[0] = input_depth * filter_rows * filter_cols;    // patch size
    reshape_dims[1] = output_rows * output_cols * input_batches;  // num_patches

    const auto reshape_op =
        TensorReshapingOp<NewDimension, decltype(image_patch_op)>(
            image_patch_op, reshape_dims);

    evaluators.emplace_back(reshape_op, device);

    input_mappers.emplace_back(evaluators[i], nocontract_dim, nocontract_dim,
                               contract_dim, contract_dim);
  }

  // We read properties of extracted image patches directly from evaluator.
  const Index patch_depth = evaluators[0].impl().dimensions()[0];
  const Index patch_rows = evaluators[0].impl().dimensions()[1];
  const Index patch_cols = evaluators[0].impl().dimensions()[2];

  // Number of patches is the same as the maximum column available through the
  // InputMapper (SubMapper).
  const Index num_patches = evaluators[0].impl().dimensions()[3];

  // The size of a single patch, it's the same as the maximum depth available
  // through the InputMapper (SubMapper).
  const Index patch_size = patch_depth * patch_rows * patch_cols;

  PackRhsImpl pack_rhs;

  const Index packed_total_size = input_dims.TotalSize();

  // Round up row/col/memory offsets to make them multiple of packet size.
  const auto round_up = [](const Index idx) {
    return (idx / packet_size) * packet_size;
  };

  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    int input_idx =
        num_inputs == 1 ? 1 : internal::random<int>(0, num_inputs - 1);

    // Depth offset must be a multiple packet size.
    Index depth_offset =
        (patch_size > block_rows)
            ? round_up(internal::random<Index>(0, patch_size - 10))
            : 0;
    Index col_offset = internal::random<Index>(0, num_patches - 10);

    Index depth = std::min(block_rows, patch_size - depth_offset);
    Index cols = std::min(block_cols, num_patches - col_offset);

    // Write packed data to random memory location to emulate cold caches.
    Index packed_size = depth * cols;
    Index packed_offset =
        internal::random<Index>(0, packed_total_size - packed_size - 1);

    SubMapper sub_mapper =
        input_mappers[input_idx].getSubMapper(depth_offset, col_offset);
    pack_rhs(packed.data() + packed_offset, sub_mapper, depth, cols);
  }
  tensorflow::testing::StopTiming();
  tensorflow::testing::SetLabel(
      absl::StrCat("patch: ", patch_rows, "x", patch_cols, " D", patch_depth,
                   "; num_patches=", num_patches, " patch_size=", patch_size,
                   " num_inputs=", num_inputs));
}

template <typename T>
static void PackLhsHelper(int iters,
                          /* Input dimensions: */
                          int input_depth,
                          /* Filter (kernel) dimensions: */
                          int filter_count, int filter_cols, int filter_rows,
                          /* Block dimensions: */
                          Index block_rows, Index block_cols) {
  // Set random seed for benchmark repeatability.
  srand(12345);

  eigen_assert(block_rows <= filter_count);
  eigen_assert(block_cols <= input_depth * filter_rows * filter_cols);

  tensorflow::testing::UseRealTime();
  tensorflow::testing::StopTiming();

  using Dimensions = Eigen::DSizes<Eigen::Index, 4>;

  // Default Eigen::Tensor layout is column major, so we configure dimensions
  // starting from the inner most (`filter count` aka `kernel filers`).
  Dimensions filter_dims(filter_count, filter_rows, filter_cols, input_depth);

  static const int packet_size = Eigen::internal::packet_traits<T>::size;

  // We are going to reshape filter into 2D tensor.
  using NewDimension = Eigen::DSizes<Index, 2>;

  // Contraction dimensions.
  using nocontract_t = Eigen::array<Eigen::Index, 1>;
  using contract_t = Eigen::array<Eigen::Index, 1>;

  // Input to the ReshapeOp. It is the tensorflow TTypes<T>::Tensor
  // with ColMajor layout, instead of RowMajor. But that doesn't make any
  // difference, because TensorContraction swaps LHS with RHS for row major
  // inputs, and contraction mapper always works with column major data.
  using ArgType = TensorMap<Tensor<T, 4>, Eigen::Aligned>;

  using Evaluator =
      TensorEvaluator<const TensorReshapingOp<NewDimension, ArgType>,
                      Eigen::DefaultDevice>;

  using InputMapper = Eigen::internal::TensorContractionInputMapper<
      T, Index, Eigen::internal::Lhs, Evaluator,  //
      nocontract_t, contract_t,                   //
      packet_size,                                //
      /*inner_dim_contiguous*/ true,              //
      /*inner_dim_reordered*/ false,              //
      /*Alignment*/ 0>;

  using SubMapper = Eigen::internal::TensorContractionSubMapper<
      T, Index, Eigen::internal::Lhs, Evaluator,  //
      nocontract_t, contract_t,                   //
      packet_size,                                //
      /*inner_dim_contiguous*/ true,              //
      /*inner_dim_reordered*/ false,              //
      /*Alignment*/ 0>;

#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
  using PackLhsImpl =
      Eigen::internal::gemm_pack_colmajor_block<T, Eigen::Index, SubMapper,
                                                ColMajor>;
#else
  using Traits = typename Eigen::internal::gebp_traits<T, T>;
  using PackLhsImpl =
      Eigen::internal::gemm_pack_lhs<T, Eigen::Index, SubMapper,          //
                                     Traits::mr,                          //
                                     Traits::LhsProgress,                 //
                                     typename Traits::LhsPacket4Packing,  //
                                     ColMajor>;
#endif

  Eigen::DefaultDevice device;

  // We will reshape kernel into 2D tensor.
  NewDimension reshape_dims;
  reshape_dims[0] = filter_count;
  reshape_dims[1] = input_depth * filter_rows * filter_cols;

  // We are going to contract along the 'in_depth * filter_rows * filter_cols`.
  nocontract_t nocontract_dim = {0};
  contract_t contract_dim = {1};

  // These values computed using the algorithm in TensorContraction.h, with
  // 'nocontract_dim' and 'contract_dim' values specified above.
  nocontract_t nocontract_strides = {1};
  contract_t contract_strides = {filter_count};
  nocontract_t i_strides = {1};
  contract_t k_strides = {1};

  // We use tensor of the same dimensions to store packed data.
  Tensor<T, 4> packed(filter_dims);

  // We generate multiple filter tensors, around 512mb in total size to measure
  // realistic workload when input data in not in L1-L3 cache.
  size_t input_bytes = filter_dims.TotalSize() * sizeof(T);
  size_t mem_size_bytes = 1024 * 1024 * 512;
  size_t num_filters =
      std::max(static_cast<size_t>(1), mem_size_bytes / input_bytes);

  std::vector<Tensor<T, 4>> filters;
  std::vector<Evaluator> evaluators;
  std::vector<InputMapper> input_mappers;

  filters.reserve(num_filters);
  evaluators.reserve(num_filters);
  input_mappers.reserve(num_filters);

  for (int i = 0; i < num_filters; ++i) {
    filters.emplace_back(filter_dims);
    filters[i].setRandom();

    ArgType tensor_map(filters[i].data(), filter_dims);

    const auto reshape_op =
        TensorReshapingOp<NewDimension, ArgType>(tensor_map, reshape_dims);

    evaluators.emplace_back(reshape_op, device);

    input_mappers.emplace_back(evaluators[i], nocontract_strides, i_strides,
                               contract_strides, k_strides);
  }

  PackLhsImpl pack_lhs;

  const Index packed_total_size = filter_dims.TotalSize();

  // Round up row/col/memory offsets to make them multiple of packet size.
  const auto round_up = [](const Index idx) {
    return (idx / packet_size) * packet_size;
  };

  // Block rows is in the [0, filter_count) range.
  // Block cols is in the [0, filter_rows * filter_cols * input_depth) range.

  const Index max_row = filter_count;
  const Index max_col = filter_rows * filter_cols * input_depth;

  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    int filter_idx =
        num_filters == 1 ? 1 : internal::random<int>(0, num_filters - 1);

    Index row_offset = round_up(internal::random<Index>(0, max_row - 10));
    Index col_offset = round_up(internal::random<Index>(0, max_col - 10));

    Index rows = std::min(block_rows, max_row - row_offset);
    Index cols = std::min(block_cols, max_col - col_offset);

    // Write packed data to random memory location to emulate cold caches.
    Index packed_offset = round_up(
        internal::random<Index>(0, packed_total_size - rows * cols - 1));

    SubMapper sub_mapper =
        input_mappers[filter_idx].getSubMapper(row_offset, col_offset);

// NOTE: Eigen gemm_pack_lhs accepts contraction depth (k-th dimension) as a
// first argument (aka block cols). MKL-DNN pack is generic for lhs and rhs
// and accepts block rows and cols in the same order for lhs and rhs.
#if defined(TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL)
    pack_lhs(packed.data() + packed_offset, sub_mapper, rows, cols);
#else
    pack_lhs(packed.data() + packed_offset, sub_mapper, cols, rows);
#endif
  }
  tensorflow::testing::StopTiming();
  tensorflow::testing::SetLabel(absl::StrCat(
      "filter: count=", filter_count, " dims=", filter_rows, "x", filter_cols,
      "; input: depth=", input_depth, "; num_filers=", num_filters));
}

// -------------------------------------------------------------------------- //
// Pack RHS
//
// Macro argument names:
//    N: batch size
//    H: height
//    W: width
//    C: input channels
//   FC: filter channels
//   FH: filter height
//   FW: filter width
//   SH: stride in height dimensions
//   SW: stride in width dimensions
//  ISH: patch inflate stride in height dimension
//  ISW: patch inflate stride in width dimension
//   BR: block rows
//   BC: block cols

#define BM_CONCAT(a, b) a##b

#define BM_RHS_NAME(prefix, T, N, H, W, C, FC, FH, FW, SH, SW, ISH, ISW, BR, \
                    BC)                                                      \
  BM_CONCAT(                                                                 \
      BM_##prefix##_##T##_##N##_##H##x##W##_IC##C##_FC##FC##_##FH##x##FW,    \
      _s##SH##x##SW##_is##ISH##x##ISW##_B##BR##x##BC)

#define BM_PackRhs(T, N, H, W, C, FC, FH, FW, SH, SW, ISH, ISW, BR, BC)        \
  static void BM_RHS_NAME(PackRhs, T, N, H, W, C, FC, FH, FW, SH, SW, ISH,     \
                          ISW, BR, BC)(int iters) {                            \
    PackRhsHelper<T>(iters, N, H, W, C, FC, FH, FW, SH, SW, ISH, ISW, BR, BC); \
  }                                                                            \
  BENCHMARK(BM_RHS_NAME(PackRhs, T, N, H, W, C, FC, FH, FW, SH, SW, ISH, ISW,  \
                        BR, BC))

// Number of input channel (input depth) it equal to the number of patch
// channels (patch depth).

// NOTE: This is the most common case in Tensorflow models.
// Fast path: input channel dimension is the multiple of the packet size.
BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 32,                //
           /*num_filters*/ 64,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 32,                //
           /*num_filters*/ 64,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 2, 2,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

// Slow path: input channel dimension is not the multiple of the packet size.
BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 30,                //
           /*num_filters*/ 64,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 30,                //
           /*num_filters*/ 64,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 2, 2,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

// Slow path with input channel dimension smaller than the packet size.
BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 256, 256,             //
           /*channels*/ 4,                 //
           /*num_filters*/ 16,             //
           /*filter*/ 8, 8,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 256, 256,             //
           /*channels*/ 4,                 //
           /*num_filters*/ 16,             //
           /*filter*/ 8, 8,                //
           /*stride*/ 2, 4,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);

// Short and wide block with small input channel dimension.
BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 4,                 //
           /*num_filters*/ 16,             //
           /*filter*/ 3, 3,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 36, 432);

BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 4,                 //
           /*num_filters*/ 16,             //
           /*filter*/ 3, 3,                //
           /*stride*/ 2, 2,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 36, 432);

BM_PackRhs(/*type*/ float,                 //
           /*batch*/ 32,                   //
           /*image*/ 32, 32,               //
           /*channels*/ 96,                //
           /*num_filters*/ 96,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 2, 2,  //
           /*block*/ 272, 240);

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
using qint8 = Eigen::QInt8;
BM_PackRhs(/*type*/ qint8,                 //
           /*batch*/ 32,                   //
           /*image*/ 64, 64,               //
           /*channels*/ 32,                //
           /*num_filters*/ 64,             //
           /*filter*/ 5, 5,                //
           /*stride*/ 1, 1,                //
           /*patch inflate stride*/ 1, 1,  //
           /*block*/ 256, 56);
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

// -------------------------------------------------------------------------- //
// Pack LHS
//
// Macro argument names:
//    C: input channels
//   FC: filter channels
//   FH: filter height
//   FW: filter width
//   BR: block rows
//   BC: block cols

#define BM_LHS_NAME(prefix, T, C, FC, FH, FW, BR, BC) \
  BM_CONCAT(BM_##prefix##_##T##_##C##_FC##FC##_##FH##x##FW, _B##BR##x##BC)

#define BM_PackLhs(T, C, FC, FH, FW, BR, BC)                              \
  static void BM_LHS_NAME(PackLhs, T, C, FC, FH, FW, BR, BC)(int iters) { \
    PackLhsHelper<T>(iters, C, FC, FH, FW, BR, BC);                       \
  }                                                                       \
  BENCHMARK(BM_LHS_NAME(PackLhs, T, C, FC, FH, FW, BR, BC))

// Number of input channel (input depth) it equal to the number of patch
// channels (patch depth).

BM_PackLhs(/*type*/ float,            //
           /*input channels*/ 128,    //
           /*filter channels*/ 1024,  //
           /*filter dims*/ 3, 3,      //
           /*block*/ 256, 56);

BM_PackLhs(/*type*/ float,            //
           /*input channels*/ 128,    //
           /*filter channels*/ 1024,  //
           /*filter dims*/ 3, 3,      //
           /*block*/ 56, 256);

BM_PackLhs(/*type*/ float,          //
           /*input channels*/ 30,   //
           /*filter channels*/ 64,  //
           /*filter dims*/ 3, 3,    //
           /*block*/ 256, 56);

BM_PackLhs(/*type*/ float,          //
           /*input channels*/ 50,   //
           /*filter channels*/ 64,  //
           /*filter dims*/ 3, 3,    //
           /*block*/ 56, 256);
}  // namespace Eigen
