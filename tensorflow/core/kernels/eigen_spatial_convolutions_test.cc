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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"
#include "tensorflow/core/platform/test.h"

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

}  // namespace Eigen
