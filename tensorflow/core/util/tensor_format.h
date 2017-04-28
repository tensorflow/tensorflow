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

#ifndef TENSORFLOW_UTIL_TENSOR_FORMAT_H_
#define TENSORFLOW_UTIL_TENSOR_FORMAT_H_

#include <array>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

enum TensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
};

// Parse tensor format from the given string.
// Return true if the parsing succeeds, and false if it fails.
bool FormatFromString(const string& format_str, TensorFormat* format);

// Convert a tensor format into string.
string ToString(TensorFormat format);

// Returns the index of the batch dimension.
inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
  if (format == FORMAT_NHWC || format == FORMAT_NCHW) {
    return 0;
  } else {
    LOG(FATAL) << "Unknown format " << format;
  }
}

// Returns the index of the feature dimension.
inline int GetTensorFeatureDimIndex(int num_dims, TensorFormat format) {
  if (format == FORMAT_NHWC) {
    return num_dims - 1;
  } else if (format == FORMAT_NCHW) {
    return 1;
  } else {
    LOG(FATAL) << "Unknown format " << format;
  }
}

// Returns the index of the `dim`-th spatial dimension.
inline int GetTensorSpatialDimIndex(int num_dims, TensorFormat format,
                                    int dim) {
  CHECK(dim >= 0 && dim < num_dims - 2) << dim << " " << num_dims;
  if (format == FORMAT_NHWC) {
    return dim + 1;
  } else if (format == FORMAT_NCHW) {
    return dim + 2;
  } else {
    LOG(FATAL) << "Unknown format " << format;
  }
}

// Return the position index from a format given a dimension specification with
// a char. The chars can be N (batch), C (channels), H (y), W (x), or
// 0 .. (NDIMS-1).
template <int NDIMS>
inline int32 GetTensorDimIndex(TensorFormat format, char dimension) {
  if (format == FORMAT_NHWC) {
    switch (dimension) {
      case 'N':
        return 0;
      case '0':
        return 1;
      case '1':
        return 2;
      case '2':
        return 3;
      case 'H':
        return NDIMS - 1;
      case 'W':
        return NDIMS;
      case 'C':
        return 1 + NDIMS;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (format == FORMAT_NCHW) {
    switch (dimension) {
      case 'N':
        return 0;
      case 'C':
        return 1;
      case '0':
        return 2;
      case '1':
        return 3;
      case '2':
        return 4;
      case 'H':
        return NDIMS;
      case 'W':
        return NDIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else {
    LOG(FATAL) << "Invalid format: " << static_cast<int>(format);
    return -1;  // Avoid compiler warning about missing return value
  }
}

inline int32 GetTensorDimIndex(TensorFormat format, char dimension) {
  return GetTensorDimIndex<2>(format, dimension);
}

// Return the given tensor dimension from a tensor. The tensor is interpretted
// using the specified format, and a dimension specification using a char.
inline int64 GetTensorDim(const Tensor& tensor, TensorFormat format,
                          char dimension) {
  int index = (tensor.dims() == 5) ? GetTensorDimIndex<3>(format, dimension)
                                   : GetTensorDimIndex<2>(format, dimension);
  CHECK(index >= 0 && index < tensor.dims())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return tensor.dim_size(index);
}

// Return the given tensor dimension from a vector that represents the
// dimensions of a tensor.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
inline int64 GetTensorDim(const TensorShape& tensor_shape, TensorFormat format,
                          char dimension) {
  int index = (tensor_shape.dims() == 5)
                  ? GetTensorDimIndex<3>(format, dimension)
                  : GetTensorDimIndex<2>(format, dimension);
  CHECK(index >= 0 && index < tensor_shape.dims())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return tensor_shape.dim_size(index);
}

// Return the given tensor dimension from a tensor shape.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
template <typename T>
T GetTensorDim(const std::vector<T>& attributes, TensorFormat format,
               char dimension) {
  int index = (attributes.size() == 5)
                  ? GetTensorDimIndex<3>(format, dimension)
                  : GetTensorDimIndex<2>(format, dimension);
  CHECK(index >= 0 && index < attributes.size())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return attributes[index];
}

// Return the string that specifies the data format for convnet operations.
string GetConvnetDataFormatAttrString();
string GetConvnet3dDataFormatAttrString();

// Return a tensor shape for the given format. Works for both 2D and 3D
// operations.
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N,
                                   gtl::ArraySlice<int64> spatial, int64 C) {
  gtl::InlinedVector<int64, 5> dim_sizes(spatial.size() + 2);
  dim_sizes[GetTensorBatchDimIndex(dim_sizes.size(), format)] = N;
  for (int dim = 0; static_cast<size_t>(dim) < spatial.size(); dim++) {
    dim_sizes[GetTensorSpatialDimIndex(dim_sizes.size(), format, dim)] =
        spatial[dim];
  }
  dim_sizes[GetTensorFeatureDimIndex(dim_sizes.size(), format)] = C;

  return TensorShape(dim_sizes);
}

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N, int64 H,
                                   int64 W, int64 C) {
  return ShapeFromFormat(format, N, {{H, W}}, C);
}

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat dst_format,
                                   const TensorShape& src_shape,
                                   TensorFormat src_format) {
  if (src_format == dst_format) {
    return src_shape;
  }

  const int64 channels = GetTensorDim(src_shape, src_format, 'C');
  const int64 batch = GetTensorDim(src_shape, src_format, 'N');

  if (src_shape.dims() == 5) {
    return ShapeFromFormat(dst_format, batch,
                           {{GetTensorDim(src_shape, src_format, '0'),
                             GetTensorDim(src_shape, src_format, '1'),
                             GetTensorDim(src_shape, src_format, '2')}},
                           channels);
  }

  return ShapeFromFormat(dst_format, batch,
                         {{GetTensorDim(src_shape, src_format, 'H'),
                           GetTensorDim(src_shape, src_format, 'W')}},
                         channels);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_FORMAT_H_
