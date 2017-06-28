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
  // NCHW_VECT_C is the most performant tensor format for cudnn6's quantized
  // convolution. It is laid out in the same order as NCHW, but each element of
  // a tensor in this format is a vector of 4 feature maps. A batch image with
  // dimension sizes [N,C,H,W] is represented as a 5D tensor with shape
  // [N,C/4,H,W,4]. This format requires C to be a multiple of 4, and requires
  // the data type to be int8.
  FORMAT_NCHW_VECT_C = 2,
};

// Parse tensor format from the given string.
// Return true if the parsing succeeds, and false if it fails.
bool FormatFromString(const string& format_str, TensorFormat* format);

// Convert a tensor format into string.
string ToString(TensorFormat format);

// Returns the number of spatial dims of a tensor of rank 'num_dims' and tensor
// format 'format'.
inline int GetTensorSpatialDims(int num_dims, TensorFormat format) {
  if (format == FORMAT_NCHW_VECT_C) {
    return num_dims - 3;  // Exclude N,C,InnerC.
  } else {
    return num_dims - 2;  // Exclude N,C.
  }
}

// Returns the index of the batch dimension.
inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return 0;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the feature dimension. If format is NCHW_VECT_C, returns
// the outer feature map count -- the size of the second dimension.
inline int GetTensorFeatureDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return num_dims - 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return 1;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the inner feature dimension.
inline int GetTensorInnerFeatureDimIndex(int num_dims, TensorFormat format) {
  return format == FORMAT_NCHW_VECT_C ? num_dims - 1 : -1;
}

// Returns the index of the `dim`-th spatial dimension.
inline int GetTensorSpatialDimIndex(int num_dims, TensorFormat format,
                                    int dim) {
  CHECK(dim >= 0 && dim < GetTensorSpatialDims(num_dims, format))
      << dim << " " << num_dims << " " << ToString(format);
  switch (format) {
    case FORMAT_NHWC:
      return dim + 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return dim + 2;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Return the position index from a format given a dimension specification with
// a char. The chars can be N (batch), C (channels), H (y), W (x), or
// 0 .. (NDIMS-1). If format is NCHW_VECT_C and dimension is C, returns the
// outer feature map count -- the size of the second dimension.
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
  } else if (format == FORMAT_NCHW || format == FORMAT_NCHW_VECT_C) {
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

// Return the given tensor dimension from a vector that represents the
// dimensions of a tensor.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
template <typename T>
T GetTensorDim(gtl::ArraySlice<T> attributes, TensorFormat format,
               char dimension) {
  int index = (GetTensorSpatialDims(attributes.size(), format) == 3)
                  ? GetTensorDimIndex<3>(format, dimension)
                  : GetTensorDimIndex<2>(format, dimension);
  CHECK(index >= 0 && index < attributes.size())
      << "Invalid index from the dimension: " << index << ", " << format << ", "
      << dimension;
  return attributes[index];
}

template <typename T>
T GetTensorDim(const std::vector<T>& attributes, TensorFormat format,
               char dimension) {
  return GetTensorDim(gtl::ArraySlice<T>(attributes), format, dimension);
}

// Return the given tensor dimension from a tensor shape.
// The tensor is interpretted using the specified format, and a dimension
// specification using a char.
inline int64 GetTensorDim(const TensorShape& tensor_shape, TensorFormat format,
                          char dimension) {
  return GetTensorDim(gtl::ArraySlice<int64>(tensor_shape.dim_sizes()), format,
                      dimension);
}

// Return the given tensor dimension from a tensor. The tensor is interpretted
// using the specified format, and a dimension specification using a char.
inline int64 GetTensorDim(const Tensor& tensor, TensorFormat format,
                          char dimension) {
  return GetTensorDim(tensor.shape(), format, dimension);
}

// Return the string that specifies the data format for convnet operations.
string GetConvnetDataFormatAttrString();
string GetConvnet3dDataFormatAttrString();

// Return a tensor shape for the given format. Works for both 2D and 3D
// operations. If format is FORMAT_NCHW_VECT_C, the output TensorShape has rank
// spatial.size()+3 (N,C,spatial,InnerC); otherwise, it has rank
// spatial.size()+2 (e.g. N,C,spatial or N,spatial,C).
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N,
                                   gtl::ArraySlice<int64> spatial, int64 C) {
  const int dims =
      spatial.size() + (format == FORMAT_NCHW_VECT_C ? 3  // Include N,C,InnerC.
                                                     : 2);  // Include N,C.
  gtl::InlinedVector<int64, 6> dim_sizes(dims);
  dim_sizes[GetTensorBatchDimIndex(dims, format)] = N;
  for (int dim = 0; static_cast<size_t>(dim) < spatial.size(); dim++) {
    dim_sizes[GetTensorSpatialDimIndex(dims, format, dim)] = spatial[dim];
  }

  int feature_index = GetTensorFeatureDimIndex(dims, format);
  if (format == FORMAT_NCHW_VECT_C) {
    CHECK_EQ(0, C % 4) << "NCHW_VECT_C requires C to be a multiple of 4, but C="
                       << C;
    dim_sizes[feature_index] = C / 4;
    dim_sizes[GetTensorInnerFeatureDimIndex(dims, format)] = 4;
  } else {
    dim_sizes[feature_index] = C;
  }
  return TensorShape(dim_sizes);
}

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N, int64 H,
                                   int64 W, int64 C) {
  return ShapeFromFormat(format, N, {H, W}, C);
}

// Return a tensor shape from the given format, and tensor dimensions.
inline TensorShape ShapeFromFormat(TensorFormat dst_format,
                                   const TensorShape& src_shape,
                                   TensorFormat src_format) {
  if (src_format == dst_format) {
    return src_shape;
  }

  const int64 batch = GetTensorDim(src_shape, src_format, 'N');
  const int64 channels = GetTensorDim(src_shape, src_format, 'C') *
                         (src_format == FORMAT_NCHW_VECT_C ? 4 : 1);

  if (GetTensorSpatialDims(src_shape.dims(), src_format) == 3) {
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
