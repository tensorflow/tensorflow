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

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_FORMAT_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_FORMAT_H_

#include <array>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Tensor format for input/output activations used in convolution operations.
// The mnemonics specify the meaning of each tensor dimension sorted from
// largest to smallest memory stride.
// N = Batch, H = Image Height, W = Image Width, C = Number of Channels.
// TODO(pauldonnelly): It would probably be better to switch to a registration
// process for tensor formats, so specialized formats could be defined more
// locally to where they are used.
enum TensorFormat {
  // FORMAT_NHWC is the default format in TensorFlow.
  FORMAT_NHWC = 0,

  // FORMAT_NCHW often improves performance on GPUs.
  FORMAT_NCHW = 1,

  // NCHW_VECT_C is the most performant tensor format for cudnn6's quantized
  // int8 convolution and fused convolution. It is laid out in the same order
  // as NCHW, except that the size of the Channels dimension is divided by 4,
  // and a new dimension of size 4 is appended, which packs 4 adjacent channel
  // activations for the same pixel into an int32. Thus an NCHW format tensor
  // with dimensions [N, C, H, W] would have dimensions [N, C/4, H, W, 4] in
  // NCHW_VECT_C format.
  // A pre-condition of this format is that C must be a multiple of 4.
  FORMAT_NCHW_VECT_C = 2,

  // Similar to NHWC, but the size of the W dimension is divided by 4, and a
  // new dimension of size 4 is appended, which packs 4 adjacent activations
  // in the width dimension.
  FORMAT_NHWC_VECT_W = 3,

  // Note: although the current code in this file assumes VECT_C and VECT_W
  // enums imply int8x4 vectors, this should not be relied upon.
  // In the future we may change the meaning of these enums to include vectors
  // of other types such as int16x2, with op implementations automatically
  // determining which format is implied based on the datatype.

  // FORMAT_HWNC is for TPUs.
  FORMAT_HWNC = 4,

  // FORMAT_HWCN is for TPUs.
  FORMAT_HWCN = 5,
};

// Tensor format for convolutional filters.
// The mnemonics specify the meaning of each tensor dimension sorted
// from largest to smallest memory stride.
// H = Kernel Height, W = Kernel Width, I = Input Channels, O = Output Channels.
// Note: In cudnnGetFilter4dDescriptor(), 'O' is called 'K', 'I' is called 'C'.
enum FilterTensorFormat {
  // FORMAT_HWIO is the default filter format in TensorFlow.
  // Ops that do not have a 'filter_format' attribute will assume this format.
  FORMAT_HWIO = 0,

  // FORMAT_OIHW often improves performance on GPUs.
  FORMAT_OIHW = 1,

  // OIHW_VECT_I is the most performant tensor format for cudnn6's quantized
  // int8 convolution and fused convolution. It is analogous to the NCHW_VECT_C
  // data format. It is laid out in the same order as OIHW, except that the size
  // of the Input Channels dimension is divided by 4, and a new dimension of
  // size 4 is appended, which packs 4 adjacent input channel weights into an
  // int32. Thus an OIHW format filter with dimensions [O, I, H, W] would have
  // dimensions [O, I/4, H, W, 4] in OIHW_VECT_I format.
  // A pre-condition of this format is that I must be a multiple of 4.
  FORMAT_OIHW_VECT_I = 2,
};

// Parse tensor format from the given string.
// Return true if the parsing succeeds, and false if it fails.
bool FormatFromString(const string& format_str, TensorFormat* format);

// Parse tensor format from the given string.
// Return true if the parsing succeeds, and false if it fails.
bool FilterFormatFromString(const string& format_str,
                            FilterTensorFormat* format);

// Convert a tensor format into string.
string ToString(TensorFormat format);

// Convert a filter tensor format into string.
string ToString(FilterTensorFormat format);

// Returns the number of spatial dims of a tensor of rank 'num_dims' and tensor
// format 'format'.
inline int GetTensorSpatialDims(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_HWNC:
    case FORMAT_HWCN:
      return num_dims - 2;  // Exclude N,C.
    case FORMAT_NCHW_VECT_C:
    case FORMAT_NHWC_VECT_W:
      // Note: the VECT_W is not counted as an independent spatial dim here,
      // since it just a component of the width dimension.
      return num_dims - 3;  // Exclude N,C,VectDim.
  }
}

inline int GetFilterTensorSpatialDims(int num_dims, FilterTensorFormat format) {
  if (format == FORMAT_OIHW_VECT_I) {
    return num_dims - 3;  // Exclude O,I,InnerI.
  } else {
    return num_dims - 2;  // Exclude O,I.
  }
}

// Returns the rank of a tensor with 'num_spatial_dims' spatial dimensions and
// tensor format 'format'. This is the inverse of GetTensorSpatialDims.
inline int GetTensorDimsFromSpatialDims(int num_spatial_dims,
                                        TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_HWNC:
    case FORMAT_HWCN:
      return num_spatial_dims + 2;  // Include N,C.
    case FORMAT_NCHW_VECT_C:
    case FORMAT_NHWC_VECT_W:
      return num_spatial_dims + 3;  // Include N,C,VectDim.
  }
}

// Returns the rank of a tensor with 'num_spatial_dims' spatial dimensions and
// filter tensor format 'format'.
inline int GetFilterTensorDimsFromSpatialDims(int num_spatial_dims,
                                              FilterTensorFormat format) {
  if (format == FORMAT_OIHW_VECT_I) {
    return num_spatial_dims + 3;  // Include O,I,InnerI.
  } else {
    return num_spatial_dims + 2;  // Include O,I.
  }
}

// Returns the index of the batch dimension.
inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
    case FORMAT_NHWC_VECT_W:
      return 0;
    case FORMAT_HWNC:
      return num_dims - 2;
    case FORMAT_HWCN:
      return num_dims - 1;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the feature dimension. If format is NCHW_VECT_C, returns
// the index of the outer feature dimension (i.e. dimension 1, whose size would
// be num_features / 4 in this case).
inline int GetTensorFeatureDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_HWNC:
      return num_dims - 1;
    case FORMAT_NHWC_VECT_W:
    case FORMAT_HWCN:
      return num_dims - 2;
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
  DCHECK_EQ(format, FORMAT_NCHW_VECT_C);
  return num_dims - 1;
}

// Returns the index of the inner width dimension.
inline int GetTensorInnerWidthDimIndex(int num_dims, TensorFormat format) {
  DCHECK_EQ(format, FORMAT_NHWC_VECT_W);
  return num_dims - 1;
}

// Returns the dimension index of the specified 'spatial_dim' within an
// activation tensor. If format is NHWC_VECT_W and spatial_dim is 1, returns
// the index of the outer width dimension (i.e. dimension 2, whose size would
// be width / 4 in this case).
inline int GetTensorSpatialDimIndex(int num_dims, TensorFormat format,
                                    int spatial_dim) {
  CHECK(spatial_dim >= 0 &&
        spatial_dim < GetTensorSpatialDims(num_dims, format))
      << spatial_dim << " " << num_dims << " " << ToString(format);
  switch (format) {
    case FORMAT_NHWC:
    case FORMAT_NHWC_VECT_W:
      return spatial_dim + 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C:
      return spatial_dim + 2;
    case FORMAT_HWNC:
    case FORMAT_HWCN:
      return spatial_dim;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

inline int GetFilterTensorSpatialDimIndex(int num_dims,
                                          FilterTensorFormat format, int dim) {
  CHECK(dim >= 0 && dim < GetFilterTensorSpatialDims(num_dims, format))
      << dim << " " << num_dims << " " << ToString(format);
  switch (format) {
    case FORMAT_HWIO:
      return dim;
    case FORMAT_OIHW:
    case FORMAT_OIHW_VECT_I:
      return dim + 2;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the inner input channels dimension.
inline int GetFilterTensorInnerInputChannelsDimIndex(
    int num_dims, FilterTensorFormat format) {
  DCHECK_EQ(format, FORMAT_OIHW_VECT_I);
  return num_dims - 1;
}

// Returns the index of the input channels dimension.
// If 'format' is FORMAT_OIHW_VECT_I, returns the dimension index of the
// outer input channel (i.e. 1), which holds num_input_channels / 4.
inline int GetFilterTensorInputChannelsDimIndex(int num_dims,
                                                FilterTensorFormat format) {
  switch (format) {
    case FORMAT_HWIO:
      return num_dims - 2;
    case FORMAT_OIHW:
    case FORMAT_OIHW_VECT_I:
      return 1;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// Returns the index of the output channels dimension.
inline int GetFilterTensorOutputChannelsDimIndex(int num_dims,
                                                 FilterTensorFormat format) {
  switch (format) {
    case FORMAT_HWIO:
      return num_dims - 1;
    case FORMAT_OIHW:
    case FORMAT_OIHW_VECT_I:
      return 0;
    default:
      LOG(FATAL) << "Unknown format " << format;
      return -1;  // Avoid compiler warning about missing return value
  }
}

// TODO(pauldonnelly): Replace these tensor dimension index functions with
// constant structs to improve performance and reduce code size in Compute()
// functions.

// Return the dimension index for the specified 'dimension' of the specified
// data 'tensor_format'.  'dimension' is a char that can be 'N' (batch size),
// 'C' (channels), 'H' (height), 'W' (width),  or a numbered spatial dimension:
// '0',  .. (NUM_SPATIAL_DIMS-1)..
// If 'format' is NCHW_VECT_C and 'dimension' is 'C', returns the index of
// the outer channel dimension (i.e. 1).
template <int NUM_SPATIAL_DIMS>
inline int32 GetTensorDimIndex(TensorFormat format, char dimension) {
  if (format == FORMAT_NHWC || format == FORMAT_NHWC_VECT_W) {
    // clang-format off
    switch (dimension) {
      case 'N': return 0;
      case '0': return 1;
      case '1': return 2;
      case '2': return 3;
      case 'H': return NUM_SPATIAL_DIMS - 1;
      case 'W': return NUM_SPATIAL_DIMS;
      case 'C': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (format == FORMAT_NCHW || format == FORMAT_NCHW_VECT_C) {
    switch (dimension) {
      case 'N': return 0;
      case 'C': return 1;
      case '0': return 2;
      case '1': return 3;
      case '2': return 4;
      case 'H': return NUM_SPATIAL_DIMS;
      case 'W': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (format == FORMAT_HWNC) {
    switch (dimension) {
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case 'H': return NUM_SPATIAL_DIMS - 2;
      case 'W': return NUM_SPATIAL_DIMS - 1;
      case 'N': return NUM_SPATIAL_DIMS;
      case 'C': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (format == FORMAT_HWCN) {
    switch (dimension) {
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case 'H': return NUM_SPATIAL_DIMS - 2;
      case 'W': return NUM_SPATIAL_DIMS - 1;
      case 'C': return NUM_SPATIAL_DIMS;
      case 'N': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else {
    LOG(FATAL) << "Invalid format: " << static_cast<int>(format);
    return -1;  // Avoid compiler warning about missing return value
  }
  // clang-format on
}

// Return the dimension index for the specified 'dimension' of the specified
// 'filter_tensor_format'.  'dimension' is a char that can be 'O' (num output
// channels), 'I' (num input channels), 'H' (height), 'W' (width), or a
// numbered spatial dimension: '0',  .. (NUM_SPATIAL_DIMS-1).
// If 'format' is OIHW_VECT_I and 'dimension' is 'I', returns the index of the
// outer input channels dimension (i.e. 1).
template <int NUM_SPATIAL_DIMS>
inline int GetFilterDimIndex(FilterTensorFormat filter_tensor_format,
                             char dimension) {
  // clang-format off
  if (filter_tensor_format == FORMAT_HWIO) {
    switch (dimension) {
      case '0': return 0;
      case '1': return 1;
      case '2': return 2;
      case 'H': return NUM_SPATIAL_DIMS - 2;
      case 'W': return NUM_SPATIAL_DIMS - 1;
      case 'I': return NUM_SPATIAL_DIMS;
      case 'O': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (filter_tensor_format == FORMAT_OIHW ||
             filter_tensor_format == FORMAT_OIHW_VECT_I) {
    switch (dimension) {
      case 'O': return 0;
      case 'I': return 1;
      case '0': return 2;
      case '1': return 3;
      case '2': return 4;
      case 'H': return NUM_SPATIAL_DIMS;
      case 'W': return NUM_SPATIAL_DIMS + 1;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else {
    LOG(FATAL) << "Invalid format: " << static_cast<int>(filter_tensor_format);
    return -1;  // Avoid compiler warning about missing return value
  }
  // clang-format on
}

inline int32 GetTensorDimIndex(TensorFormat format, char dimension) {
  return GetTensorDimIndex<2>(format, dimension);
}

// Return the element from 'dimension_attributes' that corresponds to the
// specified 'dimension' according to 'tensor_format'.
template <typename T>
T GetTensorDim(gtl::ArraySlice<T> dimension_attributes,
               TensorFormat tensor_format, char dimension) {
  int index =
      (GetTensorSpatialDims(dimension_attributes.size(), tensor_format) == 3)
          ? GetTensorDimIndex<3>(tensor_format, dimension)
          : GetTensorDimIndex<2>(tensor_format, dimension);
  CHECK(index >= 0 && index < dimension_attributes.size())
      << "Invalid index from the dimension: " << index << ", " << tensor_format
      << ", " << dimension;
  return dimension_attributes[index];
}

// Return the element from 'dimension_attribute' that corresponds to the
// specified 'dimension' according to 'filter_tensor_format'.
template <typename T>
T GetFilterDim(gtl::ArraySlice<T> dimension_attribute,
               FilterTensorFormat filter_tensor_format, char dimension) {
  int index = (GetFilterTensorSpatialDims(dimension_attribute.size(),
                                          filter_tensor_format) == 3)
                  ? GetFilterDimIndex<3>(filter_tensor_format, dimension)
                  : GetFilterDimIndex<2>(filter_tensor_format, dimension);
  CHECK(index >= 0 && index < dimension_attribute.size())
      << "Invalid index from the dimension: " << index << ", "
      << filter_tensor_format << ", " << dimension;
  return dimension_attribute[index];
}

template <typename T>
T GetTensorDim(const std::vector<T>& attributes, TensorFormat format,
               char dimension) {
  return GetTensorDim(gtl::ArraySlice<T>(attributes), format, dimension);
}

// Return the size of the specified 'dimension' within 'tensor_shape'
// according to 'tensor_format'.
inline int64 GetTensorDim(const TensorShape& tensor_shape,
                          TensorFormat tensor_format, char dimension) {
  return GetTensorDim(gtl::ArraySlice<int64>(tensor_shape.dim_sizes()),
                      tensor_format, dimension);
}

// Return the size of the specified 'dimension' within 'tensor_shape'
// according to 'tensor_filter_format'.
inline int64 GetFilterDim(const TensorShape& tensor_shape,
                          FilterTensorFormat tensor_filter_format,
                          char dimension) {
  return GetFilterDim(gtl::ArraySlice<int64>(tensor_shape.dim_sizes()),
                      tensor_filter_format, dimension);
}

// Return the size of the specified 'dimension' of 'tensor' according to
// 'tensor_format'.
inline int64 GetTensorDim(const Tensor& tensor, TensorFormat tensor_format,
                          char dimension) {
  return GetTensorDim(tensor.shape(), tensor_format, dimension);
}

// Return the size of the specified 'dimension' of 'tensor' according to
// 'filter_tensor_format'.
inline int64 GetFilterDim(const Tensor& tensor,
                          FilterTensorFormat filter_tensor_format,
                          char dimension) {
  return GetFilterDim(tensor.shape(), filter_tensor_format, dimension);
}

// Return the string that specifies the data format for convnet operations.
string GetConvnetDataFormatAttrString();
string GetConvnet3dDataFormatAttrString();

// Return the string that specifies the filter format for convnet operations.
string GetConvnetFilterFormatAttrString();
string GetConvnet3dFilterFormatAttrString();

// Returns a tensor shape for the specified format and dimension sizes.
// Works for both 2D and 3D operations. The output shapes are as follows:
// FORMAT_NHWC:        (N, spatial, C); rank = spatial.size() + 2
// FORMAT_NCHW:        (N, C, spatial); rank = spatial.size() + 2
// FORMAT_NCHW_VECT_C: (N, C, spatial, InnerC); rank = spatial.size() + 3
// FORMAT_NHWC_VECT_W: (N, spatial, C, InnerW); rank = spatial.size() + 3
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N,
                                   gtl::ArraySlice<int64> spatial, int64 C) {
  const int dims = GetTensorDimsFromSpatialDims(spatial.size(), format);
  gtl::InlinedVector<int64, 6> dim_sizes(dims);
  dim_sizes[GetTensorBatchDimIndex(dims, format)] = N;
  for (int dim = 0; static_cast<size_t>(dim) < spatial.size(); dim++) {
    auto dim_size = spatial[dim];
    if (format == FORMAT_NHWC_VECT_W && dim == spatial.size() - 1) {
      CHECK_EQ(0, dim_size % 4)
          << "FORMAT_NHWC_VECT_W requires W to be a multiple of 4, but W="
          << dim_size;
      dim_sizes[GetTensorInnerWidthDimIndex(dims, format)] = 4;
      dim_size /= 4;
    }
    dim_sizes[GetTensorSpatialDimIndex(dims, format, dim)] = dim_size;
  }

  int feature_index = GetTensorFeatureDimIndex(dims, format);
  if (format == FORMAT_NCHW_VECT_C) {
    CHECK_EQ(0, C % 4) << "NCHW_VECT_C requires C to be a multiple of 4, but C="
                       << C;
    C /= 4;
    dim_sizes[GetTensorInnerFeatureDimIndex(dims, format)] = 4;
  }
  dim_sizes[feature_index] = C;
  return TensorShape(dim_sizes);
}

// Return a tensor shape of the specified 'format', and dimensions.
// Works for both 2D and 3D operations. If 'format' is OIHW_VECT_I,
// the output TensorShape has spatial.size() + 3 dimensions, otherwise
// it has spatial.size() + 2 dimensions.
inline TensorShape ShapeFromFilterTensorFormat(FilterTensorFormat format,
                                               gtl::ArraySlice<int64> spatial,
                                               int64 I, int64 O) {
  const int dims = GetFilterTensorDimsFromSpatialDims(spatial.size(), format);
  gtl::InlinedVector<int64, 6> dim_sizes(dims);
  dim_sizes[GetFilterTensorOutputChannelsDimIndex(dims, format)] = O;
  for (int dim = 0; static_cast<size_t>(dim) < spatial.size(); dim++) {
    dim_sizes[GetFilterTensorSpatialDimIndex(dims, format, dim)] = spatial[dim];
  }

  if (format == FORMAT_OIHW_VECT_I) {
    CHECK_EQ(0, I % 4) << "OIHW_VECT_I requires I to be a multiple of 4, but I="
                       << I;
    I /= 4;
    dim_sizes[GetFilterTensorInnerInputChannelsDimIndex(dims, format)] = 4;
  }
  dim_sizes[GetFilterTensorInputChannelsDimIndex(dims, format)] = I;
  return TensorShape(dim_sizes);
}

// Return a tensor shape of the specified 'format', and dimensions.
inline TensorShape ShapeFromFormat(TensorFormat format, int64 N, int64 H,
                                   int64 W, int64 C) {
  return ShapeFromFormat(format, N, {H, W}, C);
}

// Return a filter tensor shape of the specified 'format', and dimensions.
inline TensorShape ShapeFromFilterTensorFormat(FilterTensorFormat format,
                                               int64 H, int64 W, int64 I,
                                               int64 O) {
  return ShapeFromFilterTensorFormat(format, {H, W}, I, O);
}

// Returns a copy of the specified tensor 'src_shape' converted from
// 'src_format' to 'dst_format'.
inline TensorShape ShapeFromFormat(TensorFormat dst_format,
                                   const TensorShape& src_shape,
                                   TensorFormat src_format) {
  if (src_format == dst_format) {
    return src_shape;
  }

  const int64 batch = GetTensorDim(src_shape, src_format, 'N');
  const int64 channels = GetTensorDim(src_shape, src_format, 'C') *
                         (src_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  const int num_src_spatial_dims =
      GetTensorSpatialDims(src_shape.dims(), src_format);
  std::vector<int64> spatial_dims(num_src_spatial_dims);
  for (int spatial_dim = 0; spatial_dim < num_src_spatial_dims; ++spatial_dim) {
    spatial_dims[spatial_dim] =
        gtl::ArraySlice<int64>(src_shape.dim_sizes())[GetTensorSpatialDimIndex(
            src_shape.dims(), src_format, spatial_dim)];
  }
  if (src_format == FORMAT_NHWC_VECT_W) {
    spatial_dims[num_src_spatial_dims - 1] *= 4;
  }
  return ShapeFromFormat(dst_format, batch, {spatial_dims}, channels);
}

// Returns a copy of the specified filter tensor 'src_shape' converted from
// 'src_filter_format' to 'dst_filter_format'.
inline TensorShape ShapeFromFilterFormat(FilterTensorFormat dst_filter_format,
                                         const TensorShape& src_shape,
                                         FilterTensorFormat src_filter_format) {
  if (src_filter_format == dst_filter_format) {
    return src_shape;
  }

  const int64 output_channels = GetFilterDim(src_shape, src_filter_format, 'O');
  const int64 input_channels =
      GetFilterDim(src_shape, src_filter_format, 'I') *
      (src_filter_format == FORMAT_OIHW_VECT_I ? 4 : 1);

  if (GetFilterTensorSpatialDims(src_shape.dims(), src_filter_format) == 3) {
    return ShapeFromFilterTensorFormat(
        dst_filter_format,
        {{GetFilterDim(src_shape, src_filter_format, '0'),
          GetFilterDim(src_shape, src_filter_format, '1'),
          GetFilterDim(src_shape, src_filter_format, '2')}},
        input_channels, output_channels);
  }

  return ShapeFromFilterTensorFormat(
      dst_filter_format,
      {{GetFilterDim(src_shape, src_filter_format, 'H'),
        GetFilterDim(src_shape, src_filter_format, 'W')}},
      input_channels, output_channels);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_FORMAT_H_
