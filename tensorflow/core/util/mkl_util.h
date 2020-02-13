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

#ifndef TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
#ifdef INTEL_MKL

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using mkldnn::engine;
using mkldnn::memory;
using mkldnn::padding_kind;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;

using CPUDevice = Eigen::ThreadPoolDevice;
using MemoryArgsMap = std::unordered_map<int, memory>;
using ReorderPd = mkldnn::reorder::primitive_desc;

#ifdef _WIN32
typedef unsigned int uint;
#endif

namespace tensorflow {

// The file contains a number of utility classes and functions used by MKL
// enabled kernels

// This class encapsulates all the meta data that is associated with an MKL
// tensor. A tensor is an MKL tensor if it was created as the result of an
// MKL operation, and did not go through a conversion to a standard
// Tensorflow tensor.

// The dimensions order that MKL-DNN internally uses for 2D activations
// [Batch, Channel, Height, Width] and
// for 2D filters [Out_Channel, In_Channel, Height, Width].
typedef enum {
  Dim_N = 0,
  Dim_C = 1,
  Dim_H = 2,
  Dim_W = 3,
  Dim_O = 0,
  Dim_I = 1
} MklDnnDims;

// The dimensions order that MKL-DNN internally uses for 3D activations
// [Batch, Channel, Depth, Height, Width] and
// for 3D filters [Out_Channel, In_Channel, Depth, Height, Width].
typedef enum {
  Dim3d_N = 0,
  Dim3d_C = 1,
  Dim3d_D = 2,
  Dim3d_H = 3,
  Dim3d_W = 4,
  Dim3d_O = 0,
  Dim3d_I = 1
} MklDnnDims3D;

// Enum for the order of dimensions of a TF 2D filter with shape [filter_height,
// filter_width, in_channels, out_channels]
typedef enum {
  TF_2DFILTER_DIM_H = 0,
  TF_2DFILTER_DIM_W = 1,
  TF_2DFILTER_DIM_I = 2,
  TF_2DFILTER_DIM_O = 3
} TFFilterDims2d;

// Enum for the order of dimensions of a TF 3D filter with shape [filter_depth,
// filter_height, filter_width, in_channels, out_channels]
typedef enum {
  TF_3DFILTER_DIM_P = 0,
  TF_3DFILTER_DIM_H = 1,
  TF_3DFILTER_DIM_W = 2,
  TF_3DFILTER_DIM_I = 3,
  TF_3DFILTER_DIM_O = 4
} TFFilterDims3d;

// The dimensions order that MKL-DNN requires for the filter in a grouped
// convolution (2D only)
typedef enum {
  MKL_GROUP_FILTER_DIM_G = 0,
  MKL_GROUP_FILTER_DIM_O = 1,
  MKL_GROUP_FILTER_DIM_I = 2,
  MKL_GROUP_FILTER_DIM_H = 3,
  MKL_GROUP_FILTER_DIM_W = 4
} MklDnnFilterGroupDims;

// Enum used to templatize MklOp kernel implementation
// that support both fp32 and int8 versions.
enum class MklQuantization {
  QUANTIZED_VERSION,
  FP_VERSION,
};

static const int kSmallBatchSize = 32;

#ifdef ENABLE_MKLDNN_V1
// In MKL-DNN v1.x, the format (ex. NCHW) used to initialize a memory descriptor
// (md) structure will no longer be recorded in its `format` field. Instead, it
// will be set to a canonical `blocked` format for every fully described md.
//
// Currently, we query this `format` field while mapping MKL-DNN's data format
// to TF's data format. Due to the above restriction, we will now get this data
// format information from TF's `data_format` attribute (i.e. via
// `TensorFormat`) for MKL-DNN v1.x.
//
// Some MKL-DNN operators such as ReLU do not have a `data_format` attribute
// since they are usually in `blocked` format. Therefore, in order to
// distinguish between blocked and non-blocked formats, we have defined a new
// enum called `MklTensorFormat` that is semantically similar to `TensorFormat`
// but with the following additional fields namely:
//  1) FORMAT_BLOCKED: as described above, this is needed for element-wise
//     operators such as ReLU.
//  2) FORMAT_INVALID: for error-checking (ex. unsupported format)
//  3) FORMAT_X, FORMAT_NC, FORMAT_TNC: to distinguish between MKL tensors based
//     on their dimensions in operators such as Softmax, i.e.:
//        FORMAT_X   - 1D tensor
//        FORMAT_NC  - 2D tensor
//        FORMAT_TNC - 3D tensor
enum class MklTensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NDHWC = 2,
  FORMAT_NCDHW = 3,
  FORMAT_X = 4,
  FORMAT_NC = 5,
  FORMAT_TNC = 6,
  FORMAT_BLOCKED = 7,
  FORMAT_INVALID = 8,
};

#endif  // ENABLE_MKLDNN_V1

// Forward declarations
MEMORY_FORMAT MklTensorFormatToMklDnnDataFormat(MKL_TENSOR_FORMAT format);

TensorFormat MklDnn3DDataFormatToTFDataFormat(MKL_TENSOR_FORMAT format);
TensorFormat MklDnnDataFormatToTFDataFormat(MKL_TENSOR_FORMAT format);

memory::dims CalculateTFStrides(const memory::dims& dims_tf_order);
Status CreateBlockedMemDescHelper(const memory::dims& dim,
                                  const memory::dims& strides,
                                  memory::data_type dtype,
                                  mkldnn_memory_desc_t* blocked_md);

#ifdef ENABLE_MKLDNN_V1
inline std::ostream& operator<<(std::ostream& os,
                                const memory::format_tag& tag) {
  if (tag == memory::format_tag::undef) {
    os << "undef";
  } else if (tag == memory::format_tag::any) {
    os << "any";
  } else {
    os << "invalid";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const MklTensorFormat& format) {
  if (format == MklTensorFormat::FORMAT_NHWC) {
    os << "FORMAT_NHWC";
  } else if (format == MklTensorFormat::FORMAT_NCHW) {
    os << "FORMAT_NCHW";
  } else if (format == MklTensorFormat::FORMAT_NDHWC) {
    os << "FORMAT_NDHWC";
  } else if (format == MklTensorFormat::FORMAT_NCDHW) {
    os << "FORMAT_NCDHW";
  } else if (format == MklTensorFormat::FORMAT_X) {
    os << "FORMAT_X";
  } else if (format == MklTensorFormat::FORMAT_NC) {
    os << "FORMAT_NC";
  } else if (format == MklTensorFormat::FORMAT_TNC) {
    os << "FORMAT_TNC";
  } else if (format == MklTensorFormat::FORMAT_BLOCKED) {
    os << "FORMAT_BLOCKED";
  } else {
    os << "INVALID FORMAT";
  }
}
#endif  // ENABLE_MKLDNN_V1

template <typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
  for (size_t i = 0; i < size; ++i)
    if (a1[i] != a2[i]) return false;
  return true;
}

class MklDnnShape {
 private:
  typedef struct {
    // Flag to indicate if the tensor is an MKL tensor or not
    bool is_mkl_tensor_ = false;
    // Number of dimensions in Tensorflow format
    size_t dimension_ = 0;
    mkldnn_dims_t sizes_;  // Required by MKL for conversions
    MKL_TENSOR_FORMAT tf_data_format_ = MKL_TENSOR_FORMAT_UNDEF;
    memory::data_type T_ = MEMORY_DATA_TYPE_UNDEF;
    // MKL layout
    mkldnn_memory_desc_t mkl_md_;
    /// TF dimension corresponding to this MKL dimension
    mkldnn_dims_t map_;
  } MklShapeData;
  MklShapeData data_;

  typedef std::remove_extent<mkldnn_dims_t>::type mkldnn_dim_t;

  // Helper function to compare mkldnn_blocking_desc_t.
  inline bool blocking_desc_is_equal(const mkldnn_blocking_desc_t& lhs,
                                     const mkldnn_blocking_desc_t& rhs,
                                     int ndims) const {
    return lhs.offset_padding == rhs.offset_padding &&
           array_cmp(lhs.block_dims, rhs.block_dims, ndims) &&
           array_cmp(lhs.strides[0], rhs.strides[0], ndims) &&
           array_cmp(lhs.strides[1], rhs.strides[1], ndims) &&
           array_cmp(lhs.padding_dims, rhs.padding_dims, ndims) &&
           array_cmp(lhs.offset_padding_to_data, rhs.offset_padding_to_data,
                     ndims);
  }

  // Helper function to compare mkldnn_wino_desc_t.
  inline bool wino_desc_is_equal(const mkldnn_wino_desc_t& lhs,
                                 const mkldnn_wino_desc_t& rhs) const {
    return lhs.wino_format == rhs.wino_format && lhs.alpha == rhs.alpha &&
           lhs.ic == rhs.ic && lhs.oc == rhs.oc &&
           lhs.ic_block == rhs.ic_block && lhs.oc_block == rhs.oc_block &&
           lhs.ic2_block == rhs.ic2_block && lhs.oc2_block == rhs.oc2_block &&
           lhs.r == rhs.r;
  }

  // Helper function to compare mkldnn_rnn_packed_desc_t.
  inline bool rnn_packed_desc_is_equal(
      const mkldnn_rnn_packed_desc_t& lhs,
      const mkldnn_rnn_packed_desc_t& rhs) const {
    return lhs.format == rhs.format && lhs.n_parts == rhs.n_parts &&
           lhs.offset_compensation == rhs.offset_compensation &&
           lhs.size == rhs.size && lhs.n == rhs.n &&
           array_cmp(lhs.parts, rhs.parts, lhs.n_parts) &&
           array_cmp(lhs.part_pack_size, rhs.part_pack_size, lhs.n_parts);
  }
#define INVALID_DIM_SIZE -1

 public:
  MklDnnShape() {
    for (size_t i = 0; i < sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
         ++i) {
      data_.sizes_[i] = -1;
    }
    for (size_t i = 0; i < sizeof(data_.map_) / sizeof(data_.map_[0]); ++i) {
      data_.map_[i] = -1;
    }
  }

  ~MklDnnShape() {}
  TF_DISALLOW_COPY_AND_ASSIGN(MklDnnShape);  // Cannot copy

  /// Helper function to compare memory::desc objects for MklDnn.
  /// May be this should go into MklDnn directly.
  inline bool CompareMklDnnLayouts(const memory::desc& md1,
                                   const memory::desc& md2) const {
    mkldnn_memory_desc_t mdd1 = md1.data;
    mkldnn_memory_desc_t mdd2 = md2.data;

    assert(mdd1.primitive_kind == mkldnn::primitive::kind::memory);
    assert(mdd2.primitive_kind == mkldnn::primitive::kind::memory);
    bool base_equal = mdd1.ndims == mdd2.ndims &&
                      array_cmp(mdd1.dims, mdd2.dims, mdd1.ndims) &&
                      mdd1.data_type == mdd2.data_type &&
                      mdd1.format == mdd2.format;
    if (!base_equal) return false;
    if (mdd1.format == memory::format::blocked) {
      return blocking_desc_is_equal(mdd1.layout_desc.blocking,
                                    mdd2.layout_desc.blocking, mdd1.ndims);
    } else if (mdd1.format == memory::format::wino_fmt) {
      return wino_desc_is_equal(mdd1.layout_desc.wino_desc,
                                mdd2.layout_desc.wino_desc);
    } else if (mdd1.format == memory::format::rnn_packed) {
      return rnn_packed_desc_is_equal(mdd1.layout_desc.rnn_packed_desc,
                                      mdd2.layout_desc.rnn_packed_desc);
    }

    return true;
  }

  /// Equality function for MklDnnShape objects
  /// @return true if both are equal; false otherwise.
  inline bool operator==(const MklDnnShape& input_shape) const {
    if (this->IsMklTensor() != input_shape.IsMklTensor()) {
      return false;
    }

    // If input tensors are in MKL layout, then we check for dimensions and
    // sizes.
    if (this->IsMklTensor()) {
#ifdef ENABLE_MKLDNN_V1
      const mkldnn_memory_desc_t& cur_md = (this->GetMklLayout()).data;
      const mkldnn_memory_desc_t& input_shape_md =
          input_shape.GetMklLayout().data;
      return this->GetTfShape() == input_shape.GetTfShape() &&
             mkldnn_memory_desc_equal(&cur_md, &input_shape_md);
#else
      return this->GetTfShape() == input_shape.GetTfShape() &&
             CompareMklDnnLayouts(this->GetMklLayout(),
                                  input_shape.GetMklLayout());
#endif  // ENABLE_MKLDNN_V1
    }

    // Both inputs are not MKL tensors.
    return true;
  }

  /// Equality operator for MklDnnShape and TFShape.
  /// Returns: true if TF shapes for both are the same, false otherwise
  inline bool operator==(const TensorShape& input_shape) const {
    if (!this->IsMklTensor()) {
      return false;
    }

    return this->GetTfShape() == input_shape;
  }

  inline const bool IsMklTensor() const { return data_.is_mkl_tensor_; }
  inline void SetMklTensor(bool is_mkl_tensor) {
    data_.is_mkl_tensor_ = is_mkl_tensor;
  }

  inline void SetDimensions(const size_t dimension) {
    data_.dimension_ = dimension;
  }
  inline size_t GetDimension(char dimension) const {
    int index = GetMklDnnTensorDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }

  inline size_t GetDimension3D(char dimension) const {
    int index = GetMklDnnTensor3DDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }

  inline int32 GetMklDnnTensorDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return MklDnnDims::Dim_N;
      case 'C':
        return MklDnnDims::Dim_C;
      case 'H':
        return MklDnnDims::Dim_H;
      case 'W':
        return MklDnnDims::Dim_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  }

  inline int32 GetMklDnnTensor3DDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return MklDnnDims3D::Dim3d_N;
      case 'C':
        return MklDnnDims3D::Dim3d_C;
      case 'D':
        return MklDnnDims3D::Dim3d_D;
      case 'H':
        return MklDnnDims3D::Dim3d_H;
      case 'W':
        return MklDnnDims3D::Dim3d_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  }

  inline size_t GetDimension() const { return data_.dimension_; }
  inline const int* GetSizes() const {
    return reinterpret_cast<const int*>(&data_.sizes_[0]);
  }

  // Returns an mkldnn::memory::dims object that contains the sizes of this
  // MklDnnShape object.
  inline memory::dims GetSizesAsMklDnnDims() const {
    memory::dims retVal;
    if (data_.is_mkl_tensor_) {
      size_t dimensions = sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
      for (size_t i = 0; i < dimensions; i++) {
        if (data_.sizes_[i] != INVALID_DIM_SIZE)
          retVal.push_back(data_.sizes_[i]);
      }
    } else {
      CHECK_EQ(data_.is_mkl_tensor_, true);
    }
    return retVal;
  }

  inline int64 DimSize(int index) const {
    CHECK_LT(index, sizeof(data_.sizes_) / sizeof(data_.sizes_[0]));
    return data_.sizes_[index];
  }

  /// Return TensorShape that describes the Tensorflow shape of the tensor
  /// represented by this MklShape.
  inline TensorShape GetTfShape() const {
    CHECK_EQ(data_.is_mkl_tensor_, true);

    std::vector<int32> shape(data_.dimension_, -1);
    // As mentioned in the comment above, we now rely on TF's `data_format`
    // attribute to determine if TF shape is in blocked format or not.
    if (data_.tf_data_format_ != MKL_TENSOR_FORMAT_BLOCKED) {
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[TfDimIdx(idx)];
      }
    } else {
      // If Tensorflow shape is in Blocked format, then we don't have dimension
      // map for it. So we just create Tensorflow shape from sizes in the
      // specified order.
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[idx];
      }
    }

    TensorShape ts;
    bool ret = TensorShapeUtils::MakeShape(shape, &ts).ok();
    CHECK_EQ(ret, true);
    return ts;
  }

  inline void SetElemType(memory::data_type dt) { data_.T_ = dt; }
  inline const memory::data_type GetElemType() { return data_.T_; }

#ifndef ENABLE_MKLDNN_V1
  // Memory primitive descriptor is deprecated in MKL-DNN v1.x.
  inline void SetMklLayout(memory::primitive_desc* pd) {
    CHECK_NOTNULL(pd);
    data_.mkl_md_ = pd->desc().data;
  }
#endif  // !ENABLE_MKLDNN_V1

  inline void SetMklLayout(memory::desc* md) {
    CHECK_NOTNULL(md);
    data_.mkl_md_ = md->data;
  }

  inline const memory::desc GetMklLayout() const {
    return memory::desc(data_.mkl_md_);
  }

  inline MKL_TENSOR_FORMAT GetTfDataFormat() const {
    return data_.tf_data_format_;
  }

  /// We don't create primitive_descriptor for TensorFlow layout now.
  /// We use lazy evaluation and create it only when needed. Input format can
  /// also be Blocked format.
  inline void SetTfLayout(size_t dims, const memory::dims& sizes,
                          MKL_TENSOR_FORMAT format) {
    DCHECK_EQ(dims, sizes.size())
        << "SetTfLayout: Number of dimensions does not"
           "match with dimension array";
    data_.dimension_ = dims;
    for (size_t ii = 0; ii < dims; ++ii) {
      data_.sizes_[ii] = sizes[ii];
    }
    data_.tf_data_format_ = format;
    if (format != MKL_TENSOR_FORMAT_BLOCKED) {
      if (dims == 2) {
        data_.map_[0] = MklDnnDims::Dim_N;
        data_.map_[1] = MklDnnDims::Dim_C;
      } else {
        SetTfDimOrder(dims, format);
      }
    }
  }

  inline const memory::desc GetTfLayout() const {
    memory::dims dims;
    for (size_t ii = 0; ii < data_.dimension_; ++ii) {
      dims.push_back(data_.sizes_[ii]);
    }

    // Create Blocked memory desc if input TF format was set like that.
    if (data_.tf_data_format_ == MKL_TENSOR_FORMAT_BLOCKED) {
      auto strides = CalculateTFStrides(dims);
      mkldnn_memory_desc_t blocked_md;
      TF_CHECK_OK(
          CreateBlockedMemDescHelper(dims, strides, data_.T_, &blocked_md));
      return memory::desc(blocked_md);
    } else {
#ifdef ENABLE_MKLDNN_V1
      auto format_tag =
          MklTensorFormatToMklDnnDataFormat(data_.tf_data_format_);
      DCHECK_NE(format_tag, memory::format_tag::undef);
      return memory::desc(dims, data_.T_, format_tag);
#else
      return memory::desc(dims, data_.T_, data_.tf_data_format_);
#endif  // ENABLE_MKLDNN_V1
    }
  }

  inline const memory::desc GetCurLayout() const {
    return IsMklTensor() ? GetMklLayout() : GetTfLayout();
  }

  // We don't need a case of default dimension order because
  // when an operator that does not get data_format attribute gets all inputs
  // in Tensorflow format, it will produce output in Tensorflow format.
  inline void SetTfDimOrder(const size_t dimension, const mkldnn_dims_t map) {
    CHECK(dimension == data_.dimension_);
    for (size_t ii = 0; ii < dimension; ii++) {
      data_.map_[ii] = map[ii];
    }
  }

  inline void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
    if (dimension == 5) {
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<3>(data_format, '0')] =
          MklDnnDims3D::Dim3d_D;
      data_.map_[GetTensorDimIndex<3>(data_format, '1')] =
          MklDnnDims3D::Dim3d_H;
      data_.map_[GetTensorDimIndex<3>(data_format, '2')] =
          MklDnnDims3D::Dim3d_W;
      data_.map_[GetTensorDimIndex<3>(data_format, 'C')] =
          MklDnnDims3D::Dim3d_C;
      data_.map_[GetTensorDimIndex<3>(data_format, 'N')] =
          MklDnnDims3D::Dim3d_N;
    } else {
      CHECK_EQ(dimension, 4);
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<2>(data_format, 'W')] = MklDnnDims::Dim_W;
      data_.map_[GetTensorDimIndex<2>(data_format, 'H')] = MklDnnDims::Dim_H;
      data_.map_[GetTensorDimIndex<2>(data_format, 'C')] = MklDnnDims::Dim_C;
      data_.map_[GetTensorDimIndex<2>(data_format, 'N')] = MklDnnDims::Dim_N;
    }
  }

  inline void SetTfDimOrder(const size_t dimension, MKL_TENSOR_FORMAT format) {
    TensorFormat data_format = MklDnnDataFormatToTFDataFormat(format);
    SetTfDimOrder(dimension, data_format);
  }

  inline const mkldnn_dim_t* GetTfToMklDimMap() const { return &data_.map_[0]; }
  inline size_t TfDimIdx(int index) const { return data_.map_[index]; }
  inline int64 TfDimSize(int index) const {
    return data_.sizes_[TfDimIdx(index)];
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Channel dimension.
  inline bool IsMklChannelDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_C;
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Batch dimension.
  inline bool IsMklBatchDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_N;
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Width dimension.
  inline bool IsMklWidthDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_W;
  }
  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Height dimension.
  inline bool IsMklHeightDim(int d) const {
    return TfDimIdx(d) == MklDnnDims::Dim_H;
  }

  /// Check if the TF-MKL dimension ordering map specifies if the input
  /// tensor is in NCHW format.
  inline bool IsTensorInNCHWFormat() const {
    TensorFormat data_format = FORMAT_NCHW;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  /// Check if the TF-MKL dimension ordering map specifies if the input
  /// tensor is in NHWC format.
  inline bool IsTensorInNHWCFormat() const {
    TensorFormat data_format = FORMAT_NHWC;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  /// The following methods are used for serializing and de-serializing the
  /// contents of the mklshape object.
  /// The data is serialized in this order
  /// is_mkl_tensor_ : dimension_ : sizes_ : map_: format_ : T_ : mkl_pd_;

  /// Size of buffer to hold the serialized object, the size is computed by
  /// following above mentioned order
  inline size_t GetSerializeBufferSize() const { return sizeof(MklShapeData); }

  void SerializeMklDnnShape(unsigned char* buf, size_t buf_size) const {
    CHECK(buf_size >= GetSerializeBufferSize())
        << "Buffer size is too small to SerializeMklDnnShape";
    *reinterpret_cast<MklShapeData*>(buf) = data_;
  }

  void DeSerializeMklDnnShape(const unsigned char* buf, size_t buf_size) {
    // Make sure buffer holds at least is_mkl_tensor_.
    CHECK(buf_size >= sizeof(data_.is_mkl_tensor_))
        << "Buffer size is too small in DeSerializeMklDnnShape";

    const bool is_mkl_tensor = *reinterpret_cast<const bool*>(buf);
    if (is_mkl_tensor) {  // If it is an MKL Tensor then read the rest
      CHECK(buf_size >= GetSerializeBufferSize())
          << "Buffer size is too small in DeSerializeMklDnnShape";
      data_ = *reinterpret_cast<const MklShapeData*>(buf);
    }
  }
};

// List of MklShape objects. Used in Concat/Split layers.
typedef std::vector<MklDnnShape> MklDnnShapeList;

template <typename T>
class MklDnnData;

inline void ExecutePrimitive(const std::vector<primitive>& net,
                             const std::vector<MemoryArgsMap>* net_args,
                             const engine& cpu_engine) {
#ifdef ENABLE_MKLDNN_V1
  DCHECK(net_args);
  DCHECK_EQ(net.size(), net_args->size());
  stream cpu_stream(cpu_engine);
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(cpu_stream, net_args->at(i));
  }
  cpu_stream.wait();
#else
  stream(stream::kind::eager).submit(net).wait();
#endif  // ENABLE_MKLDNN_V1
}

template <typename T>
inline Status ConvertMklToTF(OpKernelContext* context,
                             const Tensor& input_mkl_tensor,
                             const MklDnnShape& input_mkl_shape,
                             Tensor* output_tf_tensor) {
  try {
    if (!input_mkl_shape.IsMklTensor()) {
      // Return input as is since it is already a TF tensor
      *output_tf_tensor = input_mkl_tensor;
      return Status::OK();
    }

    // Allocate output tensor.
    TensorShape output_tf_shape = input_mkl_shape.GetTfShape();
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<T>::v(), output_tf_shape,
                                       output_tf_tensor));

    engine cpu_engine(ENGINE_CPU, 0);
    MklDnnData<T> input(&cpu_engine);

    // Get MKL layout of input tensor.
    auto input_mkl_md = input_mkl_shape.GetMklLayout();
    auto output_tf_md = input_mkl_shape.GetTfLayout();
#ifndef ENABLE_MKLDNN_V1
    // Memory primitive descriptor is deprecated in MKL-DNN v1.x.
    auto output_tf_pd = memory::primitive_desc(output_tf_md, cpu_engine);
#endif  // !ENABLE_MKLDNN_V1
    input.SetUsrMem(input_mkl_md, &input_mkl_tensor);

    if (input.IsReorderNeeded(OUTPUT_TF_MD)) {
      std::vector<primitive> net;
      std::vector<MemoryArgsMap> net_args;
      bool status = input.CheckReorderToOpMem(GET_CHECK_REORDER_TO_OP_MEM_ARGS(
          OUTPUT_TF_MD, output_tf_tensor, net, net_args, cpu_engine));
      if (!status) {
        return Status(error::Code::INTERNAL,
                      "ConvertMklToTF(): Failed to create reorder for input");
      }
      ExecutePrimitive(net, NET_ARGS_PTR, cpu_engine);
    } else {
      // If not, just forward input tensor to output tensor.
      bool status =
          output_tf_tensor->CopyFrom(input_mkl_tensor, output_tf_shape);
      if (!status) {
        return Status(
            error::Code::INTERNAL,
            "ConvertMklToTF(): Failed to forward input tensor to output");
      }
    }
    return Status::OK();
  } catch (mkldnn::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    LOG(FATAL) << "Operation received an exception: " << error_msg;
  }
}

// Get the MKL shape from the second string tensor
inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape,
                        bool eager_mode) {
  if (!eager_mode) {
    mklshape->DeSerializeMklDnnShape(
        ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
            .flat<uint8>()
            .data(),
        ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
                .flat<uint8>()
                .size() *
            sizeof(uint8));
  } else {
    mklshape->SetMklTensor(false);
  }
}

inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape) {
  GetMklShape(ctext, n, mklshape, false);
}

// Gets the actual input
inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
  return ctext->input(GetTensorDataIndex(n, ctext->num_inputs()));
}

inline void GetMklInputList(OpKernelContext* ctext, StringPiece name,
                            OpInputList* input_tensors) {
  CHECK_NOTNULL(input_tensors);
  TF_CHECK_OK(ctext->input_list(name, input_tensors));
}

inline void GetMklShapeList(OpKernelContext* ctext, StringPiece name,
                            MklDnnShapeList* mkl_shapes) {
  OpInputList input_mkl_tensors;
  GetMklInputList(ctext, strings::StrCat("mkl_", name), &input_mkl_tensors);

  for (int i = 0; i < input_mkl_tensors.size(); i++) {
    (*mkl_shapes)[i].DeSerializeMklDnnShape(
        input_mkl_tensors[i].flat<uint8>().data(),
        input_mkl_tensors[i].flat<uint8>().size() * sizeof(uint8));
  }
}

/// Get shape of input tensor pointed by 'input_idx' in TensorShape format.
/// If the input tensor is in MKL layout, then obtains TensorShape from
/// MklShape.
inline TensorShape GetTfShape(OpKernelContext* context, size_t input_idx,
                              bool eager_mode = false) {
  // Sanity check.
  CHECK_NOTNULL(context);
  CHECK_LT(input_idx, context->num_inputs());

  MklDnnShape input_mkl_shape;
  GetMklShape(context, input_idx, &input_mkl_shape, eager_mode);
  if (input_mkl_shape.IsMklTensor() && !eager_mode) {
    return input_mkl_shape.GetTfShape();
  } else {
    const Tensor& t = MklGetInput(context, input_idx);
    return t.shape();
  }
}

// Allocate the second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      const MklDnnShape& mkl_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
  OP_REQUIRES_OK(ctext, ctext->allocate_output(
                            GetTensorMetaDataIndex(n, ctext->num_outputs()),
                            second_shape, &second_tensor));
  mkl_shape.SerializeMklDnnShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}

// Allocate the output tensor, create a second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      Tensor** output,
                                      const TensorShape& tf_shape,
                                      const MklDnnShape& mkl_shape,
                                      bool eager_mode = false) {
  OP_REQUIRES_OK(
      ctext, ctext->allocate_output(GetTensorDataIndex(n, ctext->num_outputs()),
                                    tf_shape, output));
  if (!eager_mode) {
    Tensor* second_tensor = nullptr;
    TensorShape second_shape;
    second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
    OP_REQUIRES_OK(ctext, ctext->allocate_output(
                              GetTensorMetaDataIndex(n, ctext->num_outputs()),
                              second_shape, &second_tensor));
    mkl_shape.SerializeMklDnnShape(
        second_tensor->flat<uint8>().data(),
        second_tensor->flat<uint8>().size() * sizeof(uint8));
  }
}

// Allocates a temp tensor and returns the data buffer for temporary storage.
template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
                           const MEMORY_PRIMITIVE_DESC& pd, void** buf_out) {
  TensorShape tf_shape;

  tf_shape.AddDim(pd.get_size() / sizeof(T) + 1);
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                 tf_shape, tensor_out));
  *buf_out = static_cast<void*>(tensor_out->flat<T>().data());
}

template <typename T>
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
                           TensorShape tf_shape) {
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                 tf_shape, tensor_out));
}

inline void GetStridesFromSizes(TENSOR_FORMAT data_format, size_t* strides,
                                const size_t* sizes) {
#ifdef ENABLE_MKLDNN_V1
  DCHECK_NE(data_format, MklTensorFormat::FORMAT_INVALID);
#endif  // ENABLE_MKLDNN_V1
  // MKL requires strides in NCHW
  if (data_format == TENSOR_FORMAT_NHWC) {
    strides[0] = sizes[2];
    strides[1] = sizes[0] * sizes[2];
    strides[2] = 1;
    strides[3] = sizes[0] * sizes[1] * sizes[2];
  } else {
    strides[0] = 1;
    strides[1] = sizes[0];
    strides[2] = sizes[0] * sizes[1];
    strides[3] = sizes[0] * sizes[1] * sizes[2];
  }
}

inline void CopyMklTensorInToOut(OpKernelContext* context, int idx_in,
                                 int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);

  const Tensor& data = context->input(idx_data_in);
  const Tensor& meta = context->input(idx_meta_in);
  Tensor output(data.dtype());
  Tensor meta_output(meta.dtype());

  // TODO(intel_tf): alternatively, call forward_input_to_output_with_shape(...)
  CHECK(output.CopyFrom(data, data.shape()));
  CHECK(meta_output.CopyFrom(meta, meta.shape()));
  context->set_output(idx_data_out, output);
  context->set_output(idx_meta_out, meta_output);
}

inline void CopyTfTensorInToOutWithShape(OpKernelContext* context, int idx_in,
                                         int idx_out,
                                         const TensorShape& shape) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

  const Tensor& data = context->input(idx_data_in);
  MklDnnShape mkl_shape_output;
  mkl_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
  Tensor output(data.dtype());
  // TODO(intel_tf): alternatively, call forward_input_to_output_with_shape(...)
  CHECK(output.CopyFrom(data, shape));
  context->set_output(idx_data_out, output);
}

inline void ForwardTfTensorInToOut(OpKernelContext* context, int idx_in,
                                   int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

  MklDnnShape dnn_shape_output;
  dnn_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_out, dnn_shape_output);
  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
  } else {
    context->set_output(idx_data_out, context->input(idx_data_in));
  }
}

inline void ForwardMklTensorInToOut(OpKernelContext* context, int idx_in,
                                    int idx_out) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_meta_in = GetTensorMetaDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);
  int idx_meta_out = GetTensorMetaDataIndex(idx_out, num_outputs);

  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
    context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
  } else {
    context->set_output(idx_data_out, context->input(idx_data_in));
    context->set_output(idx_meta_out, context->input(idx_meta_in));
  }
}

// Set a dummy MKLDNN shape (called when the output is in TF format)
inline void SetDummyMklDnnShapeOutput(OpKernelContext* context,
                                      uint32 idx_data_out) {
  MklDnnShape mkl_shape_output;
  mkl_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_data_out, mkl_shape_output);
}

inline void ForwardMklTensorInToOutWithMklShape(OpKernelContext* context,
                                                int idx_in, int idx_out,
                                                const MklDnnShape& mkl_shape) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

  AllocateOutputSetMklShape(context, idx_out, mkl_shape);

  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_data_in, idx_data_out);
  } else {
    context->set_output(idx_data_out, context->input(idx_data_in));
  }
}

// Forward the MKL shape ONLY (used in elementwise and other ops where
// we call the eigen implementation and MKL shape is not used)
inline void ForwardMklMetaDataInToOut(OpKernelContext* context,
                                      uint32 idx_data_in,
                                      uint32_t idx_data_out) {
  uint32 idx_meta_in =
      GetTensorMetaDataIndex(idx_data_in, context->num_inputs());
  uint32 idx_meta_out =
      GetTensorMetaDataIndex(idx_data_out, context->num_outputs());

  if (IsRefType(context->input_dtype(idx_data_in))) {
    context->forward_ref_input_to_ref_output(idx_meta_in, idx_meta_out);
  } else {
    context->set_output(idx_meta_out, context->input(idx_meta_in));
  }
}

// -------------------------------------------------------------------

/// Return MKL-DNN data type (memory::data_type) for input type T
///
/// @input None
/// @return memory::data_type corresponding to type T
template <typename T>
static memory::data_type MklDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
memory::data_type MklDnnType<float>() {
  return memory::data_type::f32;
}

template <>
memory::data_type MklDnnType<quint8>() {
  return memory::data_type::u8;
}

template <>
memory::data_type MklDnnType<qint8>() {
  return memory::data_type::s8;
}

template <>
memory::data_type MklDnnType<qint32>() {
  return memory::data_type::s32;
}
template <>
memory::data_type MklDnnType<bfloat16>() {
#ifdef ENABLE_INTEL_MKL_BFLOAT16
  return memory::data_type::bf16;
#else
  return memory::data_type::f32;
#endif
}

// Map MklTensorFormat to MKL-DNN format tag
//
// @input: MklTensorFormat i.e. TensorFlow data format
// @return: MKL-DNN's memory format tag corresponding to MklTensorFormat.
//          Fails with an error if invalid data format.
inline MEMORY_FORMAT MklTensorFormatToMklDnnDataFormat(
    MKL_TENSOR_FORMAT format) {
#ifdef ENABLE_MKLDNN_V1
  if (format == MklTensorFormat::FORMAT_NHWC) return MEMORY_FORMAT::nhwc;
  if (format == MklTensorFormat::FORMAT_NCHW) return MEMORY_FORMAT::nchw;
  if (format == MklTensorFormat::FORMAT_NDHWC) return MEMORY_FORMAT::ndhwc;
  if (format == MklTensorFormat::FORMAT_NCDHW) return MEMORY_FORMAT::ncdhw;
  if (format == MklTensorFormat::FORMAT_X) return MEMORY_FORMAT::x;
  if (format == MklTensorFormat::FORMAT_NC) return MEMORY_FORMAT::nc;
  if (format == MklTensorFormat::FORMAT_TNC) return MEMORY_FORMAT::tnc;
  return MEMORY_FORMAT::undef;
#else
  return format;
#endif
}

/// Map TensorFlow data format into MKL-DNN 3D data format
/// @input: TensorFlow data format
/// @return: MKL-DNN 3D data format corresponding to TensorFlow data format;
///          Fails with an error if invalid data format.
inline MKL_TENSOR_FORMAT TFDataFormatToMklDnn3DDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MKL_TENSOR_FORMAT_NDHWC;
  if (format == FORMAT_NCHW) return MKL_TENSOR_FORMAT_NCDHW;
  TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));
  return MKL_TENSOR_FORMAT_INVALID;
}

/// Map TensorFlow data format into MKL-DNN data format
///
/// @input: TensorFlow data format
/// @return: MKL-DNN data format corresponding to TensorFlow data format;
///          Fails with an error if invalid data format.
inline MKL_TENSOR_FORMAT TFDataFormatToMklDnnDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MKL_TENSOR_FORMAT_NHWC;
  if (format == FORMAT_NCHW) return MKL_TENSOR_FORMAT_NCHW;
  TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));
  return MKL_TENSOR_FORMAT_INVALID;
}

/// Map MKL-DNN data format into TensorFlow data format
///
/// @input: MKL-DNN data format
/// @return: Tensorflow data format corresponding to MKL-DNN data format;
///          Fails with an error if invalid data format.
inline TensorFormat MklDnnDataFormatToTFDataFormat(MKL_TENSOR_FORMAT format) {
  if (format == MKL_TENSOR_FORMAT_NHWC || format == MKL_TENSOR_FORMAT_NDHWC)
    return FORMAT_NHWC;
  if (format == MKL_TENSOR_FORMAT_NCHW || format == MKL_TENSOR_FORMAT_NCDHW)
    return FORMAT_NCHW;
  TF_CHECK_OK(Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));

  // Return to prevent compiler warnings, otherwise TF_CHECK_OK will ensure
  // that we don't come here.
  return FORMAT_NHWC;
}

/// Map TensorShape object into memory::dims required by MKL-DNN
///
/// This function will simply map input TensorShape into MKL-DNN dims
/// naively. So it will preserve the order of dimensions. E.g., if
/// input tensor is in NHWC format, then dims will be in NHWC format also.
///
/// @input TensorShape object in shape
/// @return memory::dims corresponding to TensorShape
inline memory::dims TFShapeToMklDnnDims(const TensorShape& shape) {
  memory::dims dims(shape.dims());
  for (int d = 0; d < shape.dims(); ++d) {
    dims[d] = shape.dim_size(d);
  }
  return dims;
}

/// Map TensorShape object into memory::dims in NCHW format required by MKL-DNN
///
/// This function is a specific one than above function. It will map input
/// TensorShape into MKL-DNN dims in NCHW format. So it may not preserve the
/// order of dimensions. E.g., if input tensor is in NHWC format, then dims
/// will be in NCHW format, and not in NHWC format.
///
/// @input TensorShape object in shape
/// @return memory::dims in MKL-DNN required NCHW format
inline memory::dims TFShapeToMklDnnDimsInNCHW(const TensorShape& shape,
                                              TensorFormat format) {
  // Check validity of format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format), MKL_TENSOR_FORMAT_INVALID);

  int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
  int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
  int w = shape.dim_size(GetTensorDimIndex(format, 'W'));

  // MKL-DNN requires dimensions in NCHW format.
  return memory::dims({n, c, h, w});
}

inline memory::dims TFShapeToMklDnnDimsInNCDHW(const TensorShape& shape,
                                               TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnn3DDataFormat(format),
            MKL_TENSOR_FORMAT_INVALID);

  int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
  int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
  int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
  int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));

  // MKL-DNN requires dimensions in NCDHW format.
  return memory::dims({n, c, d, h, w});
}

/// Overloaded version of function TFShapeToMklDnnDimsInNCHW above.
/// Input parameters are self-explanatory.
inline memory::dims MklDnnDimsInNCHW(const memory::dims& in_dims,
                                     TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format), MKL_TENSOR_FORMAT_INVALID);

  int n = in_dims[GetTensorDimIndex(format, 'N')];
  int c = in_dims[GetTensorDimIndex(format, 'C')];
  int h = in_dims[GetTensorDimIndex(format, 'H')];
  int w = in_dims[GetTensorDimIndex(format, 'W')];

  // MKL-DNN requires dimensions in NCHW format.
  return memory::dims({n, c, h, w});
}

/// Overloaded version of function TFShapeToMklDnnDimsInNCDHW above.
/// Input parameters are self-explanatory.
inline memory::dims MklDnnDimsInNCDHW(const memory::dims& in_dims,
                                      TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format), MKL_TENSOR_FORMAT_INVALID);

  int n = in_dims[GetTensorDimIndex<3>(format, 'N')];
  int c = in_dims[GetTensorDimIndex<3>(format, 'C')];
  int d = in_dims[GetTensorDimIndex<3>(format, '0')];
  int h = in_dims[GetTensorDimIndex<3>(format, '1')];
  int w = in_dims[GetTensorDimIndex<3>(format, '2')];

  // MKL DNN requires dimensions in NCDHW format.
  return memory::dims({n, c, d, h, w});
}

/// Map MklDnn memory::dims object into TensorShape object.
///
/// This function will simply map input shape in MKL-DNN memory::dims format
/// in Tensorflow's TensorShape object by preserving dimension order.
///
/// @input MKL-DNN memory::dims object
/// @output TensorShape corresponding to memory::dims
inline TensorShape MklDnnDimsToTFShape(const memory::dims& dims) {
  std::vector<int32> shape(dims.size(), -1);
  for (int d = 0; d < dims.size(); d++) {
    shape[d] = dims[d];
  }

  TensorShape ret;
  CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
  return ret;
}

/// Function to calculate strides given tensor shape in Tensorflow order
/// E.g., if dims_tf_order is {1, 2, 3, 4}, then as per Tensorflow convention,
/// dimension with size 1 is outermost dimension; while dimension with size 4 is
/// innermost dimension. So strides for this tensor would be {4 * 3 * 2,
/// 4 * 3, 4, 1}, i.e., {24, 12, 4, 1}.
///
/// @input Tensorflow shape in memory::dims type
/// @return memory::dims containing strides for the tensor.
inline memory::dims CalculateTFStrides(const memory::dims& dims_tf_order) {
  CHECK_GT(dims_tf_order.size(), 0);
  memory::dims strides(dims_tf_order.size());
  int last_dim_idx = dims_tf_order.size() - 1;
  strides[last_dim_idx] = 1;
  for (int d = last_dim_idx - 1; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

inline padding_kind TFPaddingToMklDnnPadding(Padding pad) {
  // MKL-DNN only supports zero padding.
  return padding_kind::zero;
}

/// Helper function to create memory descriptor in Blocked format
///
/// @input: Tensor dimensions
/// @input: strides corresponding to dimensions. One can use utility
///         function such as CalculateTFStrides to compute strides
///         for given dimensions.
/// @output: mkldnn_memory_desc_t object corresponding to blocked memory
///          format for given dimensions and strides.
/// @return: Status indicating whether the blocked memory descriptor
///          was successfully created.
inline Status CreateBlockedMemDescHelper(const memory::dims& dim,
                                         const memory::dims& strides,
                                         memory::data_type dtype,
                                         mkldnn_memory_desc_t* blocked_md) {
  DCHECK_EQ(dim.size(), strides.size());
#ifdef ENABLE_MKLDNN_V1
  const int kNumDims = dim.size();
  mkldnn_dim_t input_dims[kNumDims];
  mkldnn_dim_t input_strides[kNumDims];
  for (int i = 0; i < kNumDims; ++i) {
    input_dims[i] = dim[i];
    input_strides[i] = strides[i];
  }
  try {
    mkldnn_memory_desc_init_by_strides(blocked_md, kNumDims, input_dims,
                                       memory::convert_to_c(dtype),
                                       input_strides);
  } catch (mkldnn::error& e) {
    return Status(error::Code::INTERNAL,
                  tensorflow::strings::StrCat(
                      "Failed to create blocked memory descriptor.",
                      "Status: ", e.status, ", message: ", e.message));
  }
#else
  // We have to construct memory descriptor in a C style. This is not at all
  // ideal but MKL-DNN does not offer any API to construct descriptor in
  // blocked format except a copy constructor that accepts
  // mkldnn_memory_desc_t.
  blocked_md->primitive_kind = mkldnn_memory;
  blocked_md->ndims = dim.size();
  blocked_md->format = mkldnn_blocked;
  blocked_md->data_type = memory::convert_to_c(dtype);

  for (size_t i = 0; i < dim.size(); i++) {
    blocked_md->layout_desc.blocking.block_dims[i] = 1;
    blocked_md->layout_desc.blocking.strides[1][i] = 1;
    blocked_md->layout_desc.blocking.strides[0][i] = strides[i];
    blocked_md->layout_desc.blocking.padding_dims[i] = dim[i];
    blocked_md->layout_desc.blocking.offset_padding_to_data[i] = 0;
    blocked_md->dims[i] = dim[i];
  }
  blocked_md->layout_desc.blocking.offset_padding = 0;
#endif  // ENABLE_MKLDNN_V1
  return Status::OK();
}

inline void CreateAndExecuteReorder(const ReorderPd& reorder_desc,
                                    const memory& src_mem,
                                    const memory& dst_mem,
                                    const engine& engine) {
  std::vector<primitive> net;
#ifdef ENABLE_MKLDNN_V1
  net.push_back(mkldnn::reorder(reorder_desc));
  std::vector<MemoryArgsMap> net_args;
  net_args.push_back({{MKLDNN_ARG_FROM, src_mem}, {MKLDNN_ARG_TO, dst_mem}});
#else
  net.push_back(mkldnn::reorder(reorder_desc, src_mem, dst_mem));
#endif  // ENABLE_MKLDNN_V1
  ExecutePrimitive(net, NET_ARGS_PTR, engine);
}

template <typename T>
inline primitive FindOrCreateReorder(const memory* from, const memory* to);

// Class to represent all the resources corresponding to a tensor in TensorFlow
// that are required to execute an operation (such as Convolution).
template <typename T>
class MklDnnData {
 private:
  /// MKL-DNN memory primitive for input user memory
  memory* user_memory_;

  /// MKL-DNN memory primitive in case input or output reorder is needed.
  memory* reorder_memory_;

  /// Operations memory descriptor
  memory::desc* op_md_;
  // flat to indicate if data is 3D or not.
  bool bIs3D;
  /// Operations temp buffer
  void* allocated_buffer_;
  /// CPU engine on which operation will be executed
  const engine* cpu_engine_;

 public:
  explicit MklDnnData(const engine* e)
      : user_memory_(nullptr),
        reorder_memory_(nullptr),
        op_md_(nullptr),
        bIs3D(false),
        allocated_buffer_(nullptr),
        cpu_engine_(e) {}

  ~MklDnnData() {
    if (allocated_buffer_ != nullptr) {
      cpu_allocator()->DeallocateRaw(allocated_buffer_);
    }
    cpu_engine_ = nullptr;  // We don't own this.
    delete (user_memory_);
    delete (reorder_memory_);
    delete (op_md_);
  }

  inline void* GetTensorBuffer(const Tensor* tensor) const {
    CHECK_NOTNULL(tensor);
    return const_cast<void*>(
        static_cast<const void*>(tensor->flat<T>().data()));
  }

  void SetIs3DData(bool bIs3D_) { bIs3D = bIs3D_; }
  bool GetIs3D() { return bIs3D; }

  /// Set user memory primitive using specified dimensions, memory format tag
  /// and data_buffer. Function automatically uses element data type by using
  /// input type T used for creating call object.
  ///
  /// In a nutshell, function allows user to describe the input tensor to
  /// an operation. E.g., filter of Conv2D is of shape {1, 2, 3, 4}, and
  /// memory format tag HWIO, and the buffer that contains actual values is
  /// pointed by data_buffer.
  inline void SetUsrMem(const memory::dims& dim, MEMORY_FORMAT fm,
                        void* data_buffer = nullptr) {
    auto md = memory::desc(dim, MklDnnType<T>(), fm);
    SetUsrMem(md, data_buffer);
  }

  inline void SetUsrMem(const memory::dims& dim, MEMORY_FORMAT fm,
                        const Tensor* tensor) {
    DCHECK(tensor);
    SetUsrMem(dim, fm, GetTensorBuffer(tensor));
  }

  /// Helper function to create memory descriptor in Blocked format
  ///
  /// @input: Tensor dimensions
  /// @input: strides corresponding to dimensions. One can use utility
  ///         function such as CalculateTFStrides to compute strides
  ///         for given dimensions.
  /// @return: memory::desc object corresponding to blocked memory format
  ///          for given dimensions and strides.
  static inline memory::desc CreateBlockedMemDesc(const memory::dims& dim,
                                                  const memory::dims& strides) {
    mkldnn_memory_desc_t blocked_md;
    TF_CHECK_OK(
        CreateBlockedMemDescHelper(dim, strides, MklDnnType<T>(), &blocked_md));
    return memory::desc(blocked_md);
  }

  /// A version of SetUsrMem call that allows user to create memory in blocked
  /// format. So in addition to accepting dimensions, it also accepts strides.
  /// This allows user to create memory for tensor in a format that is not
  /// supported by MKLDNN. E.g., MKLDNN does not support tensor format for 6
  /// dimensional tensor as a native format. But by using blocked format, a user
  /// can create memory for 6D tensor.
  inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
                        void* data_buffer = nullptr) {
    CHECK_EQ(dim.size(), strides.size());
    auto blocked_md = MklDnnData<T>::CreateBlockedMemDesc(dim, strides);
    SetUsrMem(blocked_md, data_buffer);
  }

  inline void SetUsrMem(const memory::dims& dim, const memory::dims& strides,
                        const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(dim, strides, GetTensorBuffer(tensor));
  }

#ifndef ENABLE_MKLDNN_V1
  /// Memory primitive descriptor is deprecated in MKL-DNN v1.x.
  /// A version of function to set user memory primitive that accepts memory
  /// descriptor directly, instead of accepting dimensions and format. This
  /// function is more generic that the one above, but the function above is
  /// sufficient in most cases.
  inline void SetUsrMem(const memory::desc& md, void* data_buffer = nullptr) {
    auto pd = memory::primitive_desc(md, *cpu_engine_);
    SetUsrMem(pd, data_buffer);
  }
#endif  // !ENABLE_MKLDNN_V1

  /// A version of SetUsrMem with memory descriptor and tensor
  inline void SetUsrMem(const memory::desc& md, const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(md, GetTensorBuffer(tensor));
  }

  /// A version of function to set user memory type that accepts memory
  /// descriptor directly, instead of accepting dimensions and format. This
  /// function is more generic than the one above, but the function above is
  /// sufficient in most cases.
  inline void SetUsrMem(const MEMORY_PRIMITIVE_DESC& pd,
                        void* data_buffer = nullptr) {
    DCHECK(cpu_engine_);
    if (user_memory_) delete user_memory_;
    // TODO(nhasabni): can we remove dynamic memory allocation?
    if (data_buffer) {
      user_memory_ = new MEMORY_CONSTRUCTOR(pd, *cpu_engine_, data_buffer);
    } else {
      user_memory_ = new MEMORY_CONSTRUCTOR_WITHOUT_DATA(pd, *cpu_engine_);
    }
  }

#ifndef ENABLE_MKLDNN_V1
  /// Memory primitive descriptor is deprecated in MKL-DNN v1.x
  /// A version of SetUsrMem with primitive descriptor and tensor
  inline void SetUsrMem(const memory::primitive_desc& pd,
                        const Tensor* tensor) {
    DCHECK(tensor);
    SetUsrMem(pd, GetTensorBuffer(tensor));
  }
#endif  // !ENABLE_MKLDNN_V1

  /// Get function for user memory primitive.
  inline const memory* GetUsrMem() const { return user_memory_; }

#ifndef ENABLE_MKLDNN_V1
  /// Memory primitive descriptor is deprecated in MKL-DNN v1.x.
  /// Get function for primitive descriptor of user memory primitive.
  inline const memory::primitive_desc GetUsrMemPrimDesc() const {
    DCHECK(user_memory_);
    return user_memory_->get_primitive_desc();
  }
#endif  // !ENABLE_MKLDNN_V1

  /// Get function for descriptor of user memory.
  inline memory::desc GetUsrMemDesc() const {
#ifdef ENABLE_MKLDNN_V1
    DCHECK(user_memory_);
    return user_memory_->get_desc();
#else
    // This is ugly. Why MKL-DNN does not provide desc() method of const type??
    const memory::primitive_desc pd = GetUsrMemPrimDesc();
    return const_cast<memory::primitive_desc*>(&pd)->desc();
#endif  // ENABLE_MKLDNN_V1
  }

  /// Get function for data buffer of user memory primitive.
  inline void* GetUsrMemDataHandle() const {
    CHECK_NOTNULL(user_memory_);
    return user_memory_->get_data_handle();
  }

  /// Set function for data buffer of user memory primitive.
  inline void SetUsrMemDataHandle(void* data_buffer) {
    CHECK_NOTNULL(user_memory_);
    CHECK_NOTNULL(data_buffer);
    user_memory_->set_data_handle(data_buffer);
  }

  /// Set function for data buffer of user memory primitive.
  inline void SetUsrMemDataHandle(const Tensor* tensor) {
    CHECK_NOTNULL(user_memory_);
    CHECK_NOTNULL(tensor);
    user_memory_->set_data_handle(GetTensorBuffer(tensor));
  }

  /// allocate function for data buffer
  inline void AllocateBuffer(size_t size) {
    const int64 kMemoryAlginment = 64;  // For AVX512 memory alignment.
    allocated_buffer_ = cpu_allocator()->AllocateRaw(kMemoryAlginment, size);
  }

  inline void* GetAllocatedBuffer() { return allocated_buffer_; }

  /// Get the memory primitive for input and output of an op. If inputs
  /// to an op require reorders, then this function returns memory primitive
  /// for reorder. Otherwise, it will return memory primitive for user memory.
  ///
  /// E.g., Conv2D(I, F) is a primitive with I and F being inputs. Then to
  /// execute Conv2D, we need memory primitive for I and F. Buf if reorder is
  /// required for I and F (say I_r is reorder primitive for I; F_r is reorder
  /// primitive for F), then we need I_r and F_r to perform Conv2D.
  inline const memory& GetOpMem() const {
    return reorder_memory_ ? *reorder_memory_ : *user_memory_;
  }

  /// Set memory descriptor of an operation in terms of dimensions and memory
  /// format. E.g., For Conv2D, the dimensions would be same as user dimensions
  /// but memory::format_tag would be mkldnn::any because we want MKL-DNN to
  /// choose the best layout/format for given input dimensions.
  inline void SetOpMemDesc(const memory::dims& dim, MEMORY_FORMAT fm) {
    // TODO(nhasabni): can we remove dynamic memory allocation?
    op_md_ = new memory::desc(dim, MklDnnType<T>(), fm);
  }

  /// Get function for memory descriptor for an operation
  inline const memory::desc& GetOpMemDesc() const { return *op_md_; }

  /// Predicate that checks if we need to reorder user's memory into memory
  /// pointed by op_md.
  ///
  /// @input: op_md - memory descriptor of the given input of an operation.
  /// @return: true in case reorder of input is needed; false, otherwise.
  inline bool IsReorderNeeded(const MEMORY_PRIMITIVE_DESC& op_pd) const {
    DCHECK(user_memory_);
    return op_pd != GET_MEMORY_PRIMITIVE_DESC_FROM_MEM_PTR(user_memory_);
  }

#ifndef ENABLE_MKLDNN_V1
  /// In MKL-DNN v1.x, it it is not possible to directly compare two memory
  /// format tags since they only provide a partial description of the memory
  /// layout. Hence, this function is disabled for MKL-DNN v1.x.
  ///
  /// Predicate that checks if we need to reorder user's memory into memory
  /// based on the provided format.
  ///
  /// @input: target_format - memory format of the given input of an
  ///               operation
  /// @return: true in case reorder of input is needed; false, otherwise.
  inline bool IsReorderNeeded(const memory::format& target_format) const {
    CHECK_NOTNULL(user_memory_);
    return target_format !=
           user_memory_->get_primitive_desc().desc().data.format;
  }
#endif  // !ENABLE_MKLDNN_V1

  /// Function to create a reorder from memory pointed by from to memory pointed
  /// by to. Returns created primitive.
  inline primitive CreateReorder(const memory* from, const memory* to) const {
    CHECK_NOTNULL(from);
    CHECK_NOTNULL(to);
    return reorder(*from, *to);
  }

/// Function to handle input reordering
///
/// Check if we need to reorder this input of an operation.
/// Return true and allocate reorder memory primitive if reorder is needed.
/// Otherwise, return false and do not allocate reorder memory primitive.
///
/// To check if reorder is needed, this function compares memory primitive
/// descriptor (memory descriptor for v1.x) of an operation (op_pd) for
/// the given input with the user-specified memory descriptor.
///
/// @input: op_pd - memory primitive descriptor of the given input of an
///                 operation
/// @input: net - net to which to add reorder primitive in case it is needed.
/// @input: net_args - net to which user and reorder memories are added if
///                    needed. Each entry is a key-value pair of the form
///                    <argument-type, mkldnn::memory>.
/// @return: true in case reorder of input is needed; false, otherwise.
#ifdef ENABLE_MKLDNN_V1
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(user_memory_);
    DCHECK_EQ(net.size(), net_args.size());
    if (IsReorderNeeded(op_md)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      reorder_memory_ = new memory(op_md, engine);
      net.push_back(CreateReorder(user_memory_, reorder_memory_));
      net_args.push_back(MemoryArgsMap{{MKLDNN_ARG_FROM, *user_memory_},
                                       {MKLDNN_ARG_TO, *reorder_memory_}});
#else
  inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
                                  std::vector<primitive>* net) {
    DCHECK(net);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      reorder_memory_ = new memory(op_pd);
      net->push_back(CreateReorder(user_memory_, reorder_memory_));
#endif  // ENABLE_MKLDNN_V1
      return true;
    }
    return false;
  }

/// TODO: this is a faster path with reorder primitive cache compared with
/// CheckReorderToOpMem(..., std::vector<primitive>* net).
/// TODO(gzmkl): Remove the slower path.
#ifdef ENABLE_MKLDNN_V1
  /// TODO(bhavanis): Need to use reorder cache here for better performance.
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  const engine& engine) {
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      // primitive reuse don't allow two same reorder prim in
      // one stream, so submit it immediately
      reorder_memory_ = new memory(op_md, engine);
      stream cpu_stream(engine);
      reorder(*user_memory_, *reorder_memory_)
          .execute(cpu_stream, *user_memory_, *reorder_memory_);
#else
  inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd) {
    CHECK_NOTNULL(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      reorder_memory_ = new memory(op_pd);
      std::vector<primitive> net;
      net.push_back(FindOrCreateReorder<T>(user_memory_, reorder_memory_));
      stream(stream::kind::eager).submit(net).wait();
#endif  // ENABLE_MKLDNN_V1
      return true;
    }
    return false;
  }

/// Overloaded version of above function that accepts memory buffer
/// where output of reorder needs to be stored.
///
/// @input: op_pd - memory primitive descriptor (memory descriptor for v1.x)
///                 of the given input of an operation
/// @reorder_data_handle - memory buffer where output of reorder needs to be
///                        stored. Primitive does not check if buffer has
///                        enough size to write.
/// @input: net - net to which to add reorder primitive in case it is needed.
/// @input: net_args - net to which user and reorder memories are added if
///                    needed. Each entry is a key-value pair of the form
///                    <argument-type, mkldnn::memory>.
/// @input: engine - MKL-DNN's abstraction of a computational device
/// @return: true in case reorder of input is needed; false, otherwise.
#ifdef ENABLE_MKLDNN_V1
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  void* reorder_data_handle,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      reorder_memory_ = new memory(op_md, engine, reorder_data_handle);
      net.push_back(CreateReorder(user_memory_, reorder_memory_));
      net_args.push_back(MemoryArgsMap{{MKLDNN_ARG_FROM, *user_memory_},
                                       {MKLDNN_ARG_TO, *reorder_memory_}});
#else
  inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
                                  void* reorder_data_handle,
                                  std::vector<primitive>* net) {
    CHECK_NOTNULL(net);
    CHECK_NOTNULL(reorder_data_handle);
    CHECK_NOTNULL(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      reorder_memory_ = new memory(op_pd, reorder_data_handle);
      net->push_back(CreateReorder(user_memory_, reorder_memory_));
#endif  // ENABLE_MKLDNN_V1
      return true;
    }
    return false;
  }

/// This is a faster path with reorder primitive cache compared with
/// CheckReorderToOpMem(..., std::vector<primitive>* net).
/// The slower path will be removed in the future
#ifdef ENABLE_MKLDNN_V1
  /// TODO(bhavanis): Need to use reorder cache here for better performance.
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  void* reorder_data_handle,
                                  const engine& engine) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      // primitive reuse don't allow two same reorder prim in
      // one stream, so submit it immediately
      reorder_memory_ = new memory(op_md, engine, reorder_data_handle);
      stream cpu_stream(engine);
      reorder(*user_memory_, *reorder_memory_)
          .execute(cpu_stream, *user_memory_, *reorder_memory_);
#else
  inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
                                  void* reorder_data_handle) {
    CHECK_NOTNULL(reorder_data_handle);
    CHECK_NOTNULL(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      std::vector<primitive> net;
      reorder_memory_ = new memory(op_pd, reorder_data_handle);
      net.push_back(FindOrCreateReorder<T>(user_memory_, reorder_memory_));
      stream(stream::kind::eager).submit(net).wait();
#endif  // ENABLE_MKLDNN_V1
      return true;
    }
    return false;
  }

/// Another overloaded version of CheckReorderToOpMem that accepts Tensor
/// where output of reorder needs to be stored.
///
/// @input: op_md - memory primitive descriptor (memory descriptor for v1.x)
///                 of the given input of an operation
/// @reorder_tensor - Tensor whose buffer is to be used to store output of
///                   reorder. Primitive does not check if buffer is
///                   enough size to write.
/// @input: net - net to which to add reorder primitive in case it is needed.
/// @input: net_args - net to which user and reorder memories are added if
///                    needed. Each entry is a key-value pair of the form
///                    <argument-type, mkldnn::memory>.
/// @input: engine - MKL-DNN's abstraction of a computational device
/// @return: true in case reorder of input is needed; false, otherwise.
#ifdef ENABLE_MKLDNN_V1
  inline bool CheckReorderToOpMem(const memory::desc& op_md,
                                  Tensor* reorder_tensor,
                                  std::vector<primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const engine& engine) {
    DCHECK(reorder_tensor);
    return CheckReorderToOpMem(op_md, GetTensorBuffer(reorder_tensor), net,
                               net_args, engine);
  }
#else
  inline bool CheckReorderToOpMem(const memory::primitive_desc& op_pd,
                                  Tensor* reorder_tensor,
                                  std::vector<primitive>* net) {
    CHECK_NOTNULL(net);
    CHECK_NOTNULL(reorder_tensor);
    return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor), net);
  }
#endif  // ENABLE_MKLDNN_V1

  /// TODO: this is a faster path with reorder primitive cache compared with
  /// CheckReorderToOpMem(op_md, reorder_tensor, net, net_args, engine), will
  /// remove
  /// slow path in the future
  inline bool CheckReorderToOpMem(const MEMORY_PRIMITIVE_DESC& op_pd,
                                  Tensor* reorder_tensor) {
    DCHECK(reorder_tensor);
#ifdef ENABLE_MKLDNN_V1
    return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor),
                               *cpu_engine_);
#else
    return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor));
#endif  // ENABLE_MKLDNN_V1
  }

  /// Function to handle output reorder
  ///
  /// This function performs very similar functionality as input reordering
  /// function above. The only difference is that this function does not add
  /// reorder primitive to the net. The reason for this is: the reorder
  /// primitive for output needs to be added to the list only after operation
  /// has executed. But we need to prepare a temporary buffer in case output
  /// reorder is needed. And this temporary buffer will hold the output of
  /// an operation before it is fed to reorder primitive.
  ///
  /// @input - memory primitive descriptor (memory descriptor for v1.x) for the
  ///          given output of an operation
  /// @return: true in case reorder of output is needed; false, otherwise.
  inline bool PrepareReorderToUserMemIfReq(const MEMORY_PRIMITIVE_DESC& op_pd) {
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_pd)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      reorder_memory_ =
          new MEMORY_CONSTRUCTOR_WITHOUT_DATA(op_pd, *cpu_engine_);
      return true;
    }
    return false;
  }

/// Function to actually insert reorder primitive in the net
///
/// This function completes remaining part of output reordering. It inserts
/// a reordering primitive from the temporary buffer that holds the output
/// to the user-specified output buffer.
///
/// @input: net - net to which to add reorder primitive
/// @input: net_args - net to which user and reorder memories are added if
///                    needed. Each entry is a key-value pair of the form
///                    <argument-type, mkldnn::memory>.
#ifdef ENABLE_MKLDNN_V1
  inline void InsertReorderToUserMem(std::vector<primitive>& net,
                                     std::vector<MemoryArgsMap>& net_args) {
    DCHECK(user_memory_);
    DCHECK(reorder_memory_);
    net.push_back(CreateReorder(reorder_memory_, user_memory_));
    net_args.push_back(MemoryArgsMap{{MKLDNN_ARG_FROM, *reorder_memory_},
                                     {MKLDNN_ARG_TO, *user_memory_}});
  }
#else
  inline void InsertReorderToUserMem(std::vector<primitive>* net) {
    CHECK_NOTNULL(net);
    CHECK_NOTNULL(user_memory_);
    CHECK_NOTNULL(reorder_memory_);
    net->push_back(CreateReorder(reorder_memory_, user_memory_));
  }
#endif  // ENABLE_MKLDNN_V1

  /// TODO: this is a faster path with reorder primitive cache compared with
  ///       InsertReorderToUserMem(net, net_args), will remove
  ///       slow path in the future
  inline void InsertReorderToUserMem() {
    DCHECK(user_memory_);
    DCHECK(reorder_memory_);
    DCHECK(cpu_engine_);
    // primitive reuse don't allow two same reorder prim in
    // one stream, so submit it immediately
    std::vector<primitive> net;
#ifdef ENABLE_MKLDNN_V1
    std::vector<MemoryArgsMap> net_args;
    // TODO(bhavanis): Need to use reorder cache here for better performance.
    net.push_back(CreateReorder(reorder_memory_, user_memory_));
    net_args.push_back(MemoryArgsMap{{MKLDNN_ARG_FROM, *reorder_memory_},
                                     {MKLDNN_ARG_TO, *user_memory_}});
#else
    net.push_back(FindOrCreateReorder<T>(reorder_memory_, user_memory_));
#endif  // ENABLE_MKLDNN_V1
    ExecutePrimitive(net, NET_ARGS_PTR, *cpu_engine_);
  }
};

/// Base class for operations with reuse of primitives
class MklPrimitive {
 public:
  virtual ~MklPrimitive() {}

  // Dummy data which MKL DNN never operates on
  unsigned char* DummyData = nullptr;
};

const mkldnn::memory::dims NONE_DIMS = {};

//
// LRUCache is a class which implements LRU (Least Recently Used) cache.
// The implementation is similar to that of
//    tensorflow/core/platform/cloud/expiring_lru_cache.h
// without its thread-safe part because the cache is supposed to be
// used as thread local (for instance, MklPrimitive caching).
//
// The LRU list maintains objects in chronological order based on
// creation time, with the least recently accessed object at the
// tail of LRU list, while the most recently accessed object
// at the head of LRU list.
//
// This class is used to maintain an upper bound on the total number of
// cached items. When the cache reaches its capacity, the LRU item will
// be removed and replaced by a new one from SetOp call.
//
template <typename T>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) {
    capacity_ = capacity;
    Clear();
  }

  T* GetOp(const string& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }

    // Move to the front of LRU list as the most recently accessed.
    lru_list_.erase(it->second.lru_iterator);
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return it->second.op;
  }

  void SetOp(const string& key, T* op) {
    if (lru_list_.size() >= capacity_) {
      Delete();
    }

    // Insert an entry to the front of the LRU list
    lru_list_.push_front(key);
    Entry entry(op, lru_list_.begin());
    cache_.emplace(std::make_pair(key, std::move(entry)));
  }

  void Clear() {
    if (lru_list_.empty()) return;

    // Clean up the cache
    cache_.clear();
    lru_list_.clear();
  }

 private:
  struct Entry {
    // The entry's value.
    T* op;

    // A list iterator pointing to the entry's position in the LRU list.
    std::list<string>::iterator lru_iterator;

    // Constructor
    Entry(T* op, std::list<string>::iterator it) {
      this->op = op;
      this->lru_iterator = it;
    }

    // Move construcctor
    Entry(Entry&& source) noexcept
        : lru_iterator(std::move(source.lru_iterator)) {
      op = std::move(source.op);
      source.op = std::forward<T*>(nullptr);
    }

    // Destructor
    ~Entry() {
      if (op != nullptr) delete op;
    }
  };

  // Remove the least recently accessed entry from LRU list, which
  // is the tail of lru_list_. Update cache_ correspondingly.
  bool Delete() {
    if (lru_list_.empty()) return false;
    string key = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(key);
    return true;
  }

  // Cache capacity
  size_t capacity_;

  // The cache, a map from string key to a LRU entry.
  std::unordered_map<string, Entry> cache_;

  // The LRU list of entries.
  // The front of the list contains the key of the most recently accessed
  // entry, while the back of the list is the least recently accessed entry.
  std::list<string> lru_list_;
};

template <typename T>
class MklPrimitiveFactory {
 public:
  MklPrimitiveFactory() {}

  ~MklPrimitiveFactory() {}

  MklPrimitive* GetOp(const string& key) {
    auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
    return lru_cache.GetOp(key);
  }

  void SetOp(const string& key, MklPrimitive* op) {
    auto& lru_cache = MklPrimitiveFactory<T>::GetLRUCache();
    lru_cache.SetOp(key, op);
  }

  /// Function to decide whether HW has AVX512 or AVX2
  /// For those legacy device(w/o AVX512 and AVX2),
  /// MKL-DNN GEMM will be used.
  static inline bool IsLegacyPlatform() {
    return (!port::TestCPUFeature(port::CPUFeature::AVX512F) &&
            !port::TestCPUFeature(port::CPUFeature::AVX2));
  }

  /// Fuction to check whether primitive memory optimization is enabled
  static inline bool IsPrimitiveMemOptEnabled() {
    bool is_primitive_mem_opt_enabled = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_MKL_OPTIMIZE_PRIMITIVE_MEMUSE", true,
                                   &is_primitive_mem_opt_enabled));
    return is_primitive_mem_opt_enabled;
  }

 private:
  static inline LRUCache<MklPrimitive>& GetLRUCache() {
    static const int kCapacity = 1024;  // cache capacity
    static thread_local LRUCache<MklPrimitive> lru_cache_(kCapacity);
    return lru_cache_;
  }
};

// utility class for creating keys of MKL primitive pool.
class FactoryKeyCreator {
 public:
  FactoryKeyCreator() { key_.reserve(kMaxKeyLength); }

  ~FactoryKeyCreator() {}

  void AddAsKey(const string& str) { Append(str); }

  void AddAsKey(const mkldnn::memory::dims& dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
      AddAsKey<int>(dims[i]);
    }
  }

  template <typename T>
  void AddAsKey(const T data) {
    auto buffer = reinterpret_cast<const char*>(&data);
    Append(StringPiece(buffer, sizeof(T)));
  }

  string GetKey() { return key_; }

 private:
  string key_;
  const char delimiter = 'x';
  const int kMaxKeyLength = 256;
  void Append(StringPiece s) {
    key_.append(string(s));
    key_.append(1, delimiter);
  }
};

class MklReorderPrimitive : public MklPrimitive {
 public:
  explicit MklReorderPrimitive(const memory* from, const memory* to) {
    Setup(from, to);
  }
  ~MklReorderPrimitive() {}

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

  void SetMemory(const memory* from, const memory* to) {
    context_.src_mem->set_data_handle(from->get_data_handle());
    context_.dst_mem->set_data_handle(to->get_data_handle());
  }

 private:
  struct ReorderContext {
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;
    std::shared_ptr<primitive> reorder_prim;
    ReorderContext()
        : src_mem(nullptr), dst_mem(nullptr), reorder_prim(nullptr) {}
  } context_;

  engine cpu_engine_ = engine(ENGINE_CPU, 0);

  void Setup(const memory* from, const memory* to) {
    context_.src_mem.reset(
        new MEMORY_CONSTRUCTOR_WITH_MEM_PD(from, cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new MEMORY_CONSTRUCTOR_WITH_MEM_PD(to, cpu_engine_, DummyData));
    context_.reorder_prim = std::make_shared<mkldnn::reorder>(
        reorder(*context_.src_mem, *context_.dst_mem));
  }
};

template <typename T>
class MklReorderPrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklReorderPrimitive* Get(const memory* from, const memory* to) {
    auto reorderPrim = static_cast<MklReorderPrimitive*>(
        MklReorderPrimitiveFactory<T>::GetInstance().GetReorder(from, to));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklReorderPrimitive(from, to);
      MklReorderPrimitiveFactory<T>::GetInstance().SetReorder(from, to,
                                                              reorderPrim);
    }
    reorderPrim->SetMemory(from, to);
    return reorderPrim;
  }

  static MklReorderPrimitiveFactory& GetInstance() {
    static MklReorderPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklReorderPrimitiveFactory() {}
  ~MklReorderPrimitiveFactory() {}

  static string CreateKey(const memory* from, const memory* to) {
    string prefix = "reorder";
    FactoryKeyCreator key_creator;
    auto const& from_desc = GET_MEMORY_DESC_FROM_MEM_PTR(from).data;
    auto const& to_desc = GET_MEMORY_DESC_FROM_MEM_PTR(to).data;
    const int KIdxFirstStride = 0;
    memory::dims from_dims(from_desc.dims, &from_desc.dims[from_desc.ndims]);
    memory::dims to_dims(to_desc.dims, &to_desc.dims[to_desc.ndims]);
    memory::dims from_strides(
#ifdef ENABLE_MKLDNN_V1
        from_desc.format_desc.blocking.strides,
        &from_desc.format_desc.blocking.strides[from_desc.ndims]);
#else
        from_desc.layout_desc.blocking.strides[KIdxFirstStride],
        &from_desc.layout_desc.blocking
             .strides[KIdxFirstStride][from_desc.ndims]);
#endif  // ENABLE_MKLDNN_V1
    memory::dims to_strides(
#ifdef ENABLE_MKLDNN_V1
        to_desc.format_desc.blocking.strides,
        &to_desc.format_desc.blocking.strides[to_desc.ndims]);
#else
        to_desc.layout_desc.blocking.strides[KIdxFirstStride],
        &to_desc.layout_desc.blocking.strides[KIdxFirstStride][to_desc.ndims]);
#endif  // ENABLE_MKLDNN_V1
    key_creator.AddAsKey(prefix);
#ifndef ENABLE_MKLDNN_V1
    // `format_kind` is not added in v1.x since it will always set to
    // `mkldnn_blocked`
    key_creator.AddAsKey(static_cast<int>(from_desc.format));
#endif  // !ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(from_desc.data_type));
    key_creator.AddAsKey(from_dims);
    key_creator.AddAsKey(from_strides);
#ifndef ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(to_desc.format));
#endif  // !ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(to_desc.data_type));
    key_creator.AddAsKey(to_dims);
    key_creator.AddAsKey(to_strides);
    return key_creator.GetKey();
  }

  MklPrimitive* GetReorder(const memory* from, const memory* to) {
    string key = CreateKey(from, to);
    return this->GetOp(key);
  }

  void SetReorder(const memory* from, const memory* to, MklPrimitive* op) {
    string key = CreateKey(from, to);
    this->SetOp(key, op);
  }
};

/// Fuction to find(or create) a reorder from memory pointed by
/// from to memory pointed by to, it will created primitive or
/// get primitive from pool if it is cached.
/// Returns the primitive.
template <typename T>
inline primitive FindOrCreateReorder(const memory* from, const memory* to) {
  CHECK_NOTNULL(from);
  CHECK_NOTNULL(to);
  MklReorderPrimitive* reorder_prim =
      MklReorderPrimitiveFactory<T>::Get(from, to);
  return *reorder_prim->GetPrimitive();
}

// utility function to determine if it is conv 1x1 and stride != 1
// for purpose of temporarily disabling primitive reuse
inline bool IsConv1x1StrideNot1(memory::dims filter_dims,
                                memory::dims strides) {
  if (filter_dims.size() != 4 || strides.size() != 2) return false;

  return ((filter_dims[2] == 1) && (filter_dims[3] == 1) &&
          ((strides[0] != 1) || (strides[1] != 1)));
}

}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
