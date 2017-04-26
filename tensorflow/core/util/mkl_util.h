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

#include <string>
#include <vector>

#include "third_party/mkl/include/mkl_dnn.h"
#include "third_party/mkl/include/mkl_dnn_types.h"
#include "third_party/mkl/include/mkl_service.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

// The file contains a number of utility classes and functions used by MKL
// enabled kernels

namespace tensorflow {

// This class encapsulates all the meta data that is associated with an MKL
// tensor. A tensor is an MKL tensor if it was created as the result of an
// MKL operation, and did not go through a conversion to a standard
// Tensorflow tensor.

typedef enum { W = 0, H = 1, C = 2, N = 3 } MklDims;

class MklShape {
 public:
  MklShape() {}
  TF_DISALLOW_COPY_AND_ASSIGN(MklShape);  // Cannot copy

  ~MklShape() {
    if (sizes_) delete[] sizes_;
    if (strides_) delete[] strides_;
    if (mklLayout_) CHECK_EQ(dnnLayoutDelete_F32(mklLayout_), E_SUCCESS);
    if (tfLayout_) CHECK_EQ(dnnLayoutDelete_F32(tfLayout_), E_SUCCESS);
    if (tf_to_mkl_dim_map_) delete[] tf_to_mkl_dim_map_;
  }

  const bool IsMklTensor() const { return isMklTensor_; }

  void SetMklTensor(const bool isMklTensor) { isMklTensor_ = isMklTensor; }

  void SetDimensions(const size_t dimension) { dimension_ = dimension; }

  void SetMklLayout(const void* primitive, size_t resourceType) {
    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(&mklLayout_, (dnnPrimitive_t)primitive,
                                         (dnnResourceType_t)resourceType),
        E_SUCCESS);
  }

  void SetTfLayout(const size_t dimension, const size_t* sizes,
                   const size_t* strides) {
    dimension_ = dimension;
    if (dimension > 0) {  // MKl doesn't support zero dimension tensors
      sizes_ = new size_t[dimension];
      strides_ = new size_t[dimension];

      for (int ii = 0; ii < dimension; ii++) {
        sizes_[ii] = sizes[ii];
        strides_[ii] = strides[ii];
      }
      CHECK_EQ(dnnLayoutCreate_F32(&tfLayout_, dimension, sizes, strides),
               E_SUCCESS);
    }
  }

  // Default case - MKL dim ordering is opposite of TF dim ordering
  // MKL -> (DIMS-1)...0 where (DIMS-1) is outermost dim and 0 is innermost dim
  // TF  -> 0...(DIMS-1) where 0 is outermost dim and (DIMS-1) is innermost dim
  // For layers that rely on data_format semantics (conv, pooling etc.)
  // or operate only on certain dimensions (relu, concat, split etc.),
  // Mkl APIs might require us to reorder these dimensions. In such cases,
  // kernels should explicitly set this map
  void SetTfDimOrder(const size_t dimension) {
    CHECK(dimension == dimension_);
    if (tf_to_mkl_dim_map_ == nullptr) {
      tf_to_mkl_dim_map_ = new size_t[dimension];
    }
    for (size_t ii = 0; ii < dimension; ii++) {
      tf_to_mkl_dim_map_[ii] = dimension - (ii + 1);
    }
  }

  void SetTfDimOrder(const size_t dimension, const size_t* tf_to_mkl_dim_map) {
    CHECK(dimension == dimension_);
    if (tf_to_mkl_dim_map_ == nullptr) {
      tf_to_mkl_dim_map_ = new size_t[dimension];
    }
    for (size_t ii = 0; ii < dimension; ii++) {
      tf_to_mkl_dim_map_[ii] = tf_to_mkl_dim_map[ii];
    }
  }

  void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
    CHECK_EQ(dimension, 4);
    CHECK(dimension == dimension_);
    if (tf_to_mkl_dim_map_ == nullptr) {
      tf_to_mkl_dim_map_ = new size_t[dimension];
    }
    tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'W')] = MklDims::W;
    tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'H')] = MklDims::H;
    tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'C')] = MklDims::C;
    tf_to_mkl_dim_map_[GetTensorDimIndex<2>(data_format, 'N')] = MklDims::N;
  }

  const dnnLayout_t GetMklLayout() const { return mklLayout_; }
  const dnnLayout_t GetTfLayout() const { return tfLayout_; }
  const dnnLayout_t GetCurLayout() const {
    return isMklTensor_ ? mklLayout_ : tfLayout_;
  }
  size_t GetDimension() const { return dimension_; }
  const size_t* GetSizes() const { return sizes_; }
  int64 dim_size(int index) const { return sizes_[index]; }
  const size_t* GetStrides() const { return strides_; }
  const size_t* GetTfToMklDimMap() const { return tf_to_mkl_dim_map_; }
  size_t tf_dim_idx(int index) const { return tf_to_mkl_dim_map_[index]; }

  // Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  // corresponds to MKL's Channel dimension.
  bool IsMklChannelDim(int d) const { return tf_dim_idx(d) == MklDims::C; }
  // Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  // corresponds to MKL's Batch dimension.
  bool IsMklBatchDim(int d) const { return tf_dim_idx(d) == MklDims::N; }
  // Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  // corresponds to MKL's Width dimension.
  bool IsMklWidthDim(int d) const { return tf_dim_idx(d) == MklDims::W; }
  // Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  // corresponds to MKL's Height dimension.
  bool IsMklHeightDim(int d) const { return tf_dim_idx(d) == MklDims::H; }

  // Check if the TF-Mkl dimension ordering map specifies if the input
  // tensor is in NCHW format.
  bool IsTensorInNCHWFormat() const {
    TensorFormat data_format = FORMAT_NCHW;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  // Check if the TF-Mkl dimension ordering map specifies if the input
  // tensor is in NHWC format.
  bool IsTensorInNHWCFormat() const {
    TensorFormat data_format = FORMAT_NHWC;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  void GetConvertedFlatData(dnnLayout_t targetLayout, void* input,
                            void* output) const {
    dnnLayout_t curLayout;
    if (isMklTensor_)
      curLayout = mklLayout_;
    else
      curLayout = tfLayout_;
    dnnPrimitive_t convert;
    CHECK_EQ(dnnConversionCreate_F32(&convert, curLayout, targetLayout),
             E_SUCCESS);
    CHECK_EQ(dnnConversionExecute_F32(convert, input, output), E_SUCCESS);
    CHECK_EQ(dnnDelete_F32(convert), E_SUCCESS);
  }

// The following methods are used for serializing and de-serializing the
// contents of the mklshape object.
// The data is serialized in this order
// isMklTensor_
// dimension_
// sizes_
// strides_
// mklLayout_
// tfLayout_
// tf_to_mkl_dim_map_

#define SIZE_OF_MKL_DNN_BUF \
  (dnnLayoutSerializationBufferSize_F32())  // Size of buffer needed to
                                            // serialize dnn_layout pointer

// Size of buffer to hold the serialized object, the size is computed as follows
// sizeof(isMklTensor_) + sizeof(dimension_) + sizeof(sizes_) + sizeof(strides_)
// + sizeof(mklLayout_ buffer) + sizeof(tfLayout_ buffer)
// + sizeof(tf_to_mkl_dim_map_)

#define SIZE_OF_MKL_SERIAL_DATA(dims) \
  (2 * sizeof(size_t) + 3 * dims * sizeof(size_t) + 2 * SIZE_OF_MKL_DNN_BUF)

// First we need to define some macro for offsets into the serial buffer where
// different elements of Mklshape is written/read from

#define IS_MKL_TENSOR_OFFSET 0
// Location from start of buffer where isMklTensor_ is serialized
#define DIMS_OFFSET \
  (IS_MKL_TENSOR_OFFSET + sizeof(size_t))  // Location of dimension_
#define SIZES_OFFSET(dims) \
  (DIMS_OFFSET +           \
   sizeof(size_t))  // Location of sizes. Note dim is not used here, left here
                    // to make macros consistent.
#define STRIDES_OFFSET(dims) \
  (SIZES_OFFSET(dims) + dims * sizeof(size_t))  // Location of strides
#define MKL_LAYOUT_OFFSET(dims) \
  (STRIDES_OFFSET(dims) + dims * sizeof(size_t))  // Location of mklLayout_
#define TF_LAYOUT_OFFSET(dims) \
  (MKL_LAYOUT_OFFSET(dims) + SIZE_OF_MKL_DNN_BUF)  // Location of tfLayout_
#define TF_TO_MKL_DIM_MAP_OFFSET(dims) \
  (TF_LAYOUT_OFFSET(dims) +            \
   SIZE_OF_MKL_DNN_BUF)  // Location of tf_to_mkl_dim_map_

  // TODO(agramesh1) make sure to create a const to share with rewrite pass
  // for min size of MKL metadata tensor.

  void DeSerializeMklShape(const unsigned char* buf, size_t buf_size) {
    CHECK(buf_size >= sizeof(size_t)) << "Bufsize too small in DeSerialize";
    // Make sure buffer holds at least  isMklTensor_
    isMklTensor_ =
        *reinterpret_cast<const size_t*>(buf + IS_MKL_TENSOR_OFFSET) != 0;

    if (isMklTensor_) {  // If it is an MKL Tensor then read the rest
      dimension_ = *(reinterpret_cast<const size_t*>(buf + DIMS_OFFSET));
      CHECK(buf_size >= SIZE_OF_MKL_SERIAL_DATA(dimension_))
          << "Bufsize too small in DeSerialize";
      sizes_ = new size_t[dimension_];
      strides_ = new size_t[dimension_];
      tf_to_mkl_dim_map_ = new size_t[dimension_];
      for (int i = 0; i < dimension_; i++) {
        sizes_[i] =
            reinterpret_cast<const size_t*>(buf + SIZES_OFFSET(dimension_))[i];
        strides_[i] = reinterpret_cast<const size_t*>(
            buf + STRIDES_OFFSET(dimension_))[i];
        tf_to_mkl_dim_map_[i] = reinterpret_cast<const size_t*>(
            buf + TF_TO_MKL_DIM_MAP_OFFSET(dimension_))[i];
      }
      CHECK_EQ(dnnLayoutDeserialize_F32(&mklLayout_,
                                        buf + MKL_LAYOUT_OFFSET(dimension_)),
               E_SUCCESS);
      CHECK_EQ(dnnLayoutDeserialize_F32(&tfLayout_,
                                        buf + TF_LAYOUT_OFFSET(dimension_)),
               E_SUCCESS);
    }
  }

  void SerializeMklShape(unsigned char* buf, size_t buf_size) const {
    CHECK(buf_size >= SIZE_OF_MKL_SERIAL_DATA(dimension_))
        << "Bufsize too small to Serialize";
    *reinterpret_cast<size_t*>(buf + IS_MKL_TENSOR_OFFSET) =
        isMklTensor_ ? 1 : 0;
    if (isMklTensor_) {
      *(reinterpret_cast<size_t*>(buf + DIMS_OFFSET)) = dimension_;
      for (int i = 0; i < dimension_; i++) {
        reinterpret_cast<size_t*>(buf + SIZES_OFFSET(dimension_))[i] =
            sizes_[i];
        reinterpret_cast<size_t*>(buf + STRIDES_OFFSET(dimension_))[i] =
            strides_[i];
        reinterpret_cast<size_t*>(buf +
                                  TF_TO_MKL_DIM_MAP_OFFSET(dimension_))[i] =
            tf_to_mkl_dim_map_[i];
      }
      CHECK_EQ(dnnLayoutSerialize_F32(mklLayout_,
                                      buf + MKL_LAYOUT_OFFSET(dimension_)),
               E_SUCCESS);
      CHECK_EQ(
          dnnLayoutSerialize_F32(tfLayout_, buf + TF_LAYOUT_OFFSET(dimension_)),
          E_SUCCESS);
    }
  }

 private:
  bool isMklTensor_ =
      false;  // Flag to indicate if the tensor is an  MKL tensor or not
  dnnLayout_t mklLayout_ = nullptr;  // Pointer to the MKL layout
  dnnLayout_t tfLayout_ = nullptr;   // Pointer to layout of corresponding
  // Tensorflow tensor, used when conversion from MKL to standard tensor
  size_t dimension_ = 0;
  size_t* sizes_ = nullptr;    // Required by MKL for conversions
  size_t* strides_ = nullptr;  // Required by MKL for conversions
  size_t* tf_to_mkl_dim_map_ =
      nullptr;  // TF dimension corresponding to this MKL dimension
};

// List of MklShape objects. Used in Concat/Split layers.
typedef std::vector<MklShape> MklShapeList;

// Check if all tensors specified by MklShapes are MKL tensors.
inline bool AreAllMklTensors(const MklShapeList& shapes) {
  for (auto& s : shapes) {
    if (!s.IsMklTensor()) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline Tensor ConvertMklToTF(OpKernelContext* context, const Tensor& mkl_tensor,
                             const MklShape& mkl_shape) {
  Tensor output_tensor;
  TensorShape output_shape;

  for (size_t j = 0; j < mkl_shape.GetDimension(); j++) {
    // Outermost to innermost dimension
    output_shape.AddDim(mkl_shape.GetSizes()[mkl_shape.tf_dim_idx(j)]);
  }

  // Allocate output tensor.
  context->allocate_temp(DataTypeToEnum<T>::v(), output_shape, &output_tensor);

  dnnLayout_t output_layout = static_cast<dnnLayout_t>(mkl_shape.GetTfLayout());
  void* input_buffer = const_cast<T*>(mkl_tensor.flat<T>().data());
  void* output_buffer = const_cast<T*>(output_tensor.flat<T>().data());

  if (mkl_tensor.NumElements() != 0) {
    mkl_shape.GetConvertedFlatData(output_layout, input_buffer, output_buffer);
  }

  return output_tensor;
}

// Since our ops are going to produce and also consume N addition tensors
// (Mkl) for N Tensorflow tensors, we can have following different
// orderings among these 2N tensors.
//
// E.g., for Tensorflow tensors A, B, and C, our ops will produce and
// consume A_m, B_m, and C_m additionally.
//
// INTERLEAVED: in this case 2N tensors are interleaved. So for above
//              example, the ordering looks like: A, A_m, B, B_m, C, C_m.
//
// CONTIGUOUS: in thi case N Tensorflow tensors are contiguous followed
//             by N Mkl tensors. So for above example, the ordering looks
//             like: A, B, C, A_m, B_m, C_m
//
// Following APIs map index of original Tensorflow tensors to their appropriate
// position based on selected ordering. For contiguous ordering, we need to know
// the total number of tensors (parameter total).
//
typedef enum { TENSORS_INTERLEAVED, TENSORS_CONTIGUOUS } MklTfTensorOrdering;
// NOTE: Currently, we use contiguous ordering. If you change this, then you
// would need to change Mkl op definitions in nn_ops.cc.
static MklTfTensorOrdering kTensorOrdering = TENSORS_CONTIGUOUS;

// Get index of MetaData tensor from index 'n' of Data tensor.
inline int DataIndexToMetaDataIndex(int n, int total_tensors) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    // For interleaved ordering, Mkl tensor follows immediately after
    // Tensorflow tensor.
    return n + 1;
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    // For contiguous ordering, Mkl tensor is n+total_tensors / 2 away.
    return n + total_tensors / 2;
  }
}

int inline GetTensorDataIndex(int n, int total_tensors) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    return 2 * n;  // index corresponding to nth input/output tensor
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    return n;
  }
}

int inline GetTensorMetaDataIndex(int n, int total_tensors) {
  // Get index for TensorData first and then use mapping function
  // to get TensorMetaData index from TensorData index.
  int tidx = GetTensorDataIndex(n, total_tensors);
  return DataIndexToMetaDataIndex(tidx, total_tensors);
}

// Get the MKL shape from the second string tensor
inline void GetMklShape(OpKernelContext* ctext, int n, MklShape* mklshape) {
  mklshape->DeSerializeMklShape(
      ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
          .flat<uint8>()
          .data(),
      ctext->input(GetTensorMetaDataIndex(n, ctext->num_inputs()))
              .flat<uint8>()
              .size() *
          sizeof(uint8));
}

// Gets the actual input
inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
  return ctext->input(GetTensorDataIndex(n, ctext->num_inputs()));
}

inline void GetMklInputList(OpKernelContext* ctext, StringPiece name,
                            OpInputList* input_tensors) {
  CHECK_NOTNULL(input_tensors);
  ctext->input_list(name, input_tensors);
}

inline void GetMklShapeList(OpKernelContext* ctext, StringPiece name,
                            MklShapeList* mkl_shapes) {
  OpInputList input_mkl_tensors;
  GetMklInputList(ctext, strings::StrCat("mkl_", name), &input_mkl_tensors);

  for (int i = 0; i < input_mkl_tensors.size(); i++) {
    (*mkl_shapes)[i].DeSerializeMklShape(
        input_mkl_tensors[i].flat<uint8>().data(),
        input_mkl_tensors[i].flat<uint8>().size() * sizeof(uint8));
  }
}

// Allocate the second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      const MklShape& mkl_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(SIZE_OF_MKL_SERIAL_DATA(mkl_shape.GetDimension()));
  OP_REQUIRES_OK(ctext, ctext->allocate_output(
                            GetTensorMetaDataIndex(n, ctext->num_outputs()),
                            second_shape, &second_tensor));
  mkl_shape.SerializeMklShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}

// Allocate the output tensor, create a second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      Tensor** output,
                                      const TensorShape& tf_shape,
                                      const MklShape& mkl_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(SIZE_OF_MKL_SERIAL_DATA(mkl_shape.GetDimension()));
  OP_REQUIRES_OK(
      ctext, ctext->allocate_output(GetTensorDataIndex(n, ctext->num_outputs()),
                                    tf_shape, output));
  OP_REQUIRES_OK(ctext, ctext->allocate_output(
                            GetTensorMetaDataIndex(n, ctext->num_outputs()),
                            second_shape, &second_tensor));
  mkl_shape.SerializeMklShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}

// Allocates a temp tensor and returns the data buffer for temporary storage.
// Currently
// we only support F32, will need to templatize if other types are added
inline void AllocTmpBuffer(OpKernelContext* context, Tensor* tensor_out,
                           dnnLayout_t lt_buff, void** buf_out) {
  TensorShape tf_shape;

  tf_shape.AddDim(
      dnnLayoutGetMemorySize_F32(static_cast<dnnLayout_t>(lt_buff)) /
          sizeof(float) +
      1);
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::v(),
                                                 tf_shape, tensor_out));
  *buf_out = static_cast<void*>(tensor_out->flat<float>().data());
}

inline void GetStridesFromSizes(TensorFormat data_format, size_t* strides,
                                const size_t* sizes) {
  // MKL requires strides in NCHW
  if (data_format == FORMAT_NHWC) {
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

inline void MklSizesToTFSizes(OpKernelContext* context,
                              TensorFormat data_format_,
                              const MklShape& mkl_shape,
                              TensorShape* tf_shape) {
  size_t tf_dim = mkl_shape.GetDimension();
  const size_t* tf_sizes = mkl_shape.GetSizes();

  OP_REQUIRES(context, tf_dim == 4,
              errors::InvalidArgument("MKLSizesToTFSizes: size must be 4-dim"));
  std::vector<int32> sizes;

  sizes.push_back(tf_sizes[3]);

  if (data_format_ == FORMAT_NHWC) {
    sizes.push_back(tf_sizes[1]);
    sizes.push_back(tf_sizes[0]);
    sizes.push_back(tf_sizes[2]);
  } else {
    sizes.push_back(tf_sizes[2]);
    sizes.push_back(tf_sizes[1]);
    sizes.push_back(tf_sizes[0]);
  }

  OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(sizes, tf_shape));
}

inline int32 GetMklTensorDimIndex(char dimension) {
  switch (dimension) {
    case 'N':
      return MklDims::N;
    case 'C':
      return MklDims::C;
    case 'H':
      return MklDims::H;
    case 'W':
      return MklDims::W;
    default:
      LOG(FATAL) << "Invalid dimension: " << dimension;
      return -1;  // Avoid compiler warning about missing return value
  }
}

inline int64 GetMklTensorDim(const MklShape& mkl_shape, char dimension) {
  int index = GetMklTensorDimIndex(dimension);
  CHECK(index >= 0 && index < mkl_shape.GetDimension())
      << "Invalid index from the dimension: " << index << ", " << dimension;
  return mkl_shape.dim_size(index);
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

inline void CopyTFTensorInToOut(OpKernelContext* context, int idx_in,
                                int idx_out, const TensorShape& shape) {
  int num_inputs = context->num_inputs();
  int num_outputs = context->num_outputs();
  int idx_data_in = GetTensorDataIndex(idx_in, num_inputs);
  int idx_data_out = GetTensorDataIndex(idx_out, num_outputs);

  const Tensor& data = context->input(idx_data_in);
  MklShape mkl_shape_output;
  mkl_shape_output.SetMklTensor(false);
  AllocateOutputSetMklShape(context, idx_out, mkl_shape_output);
  Tensor output(data.dtype());
  // TODO(intel_tf): alternatively, call forward_input_to_output_with_shape(...)
  CHECK(output.CopyFrom(data, shape));
  context->set_output(idx_data_out, output);
}

namespace mkl_op_registry {
static const char* kMklOpLabel = "MklOp";
static const char* kMklOpLabelPattern = "label='MklOp'";

// Check whether opname with type T is registered as MKL-compliant.
//
// @input: name of the op
// @input: T datatype to be used for checking op
// @return: true if opname is registered as Mkl op
static inline bool IsMklOp(const std::string& op_name, DataType T) {
  string kernel = KernelsRegisteredForOp(op_name);
  bool result =
      kernel.find(kMklOpLabelPattern) != string::npos && (T == DT_FLOAT);
  if (result) {
    VLOG(1) << "mkl_op_registry::" << op_name << " is " << kMklOpLabel;
  }
  return result;
}

}  // namespace mkl_op_registry

}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
