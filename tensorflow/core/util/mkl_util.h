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

class MklShape {
 public:
  MklShape() {}
  TF_DISALLOW_COPY_AND_ASSIGN(MklShape);  // Cannot copy

  ~MklShape() {
    if (sizes_) delete[] sizes_;
    if (strides_) delete[] strides_;
    if (mklLayout_) CHECK_EQ(dnnLayoutDelete_F32(mklLayout_), E_SUCCESS);
    if (tfLayout_) CHECK_EQ(dnnLayoutDelete_F32(tfLayout_), E_SUCCESS);
  }

  const bool IsMklTensor() const { return isMklTensor_; }

  void SetMklTensor(const bool isMklTensor) { isMklTensor_ = isMklTensor; }

  void SetMklLayout(const void* primitive, size_t resourceType) {
    CHECK_EQ(
        dnnLayoutCreateFromPrimitive_F32(&mklLayout_, (dnnPrimitive_t)primitive,
                                         (dnnResourceType_t)resourceType),
        E_SUCCESS);
  }

  void SetTfLayout(const size_t dimension, const size_t* sizes,
                   const size_t* strides) {
    dimension_ = dimension;
    if (dimension > 0) {  // MKl doesn't support dimension 0
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

  const dnnLayout_t GetMklLayout() const { return mklLayout_; }
  const dnnLayout_t GetTfLayout() const { return tfLayout_; }
  const dnnLayout_t GetCurLayout() const {
    return isMklTensor_ ? mklLayout_ : tfLayout_;
  }
  size_t GetDimension() const { return dimension_; }
  const size_t* GetSizes() const { return sizes_; }
  const size_t* GetStrides() const { return strides_; }

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
// sizes
// strides
// mklLayout_
// tfLayout_

#define SIZE_OF_MKL_DNN_BUF \
  (dnnLayoutSerializationBufferSize_F32())  // Size of buffer needed to
                                            // serialize dnn_layout pointer

// Size of buffer to hold the serialized object, the size is computed as follows
// sizeof(isMklTensor_) + sizeof(dimension_) + sizeof(sizes) + sizeof(strides)
// + sizeof(mklLayout_ buffer) + sizeof(tfLayout_ buffer)

#define SIZE_OF_MKL_SERIAL_DATA(dims) \
  (2 * sizeof(size_t) + 2 * dims * sizeof(size_t) + 2 * SIZE_OF_MKL_DNN_BUF)

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
      for (int i = 0; i < dimension_; i++) {
        sizes_[i] =
            reinterpret_cast<const size_t*>(buf + SIZES_OFFSET(dimension_))[i];
        strides_[i] = reinterpret_cast<const size_t*>(
            buf + STRIDES_OFFSET(dimension_))[i];
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
};

int inline GetTensorDataIndex(int n) {
  return 2 * n;  // index corresponding to nth input/output tensor
}

int inline GetTensorMetaDataIndex(int n) {
  // index corresponding to meta data of nth input/output tensor
  return 2 * n + 1;
}
// Get the MKL shape from the second string tensor
inline void GetMklShape(OpKernelContext* ctext, int n, MklShape* mklshape) {
  mklshape->DeSerializeMklShape(
      ctext->input(GetTensorMetaDataIndex(n)).flat<uint8>().data(),
      ctext->input(GetTensorMetaDataIndex(n)).flat<uint8>().size() *
          sizeof(uint8));
}

// Gets the actual input
inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
  return ctext->input(GetTensorDataIndex(n));
}

// Allocate the output tensor, create a second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklshape(OpKernelContext* ctext, int n,
                                      Tensor** output,
                                      const TensorShape& tfshape,
                                      const MklShape& mklshape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(SIZE_OF_MKL_SERIAL_DATA(mklshape.GetDimension()));
  OP_REQUIRES_OK(
      ctext, ctext->allocate_output(GetTensorDataIndex(n), tfshape, output));
  OP_REQUIRES_OK(ctext, ctext->allocate_output(GetTensorMetaDataIndex(n),
                                               second_shape, &second_tensor));
  mklshape.SerializeMklShape(
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

namespace mkl_layer_registry {

static const char* kMklLayerLabel = "MklLayer";
static const string kMklLayerLabelPattern = "label='MklLayer'";

// Check whether opname is registered as MKL-compliant in the registry.
//
// @input: name of the op
// @return: true if opname is registered as Mkl layer op
static inline bool IsMklLayer(const std::string& op_name) {
  string kernel = KernelsRegisteredForOp(op_name);
  return kernel.find(kMklLayerLabelPattern) != string::npos;
}

}  // namespace mkl_layer_registry

}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_UTIL_H_
