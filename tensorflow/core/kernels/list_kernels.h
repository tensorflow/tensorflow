/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_LIST_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_LIST_KERNELS_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// Variant compatible type for a list of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
struct TensorList {
 public:
  TensorList() {}
  TensorList(const TensorList& other);

  static const char kTypeName[];
  string TypeName() const { return kTypeName; }

  void Encode(VariantTensorData* data) const;

  bool Decode(const VariantTensorData& data);

  // TODO(apassos) fill this out
  string DebugString() const { return "TensorList"; }

  std::vector<Tensor> tensors;
  PartialTensorShape element_shape;
  DataType element_dtype;
};

Status TensorShapeFromTensor(const Tensor& t, PartialTensorShape* out);

template <typename Device, typename T>
class TensorListStack : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;
  explicit TensorListStack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("num_elements", &num_elements_));
  }

  ~TensorListStack() {}

  void Compute(OpKernelContext* c) override {
    const TensorList* l = c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, l != nullptr,
                errors::InvalidArgument(
                    "Input handle is not a list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));
    OP_REQUIRES(c, l->element_shape.IsFullyDefined(),
                errors::InvalidArgument("Tried to stack elements from a list "
                                        "with non-fully-defined shape: ",
                                        l->element_shape.DebugString()));
    if (num_elements_ != -1) {
      OP_REQUIRES(c, l->tensors.size() == num_elements_,
                  errors::InvalidArgument("Operation expected a list with ",
                                          num_elements_,
                                          " elements but got a list with ",
                                          l->tensors.size(), " elements."));
    }
    TensorShape resulting_shape;
    resulting_shape.AddDim(l->tensors.size());
    for (TensorShapeDim s : l->element_shape) {
      resulting_shape.AddDim(s.size);
    }
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, resulting_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(l->tensors.size());
    for (const auto& t : l->tensors) {
      OP_REQUIRES(
          c, l->element_shape.IsCompatibleWith(t.shape()),
          errors::InvalidArgument(
              "Tensor with invalid shape in list. List element shape shape: ",
              l->element_shape.DebugString(),
              " and tensor shape: ", t.shape().DebugString()));
      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          t.shaped<T, 2>({1, t.NumElements()})));
    }
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});

#if GOOGLE_CUDA
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      ConcatGPU<T>(c, inputs_flat, output, &output_flat);
      return;
    }
#endif  // GOOGLE_CUDA
    ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
  }

 private:
  int num_elements_;
  DataType element_dtype_;
};

template <typename Device, typename T>
class TensorListFromTensor : public OpKernel {
 public:
  TensorListFromTensor(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Tensor* output_tensor;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &output_tensor, attr));
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(c, TensorShapeFromTensor(c->input(1), &element_shape));
    TensorList output_list;
    const Tensor& t = c->input(0);
    output_list.element_dtype = t.dtype();
    TensorShape output_shape(t.shape());
    output_shape.RemoveDim(0);
    OP_REQUIRES(c, element_shape.IsCompatibleWith(output_shape),
                errors::InvalidArgument(
                    "Specified a list with shape ", element_shape.DebugString(),
                    " from a tensor with shape ", output_shape.DebugString()));
    output_list.element_shape = element_shape;
    output_list.tensors.reserve(t.shape().dim_size(0));
    for (int i = 0; i < t.shape().dim_size(0); ++i) {
      Tensor tmp = t.Slice(i, i + 1);
      TensorShape tmp_shape = tmp.shape();
      tmp_shape.RemoveDim(0);
      OP_REQUIRES(c, tmp.CopyFrom(tmp, tmp_shape),
                  errors::Unknown("Unexpected shape error."));
      // TODO(apassos) maybe not always align; but weird compiler bugs seem to
      // prevent this.
      Tensor aligned;
      OP_REQUIRES_OK(c, c->allocate_temp(tmp.dtype(), tmp.shape(), &aligned));
      aligned.flat<T>().device(c->eigen_device<Device>()) =
          tmp.unaligned_flat<T>();
      output_list.tensors.push_back(aligned);
    }
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }
};

template <typename Device>
Status TensorListBinaryAdd(OpKernelContext* c, const TensorList& a,
                           const TensorList& b, TensorList* out) {
  if (a.element_dtype != b.element_dtype) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors of different dtypes. One is ",
        DataTypeString(a.element_dtype), " and the other is ",
        DataTypeString(b.element_dtype));
  }
  out->element_dtype = a.element_dtype;
  if (!a.element_shape.IsCompatibleWith(b.element_shape)) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors with incompatible element shapes. "
        "One is ",
        a.element_shape.DebugString(), " and the other is ",
        b.element_shape.DebugString());
  }

  TF_RETURN_IF_ERROR(
      a.element_shape.MergeWith(b.element_shape, &out->element_shape));
  if (a.tensors.size() != b.tensors.size()) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors with different lengths. One is ",
        a.tensors.size(), " and the other is ", b.tensors.size());
  }
  out->tensors.reserve(a.tensors.size());
  for (int i = 0; i < a.tensors.size(); ++i) {
    const Tensor& a_tensor = a.tensors[i];
    const Tensor& b_tensor = b.tensors[i];
    if (a_tensor.dtype() == DT_INVALID) {
      out->tensors.push_back(b_tensor);
      continue;
    }
    if (b_tensor.dtype() == DT_INVALID) {
      out->tensors.push_back(a_tensor);
      continue;
    }
    if (a_tensor.shape() != b_tensor.shape()) {
      // TODO(apassos) support broadcasting additions here?
      return errors::InvalidArgument(
          "Trying to add two tensors with incompatible element shapes. "
          "One is ",
          a_tensor.shape().DebugString(), " and the other is ",
          b_tensor.shape().DebugString(), " in position ", i);
    }
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(
        c->allocate_temp(a_tensor.dtype(), a_tensor.shape(), &out_tensor));
    out->tensors.push_back(out_tensor);
    switch (out_tensor.dtype()) {
#define DTYPE_CASE(dtype)                                        \
  case DataTypeToEnum<dtype>::value:                             \
    out_tensor.flat<dtype>().device(c->eigen_device<Device>()) = \
        a_tensor.flat<dtype>() + b_tensor.flat<dtype>();         \
    break;

      TF_CALL_NUMBER_TYPES(DTYPE_CASE)

#undef DTYPE_CASE
      default:
        return errors::InvalidArgument("Trying to add unsupported dtype ",
                                       out_tensor.dtype());
    }
  }
  return Status::OK();
}

template <typename Device>
Status TensorListZerosLike(OpKernelContext* c, const TensorList& x,
                           TensorList* y) {
  y->element_dtype = x.element_dtype;
  y->element_shape = x.element_shape;
  y->tensors.reserve(x.tensors.size());
  for (const Tensor& t : x.tensors) {
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(c->allocate_temp(t.dtype(), t.shape(), &out_tensor));
    switch (out_tensor.dtype()) {
#define DTYPE_CASE(dtype)                                        \
  case DataTypeToEnum<dtype>::value:                             \
    out_tensor.flat<dtype>().device(c->eigen_device<Device>()) = \
        out_tensor.flat<dtype>().constant(dtype(0));             \
    break;

      TF_CALL_NUMBER_TYPES(DTYPE_CASE)

#undef DTYPE_CASE
      default:
        return errors::InvalidArgument(
            "Trying to compute zeros_like for unsupported dtype",
            out_tensor.dtype());
    }
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LIST_KERNELS_H_
