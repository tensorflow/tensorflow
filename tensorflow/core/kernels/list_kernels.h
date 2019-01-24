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
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/tensor_ops_util.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

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
  // The maximum allowed size of `tensors`. Defaults to -1 meaning that the size
  // of `tensors` is unbounded.
  int max_num_elements = -1;
};

Status TensorShapeFromTensor(const Tensor& t, PartialTensorShape* out);

Status GetElementShapeFromInput(OpKernelContext* c,
                                const TensorList& tensor_list, int index,
                                PartialTensorShape* element_shape);

template <typename Device, typename T>
class TensorListStack : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;
  explicit TensorListStack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("num_elements", &num_elements_));
  }

  void Compute(OpKernelContext* c) override {
    const TensorList* tensor_list =
        c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, tensor_list != nullptr,
                errors::InvalidArgument(
                    "Input handle is not a list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    if (num_elements_ != -1) {
      OP_REQUIRES(c, tensor_list->tensors.size() == num_elements_,
                  errors::InvalidArgument(
                      "Operation expected a list with ", num_elements_,
                      " elements but got a list with ",
                      tensor_list->tensors.size(), " elements."));
    }
    PartialTensorShape partial_element_shape;
    OP_REQUIRES_OK(c, GetElementShapeFromInput(c, *tensor_list, 1,
                                               &partial_element_shape));
    OP_REQUIRES(
        c,
        partial_element_shape.IsFullyDefined() || !tensor_list->tensors.empty(),
        errors::InvalidArgument("Tried to stack elements of an empty ",
                                "list with non-fully-defined element_shape: ",
                                partial_element_shape.DebugString()));

    // Check that `element_shape` input tensor is compatible with the shapes of
    // element tensors.
    if (!tensor_list->element_shape.IsFullyDefined()) {
      for (int i = 0; i < tensor_list->tensors.size(); ++i) {
        const Tensor& t = tensor_list->tensors[i];
        if (t.dtype() != DT_INVALID) {
          PartialTensorShape tmp = partial_element_shape;
          OP_REQUIRES_OK(c, tmp.MergeWith(t.shape(), &partial_element_shape));
        }
      }
    }

    // Compute the shape of the output tensor by pre-pending the leading dim to
    // the element_shape.
    TensorShape element_shape;
    OP_REQUIRES(c, partial_element_shape.AsTensorShape(&element_shape),
                errors::InvalidArgument(
                    "Tried to stack list which only contains uninitialized ",
                    "tensors and has a non-fully-defined element_shape: ",
                    partial_element_shape.DebugString()));
    TensorShape output_shape = element_shape;
    output_shape.InsertDim(0, tensor_list->tensors.size());
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(tensor_list->tensors.size());
    Tensor zeros;
    for (const auto& t : tensor_list->tensors) {
      if (t.dtype() != DT_INVALID) {
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            t.shaped<T, 2>({1, t.NumElements()})));
      } else {
        if (!zeros.NumElements()) {
          AllocatorAttributes attr;
          if (element_dtype_ == DT_VARIANT) {
            attr.set_on_host(true);
          }
          OP_REQUIRES_OK(
              c, c->allocate_temp(element_dtype_, element_shape, &zeros, attr));
          functor::SetZeroFunctor<Device, T>()(c->eigen_device<Device>(),
                                               zeros.flat<T>());
        }
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            const_cast<const Tensor&>(zeros).shaped<T, 2>(
                {1, zeros.NumElements()})));
      }
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
class TensorListGetItem : public OpKernel {
 public:
  explicit TensorListGetItem(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    OP_REQUIRES(
        c, c->input(0).shape().num_elements() == 1,
        errors::InvalidArgument("List tensors are supposed to be scalars."));
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
    int32 index = c->input(1).scalar<int32>()();
    OP_REQUIRES(c, index < l->tensors.size(),
                errors::InvalidArgument("Trying to access element ", index,
                                        " in a list with ", l->tensors.size(),
                                        " elements."));
    if (l->tensors[index].dtype() != DT_INVALID) {
      c->set_output(0, l->tensors[index]);
    } else {
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(
          c, GetElementShapeFromInput(c, *l, 2, &partial_element_shape));
      TensorShape element_shape;
      OP_REQUIRES(
          c, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument("Trying to read an uninitialized tensor but ",
                                  "element_shape is not fully defined.",
                                  partial_element_shape.DebugString()));
      Tensor* result;
      AllocatorAttributes attr;
      if (element_dtype_ == DT_VARIANT) {
        attr.set_on_host(true);
      }
      OP_REQUIRES_OK(c, c->allocate_output(0, element_shape, &result, attr));
      functor::SetZeroFunctor<Device, T>()(c->eigen_device<Device>(),
                                           result->flat<T>());
    }
  }

 private:
  DataType element_dtype_;
};

template <typename Device, typename T>
class TensorListPopBack : public OpKernel {
 public:
  explicit TensorListPopBack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

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

    OP_REQUIRES(c, !l->tensors.empty(),
                errors::InvalidArgument("Trying to pop from an empty list."));

    const Tensor& t = l->tensors.back();
    if (t.dtype() != DT_INVALID) {
      c->set_output(1, t);
    } else {
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(
          c, GetElementShapeFromInput(c, *l, 1, &partial_element_shape));
      TensorShape element_shape;
      OP_REQUIRES(
          c, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument("Trying to read an uninitialized tensor but ",
                                  "element_shape is not fully defined.",
                                  partial_element_shape.DebugString()));
      Tensor* result;
      AllocatorAttributes attr;
      if (element_dtype_ == DT_VARIANT) {
        attr.set_on_host(true);
      }
      OP_REQUIRES_OK(c, c->allocate_output(1, element_shape, &result, attr));
      functor::SetZeroFunctor<Device, T>()(c->eigen_device<Device>(),
                                           result->flat<T>());
    }
    AllocatorAttributes attr;
    attr.set_on_host(true);
    std::unique_ptr<Tensor> maybe_result = c->forward_input(
        0, 0, DT_VARIANT, TensorShape{}, c->input_memory_type(0), attr);
    if (maybe_result != nullptr) {
      maybe_result->scalar<Variant>()().get<TensorList>()->tensors.pop_back();
    } else {
      TensorList output;
      output = *l;
      output.tensors.pop_back();
      Tensor* result;
      OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
      result->scalar<Variant>()() = std::move(output);
    }
  }

 private:
  DataType element_dtype_;
};

template <typename Device, typename T>
class TensorListConcat : public OpKernel {
 public:
  using ConstMatrixVector =
      std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>;
  explicit TensorListConcat(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
    // TODO(skyewm): the HasAttr check can be removed once the
    // element_shape_except_first_dim attr has been checked in for 2 weeks
    // (around 1/14/2019).
    if (c->HasAttr("element_shape")) {
      PartialTensorShape element_shape;
      OP_REQUIRES_OK(c, c->GetAttr("element_shape", &element_shape));
      if (!element_shape.unknown_rank()) {
        element_shape_except_first_dim_ = PartialTensorShape(
            gtl::ArraySlice<int64>(element_shape.dim_sizes()).subspan(1));
      }
    }
  }

  void Compute(OpKernelContext* c) override {
    // Check that the input Variant tensor is indeed a TensorList and has the
    // correct element type.
    const TensorList* tensor_list =
        c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, tensor_list != nullptr,
                errors::InvalidArgument(
                    "Input handle is not a list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    // If the TensorList is empty, its element_shape must be fully defined
    // except for the first dimension.
    if (!element_shape_except_first_dim_.IsFullyDefined()) {
      if (!tensor_list->element_shape.unknown_rank()) {
        OP_REQUIRES(c, tensor_list->element_shape.dims() >= 1,
                    errors::InvalidArgument(
                        "Concat requires elements to be at least vectors, ",
                        "found scalars instead."));
        PartialTensorShape shape_except_first_dim(
            gtl::ArraySlice<int64>(tensor_list->element_shape.dim_sizes())
                .subspan(1));
        PartialTensorShape tmp = element_shape_except_first_dim_;
        OP_REQUIRES_OK(c, tmp.MergeWith(shape_except_first_dim,
                                        &element_shape_except_first_dim_));
      }
    }
    OP_REQUIRES(c,
                !tensor_list->tensors.empty() ||
                    element_shape_except_first_dim_.IsFullyDefined(),
                errors::InvalidArgument(
                    "All except the first dimension must be fully defined ",
                    "when concating an empty tensor list. element_shape: ",
                    tensor_list->element_shape.DebugString()));
    // 1. Compute the shape of the output tensor.
    // If `element_shape_except_first_dim_` is fully-defined we just prepend the
    // leading dim to it. Otherwise we use the shape of the first element tensor
    // and check to make sure shapes of all tensors are compatible.
    TensorShape output_shape;
    if (!element_shape_except_first_dim_.AsTensorShape(&output_shape)) {
      const Tensor& element_tensor = tensor_list->tensors[0];
      OP_REQUIRES(
          c, TensorShapeUtils::IsVectorOrHigher(element_tensor.shape()),
          errors::InvalidArgument("Concat saw a scalar shape at index ", 0,
                                  " but requires at least vectors."));
      output_shape =
          TensorShape(gtl::ArraySlice<int64>(element_tensor.shape().dim_sizes())
                          .subspan(1));
      for (int i = 1; i < tensor_list->tensors.size(); ++i) {
        const Tensor& element_tensor = tensor_list->tensors[i];
        OP_REQUIRES(
            c, TensorShapeUtils::IsVectorOrHigher(element_tensor.shape()),
            errors::InvalidArgument("Concat saw a scalar shape at index ", i,
                                    " but requires at least vectors."));
        TensorShape actual_shape(
            gtl::ArraySlice<int64>(element_tensor.shape().dim_sizes())
                .subspan(1));
        OP_REQUIRES(c, actual_shape.dim_sizes() == output_shape.dim_sizes(),
                    errors::InvalidArgument(
                        "Tried to concat tensors with unequal shapes: ",
                        output_shape.DebugString(), " vs ",
                        actual_shape.DebugString()));
      }
    }
    // 2. Build the lengths_tensor and leading dim of the output tensor by
    // iterating over all element tensors.
    Tensor* lengths_tensor = nullptr;
    OP_REQUIRES_OK(
        c,
        c->allocate_output(
            1, TensorShape({static_cast<int64>(tensor_list->tensors.size())}),
            &lengths_tensor));
    auto lengths_tensor_vec = lengths_tensor->vec<int64>();
    int64 leading_dim = 0;
    for (size_t i = 0; i < tensor_list->tensors.size(); i++) {
      int64 dim = tensor_list->tensors[i].shape().dim_size(0);
      leading_dim += dim;
      lengths_tensor_vec(i) = dim;
    }
    output_shape.InsertDim(0, leading_dim);
    Tensor* output;
    // 3. Allocate the output tensor and fill it up with the concated element
    // tensors.
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(tensor_list->tensors.size());
    for (const auto& element_tensor : tensor_list->tensors) {
      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          element_tensor.shaped<T, 2>({1, element_tensor.NumElements()})));
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
  DataType element_dtype_;
  PartialTensorShape element_shape_except_first_dim_;
};

template <typename Device, typename T>
class TensorListSplit : public OpKernel {
 public:
  TensorListSplit(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Tensor* output_tensor;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &output_tensor, attr));
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(c, TensorShapeFromTensor(c->input(1), &element_shape));
    OP_REQUIRES(c, element_shape.unknown_rank() || element_shape.dims() >= 1,
                errors::InvalidArgument(
                    "TensorListSplit requires element_shape to be at least of ",
                    "rank 1, but saw: ", element_shape.DebugString()));
    TensorList output_list;
    const Tensor& input_tensor = c->input(0);
    output_list.element_dtype = input_tensor.dtype();
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input_tensor.shape()),
                errors::InvalidArgument(
                    "Tensor must be at least a vector, but saw shape: ",
                    input_tensor.shape().DebugString()));
    TensorShape tensor_shape_without_first_dim(input_tensor.shape());
    tensor_shape_without_first_dim.RemoveDim(0);
    PartialTensorShape element_shape_without_first_dim;
    if (!element_shape.unknown_rank()) {
      element_shape_without_first_dim =
          PartialTensorShape(element_shape.dim_sizes());
      element_shape_without_first_dim.RemoveDim(0);
    }
    OP_REQUIRES(c,
                element_shape_without_first_dim.IsCompatibleWith(
                    tensor_shape_without_first_dim),
                errors::InvalidArgument(
                    "tensor shape ", input_tensor.shape().DebugString(),
                    " is not compatible with element_shape ",
                    element_shape.DebugString()));
    output_list.element_shape = element_shape;
    const Tensor& lengths = c->input(2);
    OP_REQUIRES(c, TensorShapeUtils::IsVector(lengths.shape()),
                errors::InvalidArgument(
                    "Expected lengths to be a vector, received shape: ",
                    lengths.shape().DebugString()));
    output_list.tensors.reserve(lengths.shape().dim_size(0));
    int64 start = 0;
    int64 end = 0;
    for (int i = 0; i < lengths.shape().dim_size(0); ++i) {
      int64 length = lengths.vec<int64>()(i);
      OP_REQUIRES(
          c, length >= 0,
          errors::InvalidArgument("Invalid value in lengths: ", length));
      end = start + length;
      OP_REQUIRES(c, end <= input_tensor.shape().dim_size(0),
                  errors::InvalidArgument("Attempting to slice [", start, ", ",
                                          end, "] from tensor with length ",
                                          input_tensor.shape().dim_size(0)));
      Tensor tmp = input_tensor.Slice(start, end);
      start = end;
      // TODO(apassos) maybe not always align; but weird compiler bugs seem to
      // prevent this.
      Tensor aligned;
      OP_REQUIRES_OK(c, c->allocate_temp(tmp.dtype(), tmp.shape(), &aligned));
      aligned.flat<T>().device(c->eigen_device<Device>()) =
          tmp.unaligned_flat<T>();
      output_list.tensors.emplace_back(aligned);
    }
    OP_REQUIRES(c, end == input_tensor.shape().dim_size(0),
                errors::InvalidArgument(
                    "Unused values in tensor. Length of tensor: ",
                    input_tensor.shape().dim_size(0), " Values used: ", end));
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }
};

template <typename Device, typename T>
class TensorListGather : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;
  explicit TensorListGather(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    const TensorList* tensor_list =
        c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, tensor_list != nullptr,
                errors::InvalidArgument(
                    "Input handle is not a list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    Tensor indices = c->input(1);
    PartialTensorShape partial_element_shape;
    OP_REQUIRES_OK(c, GetElementShapeFromInput(c, *tensor_list, 2,
                                               &partial_element_shape));
    OP_REQUIRES(
        c, partial_element_shape.IsFullyDefined() || indices.NumElements() > 0,
        errors::InvalidArgument("Tried to gather 0-elements from "
                                "a list with non-fully-defined shape: ",
                                partial_element_shape.DebugString()));

    // Check that `element_shape` input tensor is compatible with the shapes of
    // element tensors.
    if (!tensor_list->element_shape.IsFullyDefined()) {
      for (int index = 0; index < indices.NumElements(); ++index) {
        const int i = indices.flat<int32>()(index);
        const Tensor& t = tensor_list->tensors[i];
        if (t.dtype() != DT_INVALID) {
          PartialTensorShape tmp = partial_element_shape;
          OP_REQUIRES_OK(c, tmp.MergeWith(t.shape(), &partial_element_shape));
        }
      }
    }

    // Compute the shape of the output tensor by pre-pending the leading dim to
    // the element_shape.
    TensorShape element_shape;
    OP_REQUIRES(
        c, partial_element_shape.AsTensorShape(&element_shape),
        errors::InvalidArgument("Tried to gather uninitialized tensors from a ",
                                "list with non-fully-defined element_shape: ",
                                partial_element_shape.DebugString()));
    TensorShape output_shape = element_shape;
    output_shape.InsertDim(0, indices.NumElements());
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(indices.NumElements());
    Tensor zeros;
    for (int index = 0; index < indices.NumElements(); ++index) {
      const int i = indices.flat<int32>()(index);
      OP_REQUIRES(
          c, i < tensor_list->tensors.size(),
          errors::InvalidArgument("Index ", i, " out o range; list only has ",
                                  tensor_list->tensors.size(), " elements."));
      const Tensor& t = tensor_list->tensors[i];
      if (t.dtype() != DT_INVALID) {
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            t.shaped<T, 2>({1, t.NumElements()})));
      } else {
        if (!zeros.NumElements()) {
          AllocatorAttributes attr;
          if (element_dtype_ == DT_VARIANT) {
            attr.set_on_host(true);
          }
          OP_REQUIRES_OK(
              c, c->allocate_temp(element_dtype_, element_shape, &zeros, attr));
          functor::SetZeroFunctor<Device, T>()(c->eigen_device<Device>(),
                                               zeros.flat<T>());
        }
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            const_cast<const Tensor&>(zeros).shaped<T, 2>(
                {1, zeros.NumElements()})));
      }
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
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(t.shape()),
                errors::InvalidArgument(
                    "Tensor must be at least a vector, but saw shape: ",
                    t.shape().DebugString()));
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

template <typename Device, typename T>
class TensorListScatter : public OpKernel {
 public:
  TensorListScatter(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Tensor* output_tensor;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, {}, &output_tensor, attr));
    Tensor indices = c->input(1);
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(c, TensorShapeFromTensor(c->input(2), &element_shape));
    TensorList output_list;
    const Tensor& input_tensor = c->input(0);
    output_list.element_dtype = input_tensor.dtype();
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input_tensor.shape()),
                errors::InvalidArgument(
                    "Tensor must be at least a vector, but saw shape: ",
                    input_tensor.shape().DebugString()));
    TensorShape output_shape(input_tensor.shape());
    output_shape.RemoveDim(0);
    OP_REQUIRES(c, element_shape.IsCompatibleWith(output_shape),
                errors::InvalidArgument(
                    "Specified a list with shape ", element_shape.DebugString(),
                    " from a tensor with shape ", output_shape.DebugString()));
    output_list.element_shape = element_shape;

    OP_REQUIRES(c, indices.NumElements() == input_tensor.shape().dim_size(0),
                errors::InvalidArgument(
                    "Invalid number of rows in input tensor. Expected: ",
                    indices.NumElements(),
                    " Actual: ", input_tensor.shape().dim_size(0)));

    // Validate indices and resize output_list.tensors to fit the highest index.
    {
      size_t list_size = 0;
      for (int index = 0; index < indices.NumElements(); ++index) {
        const int i = indices.flat<int32>()(index);
        OP_REQUIRES(c, i >= 0,
                    errors::InvalidArgument(
                        "Indices in TensorListScatter must all be positive."));
        if (i >= list_size) {
          list_size = i + 1;
        }
      }
      output_list.tensors.resize(list_size, Tensor(DT_INVALID));
    }

    for (int index = 0; index < indices.NumElements(); ++index) {
      const int i = indices.flat<int32>()(index);
      Tensor tmp = input_tensor.Slice(index, index + 1);
      TensorShape tmp_shape = tmp.shape();
      tmp_shape.RemoveDim(0);
      OP_REQUIRES(c, tmp.CopyFrom(tmp, tmp_shape),
                  errors::Unknown("Unexpected shape error."));
      // TODO(apassos) maybe not always align; but weird compiler bugs seem to
      // prevent this.
      Tensor aligned;
      OP_REQUIRES_OK(c, c->allocate_temp(tmp.dtype(), tmp.shape(), &aligned));
      // TODO(apassos) do all slices in a single kernel invocation instead of
      // many small ondes.
      aligned.flat<T>().device(c->eigen_device<Device>()) =
          tmp.unaligned_flat<T>();
      std::swap(output_list.tensors[i], aligned);
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
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(
        BinaryAddTensors<Device>(c, a_tensor, b_tensor, &out_tensor));
    out->tensors.push_back(out_tensor);
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
    TF_RETURN_IF_ERROR(ZerosLikeTensor<Device>(c, t, &out_tensor));
    y->tensors.emplace_back(out_tensor);
  }
  return Status::OK();
}

template <typename Device, typename T>
class TensorListPushBackBatch : public OpKernel {
 public:
  explicit TensorListPushBackBatch(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(1);
    OP_REQUIRES(c, element_dtype_ == input.dtype(),
                errors::InvalidArgument("Invalid data types; list elements ",
                                        DataTypeString(element_dtype_),
                                        " but tried to append ",
                                        DataTypeString(input.dtype())));
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input.shape()),
                errors::InvalidArgument(
                    "Expected tensor to be at least a vector, but saw shape: ",
                    input.shape().DebugString()));

    const TensorShape& tls_shape = c->input(0).shape();

    // For purposes of input forwarding, we want the least restrictive
    // AllocatorAttributes possible.  If we need to allocate later,
    // we'll request the DT_VARIANT be allocated on host.
    AllocatorAttributes attr;

    std::unique_ptr<Tensor> tls_alias = c->forward_input(
        0 /*input_index*/, 0 /*output_index*/, DT_VARIANT, tls_shape,
        DEVICE_MEMORY /* input is always on DEVICE_MEMORY */, attr);

    const Tensor& tls = tls_alias ? *tls_alias : c->input(0);

    OP_REQUIRES(c, tls.dtype() == DT_VARIANT,
                errors::InvalidArgument(
                    "Expected input_handles dtype to be Variant, but saw: ",
                    DataTypeString(tls.dtype())));
    OP_REQUIRES(c, TensorShapeUtils::IsVector(tls_shape),
                errors::InvalidArgument(
                    "Expected input_handles to be a vector, but saw shape: ",
                    tls_shape.DebugString()));
    const int64 batch_size = tls.NumElements();
    OP_REQUIRES(c, input.dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "Expected tensor.shape[0] == input_handles.size, but saw ",
                    input.dim_size(0), " vs. ", batch_size));
    auto tls_t = tls.vec<Variant>();

    TensorShape input_element_shape = input.shape();
    input_element_shape.RemoveDim(0);
    std::vector<const TensorList*> tl_batch;
    for (int64 b = 0; b < batch_size; ++b) {
      const TensorList* l = tls_t(b).get<TensorList>();
      OP_REQUIRES(c, l != nullptr,
                  errors::InvalidArgument("Input handle at index ", b,
                                          " is not a list. Saw: '",
                                          tls_t(b).DebugString(), "'"));
      OP_REQUIRES(
          c, l->element_shape.IsCompatibleWith(input_element_shape),
          errors::InvalidArgument(
              "Tried to append a tensor with incompatible shape to a "
              "list at index ",
              b, ". Op element shape: ", input_element_shape.DebugString(),
              " list shape: ", l->element_shape.DebugString()));
      OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                  errors::InvalidArgument(
                      "Invalid data type at index ", b, "; op elements ",
                      DataTypeString(element_dtype_), " but list elements ",
                      DataTypeString(l->element_dtype)));
      tl_batch.push_back(l);
    }

    Tensor* result;

    if (tls_alias) {
      result = tls_alias.get();
      c->set_output(0, *result);
    } else {
      // DT_VARIANT tensors always allocated on host.
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          c, c->allocate_output(0, TensorShape{batch_size}, &result, attr));
    }

    if (batch_size == 0) {
      return;
    }

    auto input_t = input.flat_outer_dims<T, 2>();
    auto result_t = result->vec<Variant>();

    for (int64 b = 0; b < batch_size; ++b) {
      if (!tls_alias) {
        result_t(b) = *tl_batch[b];
      }
      TensorList* output = result_t(b).get<TensorList>();
      DCHECK(output != nullptr);
      Tensor* frame;
      PersistentTensor tmp;
      OP_REQUIRES_OK(c, c->allocate_persistent(
                            element_dtype_, input_element_shape, &tmp, &frame));
      if (input_element_shape.num_elements() > 0) {
        auto frame_t = frame->flat<T>();
        frame_t.device(c->eigen_device<Device>()) = input_t.template chip<0>(b);
      }
      output->tensors.push_back(std::move(*frame));
    }
  }

 private:
  DataType element_dtype_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LIST_KERNELS_H_
