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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/kernels/tensor_list_util.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/util/tensor_ops_util.h"
#include "tensorflow/core/util/util.h"

// stream.h isn't available in some platforms such as Android, iOS, ChromiumOS,
// and Fuchsia. Only include it for platforms that PluggableDevice is tested on.
#if !defined(PLUGGABLE_DEVICE_SUPPORTED) &&                              \
    (__x86_64__ || __i386__ || defined(__APPLE__) || defined(_WIN32)) && \
    !defined(ANDROID) && !defined(__ANDROID__) && !TARGET_OS_IOS &&      \
    !defined(PLATFORM_CHROMIUMOS) && !defined(__Fuchsia__)
#define PLUGGABLE_DEVICE_SUPPORTED
#endif

#ifdef PLUGGABLE_DEVICE_SUPPORTED
#include "xla/stream_executor/stream.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

absl::Status TensorShapeFromTensor(const Tensor& t, PartialTensorShape* out);

absl::Status GetElementShapeFromInput(OpKernelContext* c,
                                      const TensorList& tensor_list, int index,
                                      PartialTensorShape* element_shape);

absl::Status GetInputList(OpKernelContext* c, int index,
                          const TensorList** list);

absl::Status ForwardInputOrCreateNewList(OpKernelContext* c,
                                         int32_t input_index,
                                         int32_t output_index,
                                         const TensorList& input_list,
                                         TensorList** output_list);

// TODO(penporn): Move this to a proper place.
inline bool IsPluggableDevice(OpKernelContext* c) {
  return c->op_device_context() && c->op_device_context()->IsPluggableDevice();
}

template <typename Device, typename T>
inline void SetZero(OpKernelContext* ctx, Tensor& tensor) {
#ifdef PLUGGABLE_DEVICE_SUPPORTED
  if (IsPluggableDevice(ctx)) {
    auto ptr =
        se::DeviceMemoryBase(tensor.flat<T>().data(), tensor.TotalBytes());
    auto stream = ctx->op_device_context()->stream();
    auto result = stream->MemZero(&ptr, tensor.TotalBytes()).ok();
    DCHECK_EQ(true, result);
  } else {
#endif  // PLUGGABLE_DEVICE_SUPPORTED
    functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                         tensor.flat<T>());
#ifdef PLUGGABLE_DEVICE_SUPPORTED
  }
#endif  // PLUGGABLE_DEVICE_SUPPORTED
}

template <typename T>
inline void CopyTensorPluggableDevice(OpKernelContext* ctx, Tensor& src,
                                      Tensor& dst) {
#ifdef PLUGGABLE_DEVICE_SUPPORTED
  auto src_t = src.unaligned_flat<T>();
  auto dst_t = dst.flat<T>();
  DCHECK(DataTypeCanUseMemcpy(DataTypeToEnum<T>::v()));
  auto src_ptr = se::DeviceMemoryBase(src_t.data(), src.TotalBytes());
  auto dst_ptr = se::DeviceMemoryBase(dst_t.data(), dst.TotalBytes());
  auto stream = ctx->op_device_context()->stream();
  auto result = stream->Memcpy(&dst_ptr, src_ptr, src.TotalBytes()).ok();
  DCHECK_EQ(true, result);
#else
  LOG(FATAL)  // Crash OK.
      << "PluggableDevice is not supported on this platform.";
#endif  // PLUGGABLE_DEVICE_SUPPORTED
}

template <typename Device, typename T>
inline void CopyTensor(OpKernelContext* ctx, Tensor& src, Tensor& dst) {
  auto src_t = src.unaligned_flat<T>();
  auto dst_t = dst.flat<T>();
  dst_t.device(ctx->eigen_device<Device>()) = src_t;
}

template <typename T>
void ConcatPluggableDevice(
    OpKernelContext* context,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
#ifdef PLUGGABLE_DEVICE_SUPPORTED
  DCHECK(DataTypeCanUseMemcpy(DataTypeToEnum<T>::v()));

  se::Stream* stream = context->op_device_context()->stream();

  size_t num_inputs = inputs.size();
  std::vector<ptrdiff_t> sizes;
  sizes.reserve(num_inputs);
  int64 row_size = 0;
  for (const auto& input : inputs) {
    sizes.push_back(input->dimension(1));
    row_size += sizes.back();
  }

  T* out = &(*output)(0, 0);
  std::vector<const T*> inp;
  inp.reserve(num_inputs);
  for (const auto& input : inputs) {
    inp.push_back(&(*input)(0, 0));
  }
  const int64 dim0 = output->dimension(0);
  for (int64 i = 0; i < dim0; ++i) {
    for (int64 j = 0; j < num_inputs; ++j) {
      auto size = sizes[j];
      se::DeviceMemoryBase out_base{out, size * sizeof(T)};
      se::DeviceMemoryBase inp_base{const_cast<T*>(inp[j]), size * sizeof(T)};
      OP_REQUIRES_OK(context,
                     stream->Memcpy(&out_base, inp_base, size * sizeof(T)));
      out += size;
      inp[j] += size;
    }
  }
#else
  LOG(FATAL)  // Crash OK.
      << "PluggableDevice is not supported on this platform.";
#endif  // PLUGGABLE_DEVICE_SUPPORTED
}

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
    const TensorList* tensor_list = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &tensor_list));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    if (num_elements_ != -1) {
      OP_REQUIRES(c, tensor_list->tensors().size() == num_elements_,
                  errors::InvalidArgument(
                      "Operation expected a list with ", num_elements_,
                      " elements but got a list with ",
                      tensor_list->tensors().size(), " elements."));
    }
    PartialTensorShape partial_element_shape;
    OP_REQUIRES_OK(c, GetElementShapeFromInput(c, *tensor_list, 1,
                                               &partial_element_shape));
    OP_REQUIRES(
        c,
        partial_element_shape.IsFullyDefined() ||
            !tensor_list->tensors().empty(),
        errors::InvalidArgument("Tried to stack elements of an empty ",
                                "list with non-fully-defined element_shape: ",
                                partial_element_shape.DebugString()));

    // Check that `element_shape` input tensor is compatible with the shapes of
    // element tensors.
    if (!tensor_list->element_shape.IsFullyDefined()) {
      for (int i = 0; i < tensor_list->tensors().size(); ++i) {
        const Tensor& t = tensor_list->tensors()[i];
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
    output_shape.InsertDim(0, tensor_list->tensors().size());
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(tensor_list->tensors().size());
    Tensor zeros;
    for (const auto& t : tensor_list->tensors()) {
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
          SetZero<Device, T>(c, zeros);
        }
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            const_cast<const Tensor&>(zeros).shaped<T, 2>(
                {1, zeros.NumElements()})));
      }
    }
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      ConcatGPU<T>(c, inputs_flat, output, &output_flat);
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (IsPluggableDevice(c)) {
      ConcatPluggableDevice<T>(c, inputs_flat, &output_flat);
    } else {
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
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
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
    OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));
    int32_t index = c->input(1).scalar<int32>()();
    OP_REQUIRES(c, index < l->tensors().size(),
                errors::InvalidArgument("Trying to access element ", index,
                                        " in a list with ", l->tensors().size(),
                                        " elements."));
    if (l->tensors()[index].dtype() != DT_INVALID) {
      c->set_output(0, l->tensors()[index]);
    } else {
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(
          c, GetElementShapeFromInput(c, *l, 2, &partial_element_shape));
      TensorShape element_shape;
      // If l->element_shape and the element_shape input are both not fully
      // defined, try to infer the shape from other list elements. This requires
      // that all initialized list elements have the same shape.
      // NOTE(srbs): This might be a performance bottleneck since we are
      // iterating over the entire list here. This is necessary for feature
      // parity with TensorArray.read. TensorArray has a mode in which all
      // elements are required to be of the same shape, TensorList does not.
      // In that mode TensorArray sets the array's element_shape on the first
      // write call. We could do something similar here if needed.
      if (!partial_element_shape.IsFullyDefined()) {
        for (const Tensor& t : l->tensors()) {
          if (t.dtype() != DT_INVALID) {
            PartialTensorShape tmp = partial_element_shape;
            OP_REQUIRES_OK(c, tmp.MergeWith(t.shape(), &partial_element_shape));
          }
        }
      }
      OP_REQUIRES(
          c, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument("Trying to read an uninitialized tensor but ",
                                  "element_shape is not fully defined: ",
                                  partial_element_shape.DebugString(),
                                  " and no list element is set."));
      Tensor* result;
      AllocatorAttributes attr;
      if (element_dtype_ == DT_VARIANT) {
        attr.set_on_host(true);
      }
      OP_REQUIRES_OK(c, c->allocate_output(0, element_shape, &result, attr));
      SetZero<Device, T>(c, *result);
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
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
    OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));

    OP_REQUIRES(c, !l->tensors().empty(),
                errors::InvalidArgument("Trying to pop from an empty list."));

    const Tensor& t = l->tensors().back();
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
      SetZero<Device, T>(c, *result);
    }

    TensorList* output_list = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewList(c, 0, 0, *l, &output_list));
    output_list->tensors().pop_back();
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
    if (c->HasAttr("element_shape")) {
      OP_REQUIRES_OK(c, c->GetAttr("element_shape", &element_shape_));
    }
  }

  void Compute(OpKernelContext* c) override {
    PartialTensorShape element_shape_except_first_dim;
    if (!element_shape_.unknown_rank()) {
      auto dim_sizes = element_shape_.dim_sizes();
      OP_REQUIRES(c, !dim_sizes.empty(),
                  errors::InvalidArgument("element_shape must not be empty"));
      element_shape_except_first_dim =
          PartialTensorShape(absl::Span<const int64_t>(dim_sizes).subspan(1));
    }
    // Check that the input Variant tensor is indeed a TensorList and has the
    // correct element type.
    const TensorList* tensor_list = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &tensor_list));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    // The leading dimension of all list elements if they are all the same.
    // This is used as the leading dim of uninitialized tensors in the list
    // if leading_dims is not provided.
    int64_t first_dim = -1;
    if (c->num_inputs() > 1) {
      // TensorListConcatV2
      PartialTensorShape element_shape;
      OP_REQUIRES_OK(
          c, GetElementShapeFromInput(c, *tensor_list, 1, &element_shape));
      OP_REQUIRES(c, element_shape.unknown_rank() || element_shape.dims() >= 1,
                  errors::InvalidArgument(
                      "Concat requires elements to be at least vectors, ",
                      "found scalars instead."));
      // Split `element_shape` into `first_dim` and
      // `element_shape_except_first_dim`.
      first_dim = element_shape.dim_size(0);
      element_shape_except_first_dim = element_shape;
      element_shape_except_first_dim.RemoveDim(0);
    }
    // If the TensorList is empty, element_shape_except_first_dim must be fully
    // defined.
    OP_REQUIRES(c,
                !tensor_list->tensors().empty() ||
                    element_shape_except_first_dim.IsFullyDefined(),
                errors::InvalidArgument(
                    "All except the first dimension must be fully defined ",
                    "when concating an empty tensor list. element_shape: ",
                    element_shape_except_first_dim.DebugString()));
    // 1. Check that `element_shape_except_first_dim` input tensor is
    //    compatible with the shapes of element tensors.
    // 2. Check that the elements have the same shape except the first dim.
    // 3. If `first_dim` is known, check that it is compatible with the leading
    //    dims of all elements.
    // 4. If `first_dim` is unknown (-1), check whether all initialized
    //    elements have the same leading dim and if so set `first_dim` to that
    //    value.
    if (!tensor_list->element_shape.IsFullyDefined()) {
      bool check_dim = (first_dim == -1);
      int64_t inferred_first_dim = first_dim;
      for (int i = 0; i < tensor_list->tensors().size(); ++i) {
        const Tensor& t = tensor_list->tensors()[i];
        if (t.dtype() != DT_INVALID) {
          PartialTensorShape tmp = element_shape_except_first_dim;
          OP_REQUIRES(
              c, TensorShapeUtils::IsVectorOrHigher(t.shape()),
              errors::InvalidArgument("Concat saw a scalar shape at index ", i,
                                      " but requires at least vectors."));
          TensorShape shape_except_first_dim = TensorShape(
              absl::Span<const int64_t>(t.shape().dim_sizes()).subspan(1));
          OP_REQUIRES_OK(c, tmp.MergeWith(shape_except_first_dim,
                                          &element_shape_except_first_dim));
          OP_REQUIRES(c, first_dim == -1 || first_dim == t.shape().dim_size(0),
                      errors::InvalidArgument(
                          "First entry of element_shape input does not match ",
                          "the first dim of list element at index: ", i,
                          " Expected: ", first_dim,
                          " Actual: ", t.shape().dim_size(0)));
          if (check_dim) {
            if (inferred_first_dim == -1) {
              inferred_first_dim = t.shape().dim_size(0);
            } else if (inferred_first_dim != t.shape().dim_size(0)) {
              inferred_first_dim = -1;
              check_dim = false;
            }
          }
        }
      }
      first_dim = inferred_first_dim;
    }
    TensorShape output_shape;
    OP_REQUIRES(c, element_shape_except_first_dim.AsTensorShape(&output_shape),
                errors::InvalidArgument(
                    "Trying to concat list with only uninitialized tensors ",
                    "but element_shape_except_first_dim is not fully defined: ",
                    element_shape_except_first_dim.DebugString()));
    // Build the lengths_tensor and leading dim of the output tensor by
    // iterating over all element tensors.
    Tensor* lengths_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(1,
                                         TensorShape({static_cast<int64_t>(
                                             tensor_list->tensors().size())}),
                                         &lengths_tensor));
    auto lengths_tensor_vec = lengths_tensor->vec<int64_t>();
    int64_t leading_dim = 0;
    for (size_t i = 0; i < tensor_list->tensors().size(); i++) {
      int64_t dim;
      if (tensor_list->tensors()[i].dtype() != DT_INVALID) {
        dim = tensor_list->tensors()[i].shape().dim_size(0);
      } else {
        // If leading_dims is not provided or does not contain an entry for
        // index i use the inferred `first_dim` if set.
        if ((c->num_inputs() <= 2 || i >= c->input(2).NumElements()) &&
            first_dim != -1) {
          dim = first_dim;
        } else {
          OP_REQUIRES(c, c->num_inputs() > 2,
                      errors::InvalidArgument(
                          "Concating lists with uninitialized tensors is not ",
                          "supported in this version of TensorListConcat. ",
                          "Consider updating your GraphDef to run the newer ",
                          "version."));
          OP_REQUIRES(c, i < c->input(2).NumElements(),
                      errors::InvalidArgument(
                          "List contains uninitialized tensor at index ", i,
                          " but leading_dims has only ",
                          c->input(2).NumElements(), " elements."));
          dim = c->input(2).vec<int64_t>()(i);
        }
      }
      leading_dim += dim;
      lengths_tensor_vec(i) = dim;
    }
    output_shape.InsertDim(0, leading_dim);
    Tensor* output;
    // Allocate the output tensor and fill it up with the concated element
    // tensors.
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      return;
    }

    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(tensor_list->tensors().size());
    // Store the zeros tensors in a vector to prevent them from being GC'ed till
    // concat is complete.
    std::vector<Tensor> zeros_vec;
    for (int i = 0; i < tensor_list->tensors().size(); i++) {
      const Tensor& element_tensor = tensor_list->tensors()[i];
      if (element_tensor.dtype() != DT_INVALID) {
        if (element_tensor.NumElements() > 0) {
          inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
              element_tensor.shaped<T, 2>({1, element_tensor.NumElements()})));
        }
      } else {
        AllocatorAttributes attr;
        if (element_dtype_ == DT_VARIANT) {
          attr.set_on_host(true);
        }
        TensorShape element_shape = output_shape;
        element_shape.set_dim(0, lengths_tensor_vec(i));
        zeros_vec.emplace_back();
        Tensor& zeros = zeros_vec.back();
        OP_REQUIRES_OK(
            c, c->allocate_temp(element_dtype_, element_shape, &zeros, attr));
        SetZero<Device, T>(c, zeros);
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            const_cast<const Tensor&>(zeros).shaped<T, 2>(
                {1, zeros.NumElements()})));
      }
    }
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      ConcatGPU<T>(c, inputs_flat, output, &output_flat);
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (IsPluggableDevice(c)) {
      ConcatPluggableDevice<T>(c, inputs_flat, &output_flat);
    } else {
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }

 private:
  DataType element_dtype_;
  PartialTensorShape element_shape_;
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
    output_list.tensors().reserve(lengths.shape().dim_size(0));

    const auto copy_tensor = IsPluggableDevice(c)
                                 ? &CopyTensorPluggableDevice<T>
                                 : &CopyTensor<Device, T>;

    int64_t start = 0;
    int64_t end = 0;
    for (int i = 0; i < lengths.shape().dim_size(0); ++i) {
      int64_t length = lengths.vec<int64_t>()(i);
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
      copy_tensor(c, tmp, aligned);
      output_list.tensors().emplace_back(aligned);
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
    const TensorList* tensor_list = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &tensor_list));
    OP_REQUIRES(
        c, element_dtype_ == tensor_list->element_dtype,
        errors::InvalidArgument(
            "Invalid data types; op elements ", DataTypeString(element_dtype_),
            " but list elements ", DataTypeString(tensor_list->element_dtype)));
    const Tensor& indices = c->input(1);
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

        OP_REQUIRES(c, 0 <= i && i < tensor_list->tensors().size(),
                    absl::InvalidArgumentError(absl::StrCat(
                        "Trying to gather element ", i, " in a list with ",
                        tensor_list->tensors().size(), " elements.")));

        const Tensor& t = tensor_list->tensors()[i];
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
          c, i < tensor_list->tensors().size(),
          errors::InvalidArgument("Index ", i, " out o range; list only has ",
                                  tensor_list->tensors().size(), " elements."));
      const Tensor& t = tensor_list->tensors()[i];
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
          SetZero<Device, T>(c, zeros);
        }
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            const_cast<const Tensor&>(zeros).shaped<T, 2>(
                {1, zeros.NumElements()})));
      }
    }
    auto output_flat = output->shaped<T, 2>({1, output->NumElements()});

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (std::is_same<Device, Eigen::GpuDevice>::value) {
      ConcatGPU<T>(c, inputs_flat, output, &output_flat);
      return;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (IsPluggableDevice(c)) {
      ConcatPluggableDevice<T>(c, inputs_flat, &output_flat);
    } else {
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
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
    OP_REQUIRES(
        c, !TensorShapeUtils::IsMatrixOrHigher(c->input(1).shape()),
        errors::InvalidArgument(
            "TensorListFromTensor: element_shape must be at most rank 1 but ",
            "has the shape of ", c->input(1).shape().DebugString()));
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
    output_list.tensors().reserve(t.shape().dim_size(0));

    const auto copy_tensor = IsPluggableDevice(c)
                                 ? &CopyTensorPluggableDevice<T>
                                 : &CopyTensor<Device, T>;

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
      copy_tensor(c, tmp, aligned);
      output_list.tensors().push_back(aligned);
    }
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }
};

// Scatters values in `value` into `list`. Assumes that `indices` are valid.
template <typename Device, typename T>
absl::Status Scatter(OpKernelContext* c, const Tensor& value,
                     const Tensor& indices, TensorList* list) {
  const auto copy_tensor = IsPluggableDevice(c) ? &CopyTensorPluggableDevice<T>
                                                : &CopyTensor<Device, T>;
  for (int index = 0; index < indices.NumElements(); ++index) {
    const int i = indices.flat<int32>()(index);
    Tensor tmp = value.Slice(index, index + 1);
    TensorShape tmp_shape = tmp.shape();
    tmp_shape.RemoveDim(0);
    if (!tmp.CopyFrom(tmp, tmp_shape)) {
      return errors::Unknown("Unexpected shape error.");
    }
    // TODO(apassos) maybe not always align; but weird compiler bugs seem to
    // prevent this.
    Tensor aligned;
    TF_RETURN_IF_ERROR(c->allocate_temp(tmp.dtype(), tmp.shape(), &aligned));
    // TODO(apassos) do all slices in a single kernel invocation instead of
    // many small ones.
    copy_tensor(c, tmp, aligned);
    std::swap(list->tensors()[i], aligned);
  }
  return absl::OkStatus();
}

template <typename Device, typename T>
class TensorListScatterIntoExistingList : public OpKernel {
 public:
  TensorListScatterIntoExistingList(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
    const Tensor& input_tensor = c->input(1);
    const Tensor& indices = c->input(2);

    // Check that inputs are valid.
    OP_REQUIRES(c, input_tensor.dtype() == l->element_dtype,
                errors::InvalidArgument(
                    "Invalid data types; input tensor type: ",
                    DataTypeString(input_tensor.dtype()),
                    " list element_type: ", DataTypeString(l->element_dtype)));
    OP_REQUIRES(c, TensorShapeUtils::IsVectorOrHigher(input_tensor.shape()),
                errors::InvalidArgument(
                    "Tensor must be at least a vector, but saw shape: ",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(c, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument(
                    "Expected indices to be a vector, but received shape: ",
                    indices.shape().DebugString()));
    OP_REQUIRES(
        c, indices.NumElements() == input_tensor.shape().dim_size(0),
        errors::InvalidArgument(
            "Expected len(indices) == tensor.shape[0], but saw: ",
            indices.NumElements(), " vs. ", input_tensor.shape().dim_size(0)));

    // Resize the list if needed to accommodate all indices.
    TensorList* output_list = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewList(c, 0, 0, *l, &output_list));
    const auto indices_vec = indices.vec<int32>();
    int32_t max_index =
        (indices.NumElements() == 0)
            ? -1
            : *std::max_element(indices_vec.data(),
                                indices_vec.data() + indices.NumElements());
    if (max_index + 1 > output_list->tensors().size()) {
      output_list->tensors().resize(max_index + 1);
    }

    // Scatter the values.
    OP_REQUIRES_OK(c,
                   Scatter<Device, T>(c, input_tensor, indices, output_list));
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
    OP_REQUIRES(
        c, !TensorShapeUtils::IsMatrixOrHigher(c->input(2).shape()),
        errors::InvalidArgument(
            "TensorListScatter: element_shape must be at most rank 1 but has ",
            "the shape of ", c->input(2).shape().DebugString()));
    OP_REQUIRES_OK(c, TensorShapeFromTensor(c->input(2), &element_shape));
    // TensorListScatterV2 passes the num_elements input, TensorListScatter does
    // not.
    int num_elements = -1;
    if (c->num_inputs() >= 4) {
      OP_REQUIRES(c, TensorShapeUtils::IsScalar(c->input(3).shape()),
                  errors::InvalidArgument("num_elements must be a scalar"));
      num_elements = c->input(3).scalar<int>()();
    }
    OP_REQUIRES(c, num_elements >= -1,
                errors::InvalidArgument(
                    "TensorListScatter expects num_elements >= -1, found: ",
                    num_elements));
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
      int highest_index = -1;
      for (int index = 0; index < indices.NumElements(); ++index) {
        const int i = indices.flat<int32>()(index);
        OP_REQUIRES(
            c, i >= 0,
            errors::InvalidArgument(
                "Indices in TensorListScatter must all be non-negative."));
        OP_REQUIRES(c, num_elements == -1 || i < num_elements,
                    errors::InvalidArgument(
                        "TensorListScatter: Trying to scatter at index ", i,
                        " in list with size ", num_elements));
        if (i > highest_index) {
          highest_index = i;
        }
      }
      output_list.tensors().resize(std::max(highest_index + 1, num_elements),
                                   Tensor(DT_INVALID));
    }

    OP_REQUIRES_OK(c,
                   Scatter<Device, T>(c, input_tensor, indices, &output_list));
    output_tensor->scalar<Variant>()() = std::move(output_list);
  }
};

template <typename Device>
absl::Status TensorListBinaryAdd(OpKernelContext* c, const TensorList& a,
                                 const TensorList& b, TensorList* out) {
  return TensorListBinaryAdd(c, a, b, out, BinaryAddTensors<Device>);
}

template <typename Device>
absl::Status TensorListZerosLike(OpKernelContext* c, const TensorList& x,
                                 TensorList* y) {
  return TensorListZerosLike(c, x, y, ZerosLikeTensor<Device>);
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

    bool ok_to_alias = tls_alias != nullptr;
    if (tls_alias && tls_alias->dtype() == DT_VARIANT &&
        tls_alias->NumElements() > 0) {
      auto alias_t = tls_alias->flat<Variant>();
      for (int i = 0; i < tls_alias->NumElements(); ++i) {
        TensorList* tl_i = alias_t(i).get<TensorList>();
        if (tl_i == nullptr || !tl_i->RefCountIsOne()) {
          ok_to_alias = false;
          break;
        }
      }
    }
    const Tensor& tls = ok_to_alias ? *tls_alias : c->input(0);

    OP_REQUIRES(c, tls.dtype() == DT_VARIANT,
                errors::InvalidArgument(
                    "Expected input_handles dtype to be Variant, but saw: ",
                    DataTypeString(tls.dtype())));
    OP_REQUIRES(c, TensorShapeUtils::IsVector(tls_shape),
                errors::InvalidArgument(
                    "Expected input_handles to be a vector, but saw shape: ",
                    tls_shape.DebugString()));
    const int64_t batch_size = tls.NumElements();
    OP_REQUIRES(c, input.dim_size(0) == batch_size,
                errors::InvalidArgument(
                    "Expected tensor.shape[0] == input_handles.size, but saw ",
                    input.dim_size(0), " vs. ", batch_size));
    auto tls_t = tls.vec<Variant>();

    TensorShape input_element_shape = input.shape();
    input_element_shape.RemoveDim(0);
    std::vector<const TensorList*> tl_batch;
    for (int64_t b = 0; b < batch_size; ++b) {
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

    if (ok_to_alias) {
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

    for (int64_t b = 0; b < batch_size; ++b) {
      if (!ok_to_alias) {
        result_t(b) = tl_batch[b]->Copy();
      }
      TensorList* output = result_t(b).get<TensorList>();
      DCHECK(output != nullptr);
      Tensor frame;
      OP_REQUIRES_OK(
          c, c->allocate_temp(element_dtype_, input_element_shape, &frame));
      if (input_element_shape.num_elements() > 0) {
        auto frame_t = frame.flat<T>();
        // TODO(penporn): Get this if out of the batch loop.
        if (IsPluggableDevice(c)) {
          // The chip method need Eigen Device, so need to use Tensor.Slice
          // instead of chip for pluggable device. The input should be reshaped
          // to 2-D and so can be sliced by batch dim.
          auto input_t_shape =
              TensorShape({input_t.dimension(0), input_t.dimension(1)});
          auto input_reshaped = Tensor();
          OP_REQUIRES(c, input_reshaped.CopyFrom(input, input_t_shape),
                      errors::Unknown("Unexpected shape error."));

          auto input_batch = input_reshaped.Slice(b, b + 1);
          CopyTensorPluggableDevice<T>(c, input_batch, frame);
        } else {
          frame_t.device(c->eigen_device<Device>()) =
              input_t.template chip<0>(b);
        }
      }
      output->tensors().push_back(std::move(frame));
    }
  }

 private:
  DataType element_dtype_;
};

}  // namespace tensorflow

#undef PLUGGABLE_DEVICE_SUPPORTED
#endif  // TENSORFLOW_CORE_KERNELS_LIST_KERNELS_H_
