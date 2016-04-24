/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/data_flow_ops.cc.

#define EIGEN_USE_THREADS

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/kernels/tensor_array.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

Status GetHandle(const string& input_name, OpKernelContext* ctx,
                 string* container, string* ta_handle) {
  {
    Tensor tensor;
    // Assuming that input_name is at position 0 for purposes of
    // has_input.
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, false));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Tensor array handle must be 2-element vector, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<string>();
    *container = h(0);
    *ta_handle = h(1);
  }
  return Status::OK();
}

Status GetTensorArray(const string& input_name, OpKernelContext* ctx,
                      TensorArray** tensor_array) {
  string container;
  string ta_handle;
  TF_RETURN_IF_ERROR(GetHandle(input_name, ctx, &container, &ta_handle));
  ResourceMgr* rm = ctx->step_resource_manager();
  if (rm == nullptr) return errors::Internal("No per-step resource manager.");
  TF_RETURN_IF_ERROR(rm->Lookup(container, ta_handle, tensor_array));
  return Status::OK();
}

Status SetupFlowControlInputs(OpKernelContext* ctx, bool set_output) {
  const Tensor* flow_control;
  TF_RETURN_IF_ERROR(ctx->input("flow_in", &flow_control));
  if (set_output) {
    TF_RETURN_IF_ERROR(ctx->set_output("flow_out", *flow_control));
  }
  return Status::OK();
}

// CREATION *******************************************************************

// Virtual class for shared behavior between TensorArrayOp and
// TensorArrayGradOp.
class TensorArrayCreationOp : public OpKernel {
 public:
  explicit TensorArrayCreationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor tensor_array_output_handle;

    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            tensorflow::DT_STRING, tensorflow::TensorShape({2}),
                            &tensor_array_output_handle, alloc_attr));
    // Store the handle in a container of the per-step RM.
    ResourceMgr* rm = ctx->step_resource_manager();
    OP_REQUIRES(ctx, rm != nullptr,
                errors::Internal("No per-step resource manager."));

    TensorArray* output_tensor_array;
    OP_REQUIRES_OK(ctx, CreateTensorArray(ctx, rm, &tensor_array_output_handle,
                                          &output_tensor_array));
    ctx->set_output_ref(0, output_tensor_array->mu(),
                        output_tensor_array->handle());
  }

 protected:
  virtual Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                                   Tensor* tensor_array_output_handle,
                                   TensorArray** output_tensor_array) = 0;
};

// A per-run local tensor array. The tensor array uses a "per-step" resource
// manager which ensures that correct garbage collection on error or
// successful completion.
class TensorArrayOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("dynamic_size", &dynamic_size_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("clear_after_read", &clear_after_read_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("tensor_array_name", &tensor_array_name_));
    if (tensor_array_name_ == "") tensor_array_name_ = name();
  }

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
    const Tensor* tensor_size;
    TF_RETURN_IF_ERROR(ctx->input("size", &tensor_size));

    if (!TensorShapeUtils::IsScalar(tensor_size->shape())) {
      return errors::InvalidArgument(
          "TensorArray size must be scalar, but had shape: ",
          tensor_size->shape().DebugString());
    }
    const int32 size = tensor_size->scalar<int32>()();

    auto handle = tensor_array_output_handle->flat<string>();
    handle(0) = "_tensor_arrays";
    handle(1) = tensor_array_name_;

    TensorArray* tensor_array = new TensorArray(
        dtype_, *tensor_array_output_handle, size, dynamic_size_,
        false /* multiple_writes_aggregate */, clear_after_read_);

    TF_RETURN_IF_ERROR(rm->Create(handle(0), tensor_array_name_, tensor_array));

    *output_tensor_array = tensor_array;

    return Status::OK();
  }

 private:
  DataType dtype_;
  bool dynamic_size_;
  bool clear_after_read_;
  string tensor_array_name_;  // The name used to create the TensorArray.

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArray").Device(DEVICE_CPU), TensorArrayOp);

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArray")                \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// GRADIENT *******************************************************************

class TensorArrayGradOp : public TensorArrayCreationOp {
 public:
  explicit TensorArrayGradOp(OpKernelConstruction* context)
      : TensorArrayCreationOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("source", &source_));
  }

  Status CreateTensorArray(OpKernelContext* ctx, ResourceMgr* rm,
                           Tensor* tensor_array_output_handle,
                           TensorArray** output_tensor_array) override {
    string container;
    string tensor_array_name;
    TF_RETURN_IF_ERROR(
        GetHandle("handle", ctx, &container, &tensor_array_name));

    if (container != "_tensor_arrays") {
      return errors::InvalidArgument(
          "Input container should be '_tensor_arrays',  but received '",
          container, "'");
    }

    auto output_handle = tensor_array_output_handle->flat<string>();
    output_handle(0) = "_tensor_array_grads";
    output_handle(1) = strings::StrCat(tensor_array_name, "@", source_);

    TensorArray* tensor_array;
    int32 array_size;
    TF_RETURN_IF_ERROR(rm->Lookup(container, tensor_array_name, &tensor_array));
    core::ScopedUnref unref(tensor_array);

    // Once gradients are being calculated, the forward TensorArray
    // may no longer be resized by new Writes.
    tensor_array->DisableDynamicSize();
    TF_RETURN_IF_ERROR(tensor_array->Size(&array_size));

    if (!tensor_array->GradientsAllowed()) {
      return errors::InvalidArgument(
          "Unable to create a gradients TensorArray for ", tensor_array_name,
          ".  Perhaps you used the multiple_writes_aggregate flag on a "
          "previous write?  Gradient calculation is impossible when multiple "
          "writes are performed to the same index.");
    }

    auto creator = [this, tensor_array, array_size,
                    tensor_array_output_handle](TensorArray** ret) {
      *ret = new TensorArray(
          tensor_array->ElemType(), *tensor_array_output_handle, array_size,
          false /* dynamic_size */, true /* multiple_writes_aggregate */,
          true /* close_after_read */);
      return Status::OK();
    };

    Status s = rm->LookupOrCreate<TensorArray>(
        output_handle(0), output_handle(1), output_tensor_array, creator);
    (*output_tensor_array)->Unref();

    return s;
  }

 private:
  // The gradient source for creating the given
  // gradient TensorArray.  This should be unique to each gradients
  // call.  Typical values look like "gradients", "gradients_1", ...
  string source_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorArrayGradOp);
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad").Device(DEVICE_CPU),
                        TensorArrayGradOp);

REGISTER_KERNEL_BUILDER(Name("TensorArrayGrad")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("grad_handle"),
                        TensorArrayGradOp);

// WRITE **********************************************************************

template <typename Device, typename T>
class TensorArrayWriteOp : public OpKernel {
 public:
  explicit TensorArrayWriteOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    const Tensor* tensor_index;
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const int32 index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    PersistentTensor persistent_tensor(*tensor_value);
    Status s = tensor_array->WriteOrAggregate<Device, T>(ctx, index,
                                                         &persistent_tensor);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_WRITE(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("TensorArrayWrite").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayWriteOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_WRITE);

#undef REGISTER_WRITE

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWrite")       \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("handle")      \
                              .HostMemory("index"),      \
                          TensorArrayWriteOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// READ ***********************************************************************

class TensorArrayReadOp : public OpKernel {
 public:
  explicit TensorArrayReadOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    const Tensor* tensor_index;
    OP_REQUIRES_OK(ctx, ctx->input("index", &tensor_index));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_index->shape()),
                errors::InvalidArgument(
                    "TensorArray index must be scalar, but had shape: ",
                    tensor_index->shape().DebugString()));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);

    const int32 index = tensor_index->scalar<int32>()();
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));
    PersistentTensor value;
    OP_REQUIRES_OK(ctx, tensor_array->Read(index, &value));
    ctx->set_output(0, *value.AccessTensor(ctx));
  }

  bool IsExpensive() override { return false; }

 private:
  DataType dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayRead").Device(DEVICE_CPU),
                        TensorArrayReadOp);

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayRead")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle")          \
                              .HostMemory("index"),          \
                          TensorArrayReadOp);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// PACK ***********************************************************************

// Concatenate the elements in a TensorArray.  All elements must be
// defined and have the same shape.
template <typename Device, typename T>
class TensorArrayPackOp : public OpKernel {
 public:
  typedef typename TTypes<T, 2>::ConstMatrix ConstMatrix;
  typedef std::vector<std::unique_ptr<ConstMatrix> > ConstMatrixVector;

  explicit TensorArrayPackOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));

    core::ScopedUnref unref(tensor_array);
    int32 array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));

    // Simplest case
    if (array_size == 0) {
      Tensor empty(dtype_, TensorShape({}));
      ctx->set_output(0, empty);
      return;
    }

    // Read all the PersistentTensors into a vector to keep track of
    // their memory.
    std::vector<PersistentTensor> values;
    OP_REQUIRES_OK(ctx, tensor_array->ReadMany(&values));

    const Tensor* value_0_t = values[0].AccessTensor(ctx);
    TensorShape output_shape(value_0_t->shape());
    output_shape.InsertDim(0, array_size);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    ConstMatrixVector input_tensors_flat;
    input_tensors_flat.reserve(array_size);
    auto output_flat =
        output_tensor->shaped<T, 2>({1, output_shape.num_elements()});

    // Insert the first value
    input_tensors_flat.emplace_back(new ConstMatrix(
        value_0_t->shaped<T, 2>({1, value_0_t->NumElements()})));

    for (int i = 1; i < array_size; ++i) {
      const Tensor* value_t = values[i].AccessTensor(ctx);
      OP_REQUIRES(
          ctx, value_0_t->shape() == value_t->shape(),
          errors::InvalidArgument(
              "TensorArray has inconsistent shapes.  Index 0 has shape: ",
              value_0_t->shape().DebugString(), " but index ", i,
              " has shape: ", value_t->shape().DebugString()));
      input_tensors_flat.emplace_back(
          new ConstMatrix(value_t->shaped<T, 2>({1, value_t->NumElements()})));
    }

    if (std::is_same<Device, GPUDevice>::value) {
      // Switching indexing to int64 might cause performance issues.
      // Hence, we keep int32 indexing in the GPU kernel unless we need to
      // switch to int64.
      if (output_shape.num_elements() < std::numeric_limits<int32>::max()) {
        ConcatGPU32<T>(ctx->eigen_gpu_device(), input_tensors_flat,
                       &output_flat);
      } else {
        ConcatGPU64<T>(ctx->eigen_gpu_device(), input_tensors_flat,
                       &output_flat);
      }
    } else {
      ConcatCPU<T>(ctx->device(), input_tensors_flat, &output_flat);
    }
  }

 private:
  DataType dtype_;
};

#define REGISTER_PACK(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")            \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle"),         \
                          TensorArrayPackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_PACK);
REGISTER_PACK(quint8);
REGISTER_PACK(qint8);
REGISTER_PACK(qint32);
REGISTER_PACK(bfloat16);

#undef REGISTER_PACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")            \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle"),         \
                          TensorArrayPackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("TensorArrayPack")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("handle"),
                        TensorArrayPackOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

// CONCAT *********************************************************************

// Concatenate the elements in a TensorArray.  All elements must be
// defined and (excepting the first dimension) have the same shape.
template <typename Device, typename T>
class TensorArrayConcatOp : public OpKernel {
 public:
  typedef typename TTypes<T, 2>::ConstMatrix ConstMatrix;
  typedef std::vector<std::unique_ptr<ConstMatrix> > ConstMatrixVector;

  explicit TensorArrayConcatOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, false));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    OP_REQUIRES(
        ctx, dtype_ == tensor_array->ElemType(),
        errors::InvalidArgument(
            "TensorArray dtype is ", DataTypeString(tensor_array->ElemType()),
            " but Op requested dtype ", DataTypeString(dtype_), "."));

    int32 array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));

    // Simplest case
    if (array_size == 0) {
      Tensor empty(dtype_, TensorShape({}));
      ctx->set_output(0, empty);
      return;
    }

    // Read all the PersistentTensors into a vector to keep track of
    // their memory.
    std::vector<PersistentTensor> values;
    OP_REQUIRES_OK(ctx, tensor_array->ReadMany(&values));

    std::vector<const Tensor*> value_tensors;
    value_tensors.resize(values.size());

    Tensor* lengths_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1, TensorShape({static_cast<int64>(values.size())}),
                            &lengths_tensor));
    auto lengths_tensor_t = lengths_tensor->vec<int64>();

    TensorShape output_shape;
    TensorShape output_shape_except0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      value_tensors[i] = values[i].AccessTensor(ctx);
      TensorShape value_shape_t = value_tensors[i]->shape();

      OP_REQUIRES(
          ctx, TensorShapeUtils::IsVectorOrHigher(value_shape_t),
          errors::InvalidArgument(
              "Concat saw a scalar shape at index ", i,
              " but requires at least vectors.  Did you mean to call pack?"));

      lengths_tensor_t(i) = value_shape_t.dim_size(0);

      TensorShape value_shape_t_except0 = value_shape_t;
      value_shape_t_except0.RemoveDim(0);
      if (i == 0) {
        output_shape = value_shape_t;
        output_shape_except0 = value_shape_t_except0;
      } else {
        OP_REQUIRES(ctx, output_shape_except0 == value_shape_t_except0,
                    errors::InvalidArgument(
                        "TensorArray has inconsistent shapes.  Index 0 has "
                        "(excepting dimension 0) shape: ",
                        output_shape_except0.DebugString(), " but index ", i,
                        " has (excepting dimension 0) shape: ",
                        value_shape_t_except0.DebugString()));
        // Store the previous maximum length as the offset for this tensor.
        output_shape.set_dim(
            0, output_shape.dim_size(0) + value_shape_t.dim_size(0));
      }
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
    ConstMatrixVector input_tensors_flat;
    input_tensors_flat.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      const Tensor* value_t = value_tensors[i];
      if (value_t->NumElements() > 0) {
        input_tensors_flat.emplace_back(new ConstMatrix(
            value_t->shaped<T, 2>({1, value_t->NumElements()})));
      }
    }

    if (output_shape.num_elements() > 0) {
      auto output_flat =
          output_tensor->shaped<T, 2>({1, output_shape.num_elements()});
      if (std::is_same<Device, GPUDevice>::value) {
        // Switching indexing to int64 might cause performance issues.
        // Hence, we keep int32 indexing in the GPU kernel unless we need to
        // switch to int64.
        if (output_shape.num_elements() < std::numeric_limits<int32>::max()) {
          ConcatGPU32<T>(ctx->eigen_gpu_device(), input_tensors_flat,
                         &output_flat);
        } else {
          ConcatGPU64<T>(ctx->eigen_gpu_device(), input_tensors_flat,
                         &output_flat);
        }
      } else {
        ConcatCPU<T>(ctx->device(), input_tensors_flat, &output_flat);
      }
    }
  }

 private:
  DataType dtype_;
};

#define REGISTER_CONCAT(type)                                \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")          \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("lengths")         \
                              .HostMemory("handle"),         \
                          TensorArrayConcatOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_CONCAT);
REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(qint32);
REGISTER_CONCAT(bfloat16);

#undef REGISTER_CONCAT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")          \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("lengths")         \
                              .HostMemory("handle"),         \
                          TensorArrayConcatOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("TensorArrayConcat")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("dtype")
                            .HostMemory("lengths")
                            .HostMemory("handle"),
                        TensorArrayConcatOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA

// UNPACK *********************************************************************

template <typename Device, typename T>
class TensorArrayUnpackOp : public OpKernel {
 public:
  explicit TensorArrayUnpackOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));

    int32 array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));
    bool dynamic_size = tensor_array->HasDynamicSize();

    TensorShape element_shape(tensor_value->shape());

    OP_REQUIRES(ctx, FastBoundsCheck(element_shape.dim_size(0),
                                     std::numeric_limits<int32>::max()),
                errors::InvalidArgument("tensor dim0 too large to unpack"));

    // If dynamic size, we may have to resize the TensorArray to fit.
    if (dynamic_size && array_size < element_shape.dim_size(0)) {
      array_size = static_cast<int32>(element_shape.dim_size(0));
    }

    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));
    OP_REQUIRES(ctx, element_shape.dims() > 0,
                errors::InvalidArgument("Input value for unpack must be at "
                                        "least a vector but received shape: ",
                                        element_shape.DebugString()));
    OP_REQUIRES(
        ctx, element_shape.dim_size(0) == array_size,
        errors::InvalidArgument(
            "Input value must have first dimension equal to the array size (",
            element_shape.dim_size(0), " vs. ", array_size, ")"));
    element_shape.RemoveDim(0);

    auto tensor_value_t = tensor_value->shaped<T, 3>(
        {1, array_size, element_shape.num_elements()});

    Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, 0, 0};
    Eigen::DSizes<Eigen::DenseIndex, 3> sizes{1, 1,
                                              element_shape.num_elements()};

    std::vector<PersistentTensor> write_values;
    write_values.reserve(array_size);

    for (int i = 0; i < array_size; ++i) {
      Tensor* tensor_value_i;
      PersistentTensor persistent_tensor;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_persistent(tensor_array->ElemType(), element_shape,
                                        &persistent_tensor, &tensor_value_i));
      auto tensor_value_i_t =
          tensor_value_i->shaped<T, 3>({1, 1, element_shape.num_elements()});
      indices[1] = i;

      functor::Split<Device, T>()(ctx->eigen_device<Device>(), tensor_value_i_t,
                                  tensor_value_t, indices, sizes);

      write_values.push_back(persistent_tensor);
    }

    Status s =
        tensor_array->WriteOrAggregateMany<Device, T>(ctx, &write_values);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_UNPACK(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("TensorArrayUnpack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArrayUnpackOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_UNPACK);
#undef REGISTER_UNPACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayUnpack")      \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("handle"),     \
                          TensorArrayUnpackOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// SPLIT *********************************************************************

template <typename Device, typename T>
class TensorArraySplitOp : public OpKernel {
 public:
  explicit TensorArraySplitOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, SetupFlowControlInputs(ctx, true));

    TensorArray* tensor_array = nullptr;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    const Tensor* tensor_value;
    OP_REQUIRES_OK(ctx, ctx->input("value", &tensor_value));
    const Tensor* tensor_lengths;
    OP_REQUIRES_OK(ctx, ctx->input("lengths", &tensor_lengths));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(tensor_lengths->shape()),
                errors::InvalidArgument(
                    "Expected lengths to be a vector, received shape: ",
                    tensor_lengths->shape().DebugString()));
    OP_REQUIRES(ctx, FastBoundsCheck(tensor_lengths->NumElements(),
                                     std::numeric_limits<int32>::max()),
                errors::InvalidArgument(
                    "Expected lengths to have < max int32 entries"));

    int32 num_tensors = static_cast<int32>(tensor_lengths->NumElements());
    auto tensor_lengths_t = tensor_lengths->vec<int64>();
    std::vector<int64> cumulative_lengths;
    cumulative_lengths.reserve(num_tensors);
    int64 total_length = 0;
    for (int i = 0; i < num_tensors; ++i) {
      total_length += tensor_lengths_t(i);
      cumulative_lengths.push_back(total_length);
    }

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(tensor_value->shape()),
        errors::InvalidArgument(
            "Expected value to be at least a vector, but received shape: ",
            tensor_value->shape().DebugString()));

    OP_REQUIRES(
        ctx, total_length == tensor_value->shape().dim_size(0),
        errors::InvalidArgument("Expected sum of lengths to be equal to "
                                "values.shape[0], but sum of lengths is ",
                                total_length, " and value's shape is: ",
                                tensor_value->shape().DebugString()));
    int64 elements_per_row =
        (total_length == 0) ? 0 : (tensor_value->NumElements() / total_length);

    int32 array_size;
    OP_REQUIRES_OK(ctx, tensor_array->Size(&array_size));
    bool dynamic_size = tensor_array->HasDynamicSize();

    std::vector<TensorShape> element_shapes(num_tensors, tensor_value->shape());
    for (int32 i = 0; i < num_tensors; ++i) {
      element_shapes[i].set_dim(0, tensor_lengths_t(i));
    }

    // If dynamic size, we may have to resize the TensorArray to fit.
    if (dynamic_size && array_size < num_tensors) {
      array_size = num_tensors;
    }

    OP_REQUIRES(
        ctx, array_size == num_tensors,
        errors::InvalidArgument(
            "TensorArray's size is not equal to the size of lengths (",
            array_size, " vs. ", num_tensors, "), and the TensorArray is not ",
            "marked as dynamically resizeable"));

    OP_REQUIRES(
        ctx, tensor_value->dtype() == tensor_array->ElemType(),
        errors::InvalidArgument("TensorArray dtype is ",
                                DataTypeString(tensor_array->ElemType()),
                                " but Op is trying to write dtype ",
                                DataTypeString(tensor_value->dtype()), "."));

    auto tensor_value_t =
        tensor_value->shaped<T, 3>({1, total_length, elements_per_row});

    std::vector<PersistentTensor> write_values;
    write_values.reserve(array_size);

    for (int i = 0; i < array_size; ++i) {
      Tensor* tensor_value_i;
      PersistentTensor persistent_tensor;

      int64 previous_length = (i == 0) ? 0 : cumulative_lengths[i - 1];
      Eigen::DSizes<Eigen::DenseIndex, 3> indices{0, previous_length, 0};
      Eigen::DSizes<Eigen::DenseIndex, 3> sizes{1, tensor_lengths_t(i),
                                                elements_per_row};

      OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
                              tensor_array->ElemType(), element_shapes[i],
                              &persistent_tensor, &tensor_value_i));

      if (tensor_lengths_t(i) > 0) {
        auto tensor_value_i_t = tensor_value_i->shaped<T, 3>(
            {1, tensor_lengths_t(i), elements_per_row});

        functor::Split<Device, T>()(ctx->eigen_device<Device>(),
                                    tensor_value_i_t, tensor_value_t, indices,
                                    sizes);
      }

      write_values.push_back(persistent_tensor);
    }

    Status s =
        tensor_array->WriteOrAggregateMany<Device, T>(ctx, &write_values);
    OP_REQUIRES_OK(ctx, s);
  }
};

#define REGISTER_SPLIT(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("TensorArraySplit").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      TensorArraySplitOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_SPLIT);
#undef REGISTER_SPLIT

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArraySplit")       \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("lengths")     \
                              .HostMemory("handle"),     \
                          TensorArraySplitOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

// SIZE ***********************************************************************

// Get the size of the TensorArray
class TensorArraySizeOp : public OpKernel {
 public:
  explicit TensorArraySizeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    TensorArray* tensor_array;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, tensor_array->Size(&(output->scalar<int32>()())));
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorArraySize").Device(DEVICE_CPU),
                        TensorArraySizeOp);

REGISTER_KERNEL_BUILDER(Name("TensorArraySize")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .HostMemory("size"),
                        TensorArraySizeOp);

// CLOSE
// **********************************************************************

// Delete the TensorArray from its resource container.  This enables
// the user to close and release the resource in the middle of a step/run.
// TODO(ebrevdo): decide whether closing the grad op should happen
// here or on the python side.
class TensorArrayCloseOp : public OpKernel {
 public:
  explicit TensorArrayCloseOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    TensorArray* tensor_array;
    OP_REQUIRES_OK(ctx, GetTensorArray("handle", ctx, &tensor_array));
    core::ScopedUnref unref(tensor_array);
    // Instead of deleting this TA from the ResourceManager, we just
    // clear it away and mark it as closed.  The remaining memory
    // consumed store its mutex and handle Tensor.  This will be
    // cleared out at the end of the step anyway, so it's fine to keep
    // it around until the end of the step.  Further calls to the
    // TensorArray will fail because TensorArray checks internally to
    // see if it is closed or not.
    tensor_array->ClearAndMarkClosed();
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorArrayClose").Device(DEVICE_CPU),
                        TensorArrayCloseOp);

REGISTER_KERNEL_BUILDER(
    Name("TensorArrayClose").Device(DEVICE_GPU).HostMemory("handle"),
    TensorArrayCloseOp);

}  // namespace tensorflow
