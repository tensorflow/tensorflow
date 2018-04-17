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

#include <limits>

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/list_kernels.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Variant compatible type for a list of tensors. This is mutable but instances
// should never be mutated after stored in a variant tensor.
TensorList::TensorList(const TensorList& other)
    : tensors(other.tensors),
      element_shape(other.element_shape),
      element_dtype(other.element_dtype) {}

void TensorList::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  for (const Tensor& t : tensors) {
    *data->add_tensors() = t;
  }
  string metadata;
  core::PutVarint64(&metadata, static_cast<uint64>(element_dtype));
  if (!element_shape.unknown_rank()) {
    for (TensorShapeDim dim : element_shape) {
      if (dim.size > 0) {
        core::PutVarint64(&metadata, dim.size);
      } else {
        core::PutVarint64(&metadata, std::numeric_limits<uint64>::max());
      }
    }
  }
  data->set_metadata(metadata);
}

static Status TensorListDeviceCopy(
    const TensorList& from, TensorList* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  to->element_shape = from.element_shape;
  to->element_dtype = from.element_dtype;
  to->tensors.reserve(from.tensors.size());
  for (const Tensor& t : from.tensors) {
    Tensor tmp(t.dtype());
    TF_RETURN_IF_ERROR(copy(t, &tmp));
    to->tensors.push_back(tmp);
  }
  return Status::OK();
}

#define REGISTER_LIST_COPY(DIRECTION)                   \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      TensorList, DIRECTION, TensorList::kTypeName, TensorListDeviceCopy)

REGISTER_LIST_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TensorList, TensorList::kTypeName);

Status TensorListShape(const TensorList& t, TensorShape* s) {
  *s = TensorShape({});
  return Status::OK();
}

REGISTER_UNARY_VARIANT_SHAPE_FUNCTION(TensorList, TensorList::kTypeName,
                                      TensorListShape);

bool TensorList::Decode(const VariantTensorData& data) {
  tensors = data.tensors();
  string metadata;
  data.get_metadata(&metadata);
  uint64 scratch;
  StringPiece iter(metadata);
  core::GetVarint64(&iter, &scratch);
  element_dtype = static_cast<DataType>(scratch);
  std::vector<int64> dims;
  while (!iter.empty()) {
    core::GetVarint64(&iter, &scratch);
    if (scratch == std::numeric_limits<uint64>::max()) {
      dims.push_back(-1);
    } else {
      dims.push_back(scratch);
    }
  }
  element_shape = PartialTensorShape(dims);
  return true;
}

Status TensorShapeFromTensor(const Tensor& t, PartialTensorShape* out) {
  if (t.shape() == TensorShape({})) {
    if ((t.dtype() == DT_INT32 && t.scalar<int32>()() == -1) ||
        (t.dtype() == DT_INT64 && t.scalar<int64>()() == -1)) {
      return Status::OK();
    }
    return errors::InvalidArgument(
        "The only valid scalar shape tensor is the fully unknown shape "
        "specified as -1.");
  }
  if (t.dtype() == DT_INT32) {
    return PartialTensorShape::MakePartialShape(t.vec<int32>().data(),
                                                t.NumElements(), out);
  } else if (t.dtype() == DT_INT64) {
    return PartialTensorShape::MakePartialShape(t.vec<int64>().data(),
                                                t.NumElements(), out);
  }
  return errors::InvalidArgument(
      "Expected an int32 or int64 shape tensor; found ",
      DataTypeString(t.dtype()));
}

class EmptyTensorList : public OpKernel {
 public:
  explicit EmptyTensorList(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorList empty;
    empty.element_dtype = element_dtype_;
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(ctx, TensorShapeFromTensor(ctx->input(0), &element_shape));
    empty.element_shape = element_shape;
    result->scalar<Variant>()() = std::move(empty);
  }

 private:
  DataType element_dtype_;
};

const char TensorList::kTypeName[] = "tensorflow::TensorList";

REGISTER_KERNEL_BUILDER(Name("EmptyTensorList").Device(DEVICE_CPU),
                        EmptyTensorList);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("EmptyTensorList").Device(DEVICE_GPU).HostMemory("element_shape"),
    EmptyTensorList);

#endif  // GOOGLE_CUDA

class TensorListPushBack : public OpKernel {
 public:
  explicit TensorListPushBack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  ~TensorListPushBack() override {}

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(1);
    OP_REQUIRES(c, element_dtype_ == input.dtype(),
                errors::InvalidArgument("Invalid data types; list elements ",
                                        DataTypeString(element_dtype_),
                                        " but tried to append ",
                                        DataTypeString(input.dtype())));

    const TensorList* l = c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, l != nullptr,
                errors::InvalidArgument(
                    "Input handle is not a list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    OP_REQUIRES(c, l->element_shape.IsCompatibleWith(input.shape()),
                errors::InvalidArgument(
                    "Tried to append a tensor with incompatible shape to a "
                    "list. Op element shape: ",
                    input.shape().DebugString(),
                    " list shape: ", l->element_shape.DebugString()));
    OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));

    TensorList output;
    output = *l;
    output.tensors.push_back(input);
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    result->scalar<Variant>()() = std::move(output);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListPushBack").Device(DEVICE_CPU),
                        TensorListPushBack);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TensorListPushBack").Device(DEVICE_GPU),
                        TensorListPushBack);

#endif  // GOOGLE_CUDA

class TensorListLength : public OpKernel {
 public:
  explicit TensorListLength(OpKernelConstruction* c) : OpKernel(c) {}
  ~TensorListLength() override {}

  void Compute(OpKernelContext* c) override {
    const TensorList* l = c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(
        c, l != nullptr,
        errors::InvalidArgument(
            "TensorListLength received a variant which is not a list. Saw: '",
            c->input(0).scalar<Variant>()().DebugString(), "'"));
    Tensor* result;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result));
    result->scalar<int32>()() = l->tensors.size();
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorListLength").Device(DEVICE_CPU),
                        TensorListLength);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("TensorListLength").Device(DEVICE_GPU).HostMemory("length"),
    TensorListLength);

#endif  // GOOGLE_CUDA

class TensorListElementShape : public OpKernel {
 public:
  explicit TensorListElementShape(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    OP_REQUIRES(
        c, c->input(0).shape().num_elements() == 1,
        errors::InvalidArgument("List tensors are supposed to be scalars."));
    const TensorList* l = c->input(0).scalar<Variant>()().get<TensorList>();
    OP_REQUIRES(c, l != nullptr,
                errors::InvalidArgument(
                    "TensorListElementShape received a variant which is not a "
                    "list. Saw: '",
                    c->input(0).scalar<Variant>()().DebugString(), "'"));
    Tensor* result;
    OP_REQUIRES_OK(c, c->allocate_output(
                          0, TensorShape{l->element_shape.dims()}, &result));
    for (int i = 0; i < l->element_shape.dims(); ++i) {
      if (result->dtype() == DT_INT32) {
        result->flat<int32>()(i) = l->element_shape.dim_size(i);
      } else {
        result->flat<int64>()(i) = l->element_shape.dim_size(i);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorListElementShape").Device(DEVICE_CPU),
                        TensorListElementShape);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TensorListElementShape")
                            .Device(DEVICE_GPU)
                            .HostMemory("element_shape"),
                        TensorListElementShape);

#endif  // GOOGLE_CUDA

class TensorListPopBack : public OpKernel {
 public:
  explicit TensorListPopBack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  ~TensorListPopBack() override {}

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

    c->set_output(1, l->tensors.back());
    TensorList output;
    output = *l;
    output.tensors.pop_back();
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    result->scalar<Variant>()() = std::move(output);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListPopBack").Device(DEVICE_CPU),
                        TensorListPopBack);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TensorListPopBack").Device(DEVICE_GPU),
                        TensorListPopBack);

#endif  // GOOGLE_CUDA

class TensorListReserve : public OpKernel {
 public:
  explicit TensorListReserve(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    PartialTensorShape element_shape;
    OP_REQUIRES_OK(c, TensorShapeFromTensor(c->input(0), &element_shape));
    int32 num_elements = c->input(1).scalar<int32>()();
    TensorList output;
    output.element_shape = element_shape;
    output.element_dtype = element_dtype_;
    output.tensors.resize(num_elements, Tensor(DT_INVALID));
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    result->scalar<Variant>()() = std::move(output);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListReserve").Device(DEVICE_CPU),
                        TensorListReserve);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TensorListReserve")
                            .Device(DEVICE_GPU)
                            .HostMemory("element_shape")
                            .HostMemory("num_elements"),
                        TensorListReserve);

#endif  // GOOGLE_CUDA

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
    c->set_output(0, l->tensors[index]);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListGetItem").Device(DEVICE_CPU),
                        TensorListGetItem);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("TensorListGetItem").Device(DEVICE_GPU).HostMemory("index"),
    TensorListGetItem);

#endif  // GOOGLE_CUDA

class TensorListSetItem : public OpKernel {
 public:
  explicit TensorListSetItem(OpKernelConstruction* c) : OpKernel(c) {
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
    int32 index = c->input(1).scalar<int32>()();
    OP_REQUIRES(c, index < l->tensors.size(),
                errors::InvalidArgument("Trying to modify element ", index,
                                        " in a list with ", l->tensors.size(),
                                        " elements."));
    TensorList output;
    output = *l;
    output.tensors[index] = c->input(2);
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    result->scalar<Variant>()() = std::move(output);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListSetItem").Device(DEVICE_CPU),
                        TensorListSetItem);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("TensorListSetItem").Device(DEVICE_GPU).HostMemory("index"),
    TensorListSetItem);

#endif  // GOOGLE_CUDA

#define REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(T)               \
  REGISTER_KERNEL_BUILDER(Name("TensorListPushBackBatch")         \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_CPU),                \
                          TensorListPushBackBatch<CPUDevice, T>)

TF_CALL_ALL_TYPES(REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(quint8);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(qint8);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(quint16);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(qint16);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(qint32);
REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU(bfloat16);

#undef REGISTER_TENSOR_LIST_PUSH_BACK_BATCH_CPU

#define REGISTER_TENSOR_LIST_STACK_CPU(T)                         \
  REGISTER_KERNEL_BUILDER(Name("TensorListStack")                 \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_CPU),                \
                          TensorListStack<CPUDevice, T>)

TF_CALL_POD_STRING_TYPES(REGISTER_TENSOR_LIST_STACK_CPU);
REGISTER_TENSOR_LIST_STACK_CPU(quint8);
REGISTER_TENSOR_LIST_STACK_CPU(qint8);
REGISTER_TENSOR_LIST_STACK_CPU(quint16);
REGISTER_TENSOR_LIST_STACK_CPU(qint16);
REGISTER_TENSOR_LIST_STACK_CPU(qint32);
REGISTER_TENSOR_LIST_STACK_CPU(bfloat16);

#undef REGISTER_TENSOR_LIST_STACK_CPU

#define REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(T)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListFromTensor")            \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_CPU),                \
                          TensorListFromTensor<CPUDevice, T>)

TF_CALL_POD_STRING_TYPES(REGISTER_TENSOR_LIST_FROM_TENSOR_CPU);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(quint8);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(qint8);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(quint16);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(qint16);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(qint32);
REGISTER_TENSOR_LIST_FROM_TENSOR_CPU(bfloat16);

#undef REGISTER_TENSOR_LIST_FROM_TENSOR_CPU

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          TensorList, TensorList::kTypeName,
                                          TensorListBinaryAdd<CPUDevice>);

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, TensorList,
                                         TensorList::kTypeName,
                                         TensorListZerosLike<CPUDevice>);

}  // namespace tensorflow
