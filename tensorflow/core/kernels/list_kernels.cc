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
      element_dtype(other.element_dtype),
      max_num_elements(other.max_num_elements) {}

void TensorList::Encode(VariantTensorData* data) const {
  data->set_type_name(TypeName());
  std::vector<size_t> invalid_indices;
  for (size_t i = 0; i < tensors.size(); i++) {
    if (tensors.at(i).dtype() != DT_INVALID) {
      *data->add_tensors() = tensors.at(i);
    } else {
      invalid_indices.push_back(i);
    }
  }
  string metadata;
  // TODO(b/118838800): Add a proto for storing the metadata.
  // Metadata format:
  // <num_invalid_tensors><invalid_indices><element_dtype><element_shape_proto>
  core::PutVarint64(&metadata, static_cast<uint64>(invalid_indices.size()));
  for (size_t i : invalid_indices) {
    core::PutVarint64(&metadata, static_cast<uint64>(i));
  }
  core::PutVarint64(&metadata, static_cast<uint64>(element_dtype));
  core::PutVarint64(&metadata, static_cast<uint64>(max_num_elements));
  TensorShapeProto element_shape_proto;
  element_shape.AsProto(&element_shape_proto);
  element_shape_proto.AppendToString(&metadata);
  data->set_metadata(metadata);
}

static Status TensorListDeviceCopy(
    const TensorList& from, TensorList* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  to->element_shape = from.element_shape;
  to->element_dtype = from.element_dtype;
  to->max_num_elements = from.max_num_elements;
  to->tensors.reserve(from.tensors.size());
  for (const Tensor& t : from.tensors) {
    to->tensors.emplace_back(t.dtype());
    if (t.dtype() != DT_INVALID) {
      TF_RETURN_IF_ERROR(copy(t, &to->tensors.back()));
    }
  }
  return Status::OK();
}

#define REGISTER_LIST_COPY(DIRECTION)                                         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(TensorList, DIRECTION, \
                                                       TensorListDeviceCopy)

REGISTER_LIST_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TensorList, TensorList::kTypeName);

bool TensorList::Decode(const VariantTensorData& data) {
  // TODO(srbs): Change the signature to Decode(VariantTensorData data) so
  // that we do not have to copy each tensor individually below. This would
  // require changing VariantTensorData::tensors() as well.
  string metadata;
  data.get_metadata(&metadata);
  uint64 scratch;
  StringPiece iter(metadata);
  std::vector<size_t> invalid_indices;
  core::GetVarint64(&iter, &scratch);
  size_t num_invalid_tensors = static_cast<size_t>(scratch);
  invalid_indices.resize(num_invalid_tensors);
  for (size_t i = 0; i < num_invalid_tensors; i++) {
    core::GetVarint64(&iter, &scratch);
    invalid_indices[i] = static_cast<size_t>(scratch);
  }

  size_t total_num_tensors = data.tensors().size() + num_invalid_tensors;
  tensors.reserve(total_num_tensors);
  std::vector<size_t>::iterator invalid_indices_it = invalid_indices.begin();
  std::vector<Tensor>::const_iterator tensors_it = data.tensors().begin();
  for (size_t i = 0; i < total_num_tensors; i++) {
    if (invalid_indices_it != invalid_indices.end() &&
        *invalid_indices_it == i) {
      tensors.emplace_back(Tensor(DT_INVALID));
      invalid_indices_it++;
    } else if (tensors_it != data.tensors().end()) {
      tensors.emplace_back(*tensors_it);
      tensors_it++;
    } else {
      // VariantTensorData is corrupted.
      return false;
    }
  }

  core::GetVarint64(&iter, &scratch);
  element_dtype = static_cast<DataType>(scratch);
  core::GetVarint64(&iter, &scratch);
  max_num_elements = static_cast<int>(scratch);
  TensorShapeProto element_shape_proto;
  element_shape_proto.ParseFromString(string(iter.data(), iter.size()));
  element_shape = PartialTensorShape(element_shape_proto);
  return true;
}

Status TensorShapeFromTensor(const Tensor& t, PartialTensorShape* out) {
  if (t.shape() == TensorShape({})) {
    if ((t.dtype() == DT_INT32 && t.scalar<int32>()() == -1) ||
        (t.dtype() == DT_INT64 && t.scalar<int64>()() == -1)) {
      *out = PartialTensorShape();
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

Status GetElementShapeFromInput(OpKernelContext* c,
                                const TensorList& tensor_list, int index,
                                PartialTensorShape* element_shape) {
  TF_RETURN_IF_ERROR(TensorShapeFromTensor(c->input(index), element_shape));
  // Check that `element_shape` and `tensor_list.element_shape` are
  // compatible and store the merged shape in `element_shape`.
  PartialTensorShape tmp = *element_shape;
  TF_RETURN_IF_ERROR(tmp.MergeWith(tensor_list.element_shape, element_shape));
  return Status::OK();
}

Status GetInputList(OpKernelContext* c, int index, const TensorList** list) {
  if (!TensorShapeUtils::IsScalar(c->input(index).shape())) {
    return errors::InvalidArgument("Input list must be a scalar saw: ",
                                   c->input(index).shape().DebugString());
  }
  const TensorList* l = c->input(index).scalar<Variant>()().get<TensorList>();
  if (l == nullptr) {
    return errors::InvalidArgument(
        "Input handle is not a list. Saw: '",
        c->input(index).scalar<Variant>()().DebugString(), "'");
  }
  *list = l;
  return Status::OK();
}

Status ForwardInputOrCreateNewList(OpKernelContext* c, int32 input_index,
                                   int32 output_index,
                                   const TensorList& input_list,
                                   bool allocator_attr_on_host,
                                   TensorList** output_list) {
  // Attempt to forward the input tensor to the output if possible.
  AllocatorAttributes attr;
  // Note: This is a workaround to get buffer forwarding to work on CPU.
  // Setting on_host=true blocks forwarding because of a mismatch between the
  // input and output allocator attributes.
  attr.set_on_host(allocator_attr_on_host);
  std::unique_ptr<Tensor> maybe_output =
      c->forward_input(input_index, output_index, DT_VARIANT, TensorShape{},
                       c->input_memory_type(input_index), attr);
  Tensor* output_tensor;
  if (maybe_output != nullptr) {
    // Woohoo, forwarding succeeded!
    output_tensor = maybe_output.get();
    c->set_output(output_index, *output_tensor);
  } else {
    // If forwarding is not possible allocate a new output tensor and copy
    // the `input_list` to it.
    TF_RETURN_IF_ERROR(
        c->allocate_output(output_index, {}, &output_tensor, attr));
    output_tensor->scalar<Variant>()() = input_list;
  }
  *output_list = output_tensor->scalar<Variant>()().get<TensorList>();
  return Status::OK();
}

class EmptyTensorList : public OpKernel {
 public:
  explicit EmptyTensorList(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& max_num_elements_t = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(max_num_elements_t.shape()),
        errors::InvalidArgument(
            "max_num_elements expected to be a scalar ",
            "but got shape: ", max_num_elements_t.shape().DebugString()));
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorList empty;
    empty.element_dtype = element_dtype_;
    empty.max_num_elements = max_num_elements_t.scalar<int32>()();
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

REGISTER_KERNEL_BUILDER(Name("EmptyTensorList")
                            .Device(DEVICE_GPU)
                            .HostMemory("element_shape")
                            .HostMemory("max_num_elements"),
                        EmptyTensorList);

#endif  // GOOGLE_CUDA

class TensorListPushBack : public OpKernel {
 public:
  explicit TensorListPushBack(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
    is_cpu_kernel_ = c->device_type().type() == DEVICE_CPU;
  }

  ~TensorListPushBack() override {}

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(1);
    OP_REQUIRES(c, element_dtype_ == input.dtype(),
                errors::InvalidArgument("Invalid data types; list elements ",
                                        DataTypeString(element_dtype_),
                                        " but tried to append ",
                                        DataTypeString(input.dtype())));

    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
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

    if (l->max_num_elements != -1) {
      OP_REQUIRES(
          c, l->tensors.size() < l->max_num_elements,
          errors::InvalidArgument("Tried to push item into a full list",
                                  " list size: ", l->tensors.size(),
                                  " max_num_elements: ", l->max_num_elements));
    }

    TensorList* output_list = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewList(c, 0, 0, *l, !is_cpu_kernel_,
                                                  &output_list));
    output_list->tensors.push_back(input);
  }

 private:
  DataType element_dtype_;
  bool is_cpu_kernel_;
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
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
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
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
    Tensor* result;
    if (l->element_shape.unknown_rank()) {
      OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &result));
      if (result->dtype() == DT_INT32) {
        result->scalar<int32>()() = -1;
      } else {
        result->scalar<int64>()() = -1;
      }
    } else {
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
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorListElementShape").Device(DEVICE_CPU),
                        TensorListElementShape);

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
class TensorListResize : public OpKernel {
 public:
  explicit TensorListResize(OpKernelConstruction* c) : OpKernel(c) {
    is_cpu_kernel_ = c->device_type().type() == DEVICE_CPU;
  }

  void Compute(OpKernelContext* c) override {
    const TensorList* input_list = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &input_list));
    int32 size = c->input(1).scalar<int32>()();
    OP_REQUIRES(
        c, size >= 0,
        errors::InvalidArgument(
            "TensorListSlice expects size to be non-negative. Got: ", size));

    AllocatorAttributes attr;
    if (!is_cpu_kernel_) {
      attr.set_on_host(true);
    }
    std::unique_ptr<Tensor> maybe_result = c->forward_input(
        0, 0, DT_VARIANT, TensorShape{}, c->input_memory_type(0), attr);
    if (maybe_result != nullptr) {
      maybe_result->scalar<Variant>()().get<TensorList>()->tensors.resize(
          size, Tensor(DT_INVALID));
      c->set_output(0, *maybe_result);
    } else {
      Tensor* result;
      OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
      TensorList output_list;
      output_list.element_shape = input_list->element_shape;
      output_list.element_dtype = input_list->element_dtype;
      output_list.max_num_elements = input_list->max_num_elements;
      if (size > input_list->tensors.size()) {
        output_list.tensors.insert(output_list.tensors.begin(),
                                   input_list->tensors.begin(),
                                   input_list->tensors.end());
        // Add DT_INVALID tensors to the end of the list if the requested size
        // is larger than the list length.
        output_list.tensors.resize(size, Tensor(DT_INVALID));
      } else {
        output_list.tensors.insert(output_list.tensors.begin(),
                                   input_list->tensors.begin(),
                                   input_list->tensors.begin() + size);
      }
      result->scalar<Variant>()() = std::move(output_list);
    }
  }

 private:
  bool is_cpu_kernel_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListResize").Device(DEVICE_CPU),
                        TensorListResize);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("TensorListResize").Device(DEVICE_GPU).HostMemory("size"),
    TensorListResize);

#endif  // GOOGLE_CUDA

class TensorListSetItem : public OpKernel {
 public:
  explicit TensorListSetItem(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
    is_cpu_kernel_ = c->device_type().type() == DEVICE_CPU;
  }

  void Compute(OpKernelContext* c) override {
    const TensorList* l = nullptr;
    OP_REQUIRES_OK(c, GetInputList(c, 0, &l));
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
    const Tensor& value = c->input(2);
    OP_REQUIRES(c, l->element_shape.IsCompatibleWith(value.shape()),
                errors::InvalidArgument(
                    "Tried to set a tensor with incompatible shape at a "
                    "list index. Item element shape: ",
                    value.shape().DebugString(),
                    " list shape: ", l->element_shape.DebugString()));
    TensorList* output_list = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewList(c, 0, 0, *l, !is_cpu_kernel_,
                                                  &output_list));
    output_list->tensors[index] = value;
  }

 private:
  DataType element_dtype_;
  bool is_cpu_kernel_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListSetItem").Device(DEVICE_CPU),
                        TensorListSetItem);

#if GOOGLE_CUDA

#define REGISTER_TENSOR_LIST_SET_ITEM_GPU(T)                      \
  REGISTER_KERNEL_BUILDER(Name("TensorListSetItem")               \
                              .TypeConstraint<T>("element_dtype") \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("index"),               \
                          TensorListSetItem);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_TENSOR_LIST_SET_ITEM_GPU);
TF_CALL_complex64(REGISTER_TENSOR_LIST_SET_ITEM_GPU);
TF_CALL_complex128(REGISTER_TENSOR_LIST_SET_ITEM_GPU);
TF_CALL_int32(REGISTER_TENSOR_LIST_SET_ITEM_GPU);
TF_CALL_int64(REGISTER_TENSOR_LIST_SET_ITEM_GPU);
REGISTER_TENSOR_LIST_SET_ITEM_GPU(bfloat16)
#undef REGISTER_TENSOR_LIST_SET_ITEM_GPU

#endif  // GOOGLE_CUDA

class TensorListConcatLists : public OpKernel {
 public:
  explicit TensorListConcatLists(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    const TensorShape& tl_a_shape = c->input(0).shape();
    const TensorShape& tl_b_shape = c->input(1).shape();
    OP_REQUIRES(
        c, tl_a_shape == tl_b_shape,
        errors::InvalidArgument("Incompatible input TensorList tensor shapes: ",
                                tl_a_shape.DebugString(), " vs. ",
                                tl_b_shape.DebugString()));
    AllocatorAttributes attr;
    std::unique_ptr<Tensor> tl_alias = c->forward_input(
        0 /*input_index*/, 0 /*output_index*/, DT_VARIANT, tl_a_shape,
        DEVICE_MEMORY /* input is always on DEVICE_MEMORY */, attr);

    // tl_a may be aliased by tl_alias.
    const Tensor& tl_a = c->input(0);
    const Tensor& tl_b = c->input(1);

    Tensor* output;
    if (tl_alias) {
      c->set_output(0, *tl_alias);
      output = tl_alias.get();
    } else {
      attr.set_on_host(true);
      OP_REQUIRES_OK(c, c->allocate_output(0, tl_a_shape, &output, attr));
    }

    auto output_t = output->flat<Variant>();
    auto tl_a_t = tl_a.flat<Variant>();
    auto tl_b_t = tl_b.flat<Variant>();

    for (int64 b = 0; b < tl_a.NumElements(); ++b) {
      const TensorList* l_a = tl_a_t(b).get<TensorList>();
      const TensorList* l_b = tl_b_t(b).get<TensorList>();
      OP_REQUIRES(
          c, l_a != nullptr,
          errors::InvalidArgument("input_a is not a TensorList at index ", b,
                                  ".  Saw: '", tl_a_t(b).DebugString(), "'"));
      OP_REQUIRES(
          c, l_b != nullptr,
          errors::InvalidArgument("input_b is not a TensorList at index ", b,
                                  ".  Saw: '", tl_b_t(b).DebugString(), "'"));
      OP_REQUIRES(c, l_a->element_dtype == element_dtype_,
                  errors::InvalidArgument(
                      "input_a[", b, "].dtype != element_dtype.  Saw: ",
                      DataTypeString(l_a->element_dtype), " vs. ",
                      DataTypeString(element_dtype_)));
      OP_REQUIRES(c, l_b->element_dtype == element_dtype_,
                  errors::InvalidArgument(
                      "input_b[", b, "].dtype != element_dtype.  Saw: ",
                      DataTypeString(l_b->element_dtype), " vs. ",
                      DataTypeString(element_dtype_)));
      OP_REQUIRES(c, l_a->element_shape.IsIdenticalTo(l_b->element_shape),
                  errors::InvalidArgument(
                      "input_a and input_b TensorList element shapes are not "
                      "identical at index ",
                      b, ".  Saw ", l_a->element_shape.DebugString(), " vs. ",
                      l_b->element_shape.DebugString()));
      if (tl_alias) {
        TensorList* out = output_t(b).get<TensorList>();
        DCHECK(out != nullptr) << "Expected output to alias input_a, but it "
                                  "doesn't contain a TensorList at index "
                               << b;
        std::copy(l_b->tensors.begin(), l_b->tensors.end(),
                  std::back_inserter(out->tensors));
      } else {
        TensorList out = *l_a;
        std::copy(l_b->tensors.begin(), l_b->tensors.end(),
                  std::back_inserter(out.tensors));
        output_t(b) = std::move(out);
      }
    }
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorListConcatLists").Device(DEVICE_CPU),
                        TensorListConcatLists);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TensorListConcatLists").Device(DEVICE_GPU),
                        TensorListConcatLists);

#endif  // GOOGLE_CUDA

#define REGISTER_TENSOR_LIST_OPS_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("TensorListStack")                          \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListStack<CPUDevice, T>)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListGather")                         \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListGather<CPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListConcat")                         \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListConcat<CPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListConcatV2")                       \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListConcat<CPUDevice, T>)                  \
  REGISTER_KERNEL_BUILDER(Name("TensorListGetItem")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListGetItem<CPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListPopBack")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListPopBack<CPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListFromTensor")                     \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListFromTensor<CPUDevice, T>)              \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatter")                        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListScatter<CPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatterV2")                      \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListScatter<CPUDevice, T>)                 \
  REGISTER_KERNEL_BUILDER(Name("TensorListScatterIntoExistingList")        \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListScatterIntoExistingList<CPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("TensorListSplit")                          \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListSplit<CPUDevice, T>)                   \
  REGISTER_KERNEL_BUILDER(Name("TensorListPushBackBatch")                  \
                              .TypeConstraint<T>("element_dtype")          \
                              .Device(DEVICE_CPU),                         \
                          TensorListPushBackBatch<CPUDevice, T>)

TF_CALL_POD_STRING_TYPES(REGISTER_TENSOR_LIST_OPS_CPU);
REGISTER_TENSOR_LIST_OPS_CPU(quint8);
REGISTER_TENSOR_LIST_OPS_CPU(qint8);
REGISTER_TENSOR_LIST_OPS_CPU(quint16);
REGISTER_TENSOR_LIST_OPS_CPU(qint16);
REGISTER_TENSOR_LIST_OPS_CPU(qint32);
REGISTER_TENSOR_LIST_OPS_CPU(Variant);

#undef REGISTER_TENSOR_LIST_OPS_CPU

#define REGISTER_TENSOR_LIST_OPS_CPU(T)

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(ADD_VARIANT_BINARY_OP, DEVICE_CPU,
                                          TensorList,
                                          TensorListBinaryAdd<CPUDevice>);

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP,
                                         DEVICE_CPU, TensorList,
                                         TensorListZerosLike<CPUDevice>);

}  // namespace tensorflow
