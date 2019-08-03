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

// EncodeProto is a TensorFlow Op which serializes tensors into
// arbitrary protobufs.
//
// See the docstring in ../ops/encode_proto_op.cc for usage of the op.
//
// This implementation writes the serialized format using a handful of
// calls from the WireFormatLite API.

#include <memory>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/descriptors.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {
namespace {

using ::tensorflow::protobuf::Descriptor;
using ::tensorflow::protobuf::DescriptorPool;
using ::tensorflow::protobuf::FieldDescriptor;
using ::tensorflow::protobuf::internal::WireFormatLite;
using ::tensorflow::protobuf::io::CodedOutputStream;
using ::tensorflow::protobuf::io::StringOutputStream;

// Computes the total serialized size for a packed repeated field. For
// fixed-size types this can just multiply, but for variable-sized types it has
// to iterate through the values in the tensor.
template <WireFormatLite::FieldType FieldType, typename TensorT>
size_t TotalPackedSize(const Tensor& input, int message_index, int size);

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_DOUBLE, double>(const Tensor& input,
                                                            int message_index,
                                                            int size) {
  return size * WireFormatLite::kDoubleSize;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_FLOAT, double>(const Tensor& input,
                                                           int message_index,
                                                           int size) {
  return size * WireFormatLite::kFloatSize;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_FLOAT, float>(const Tensor& input,
                                                          int message_index,
                                                          int size) {
  return size * WireFormatLite::kFloatSize;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_INT64, int64>(const Tensor& input,
                                                          int message_index,
                                                          int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::Int64Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_UINT64, uint64>(const Tensor& input,
                                                            int message_index,
                                                            int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<uint64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::UInt64Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_INT32, int64>(const Tensor& input,
                                                          int message_index,
                                                          int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::Int32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_INT32, int32>(const Tensor& input,
                                                          int message_index,
                                                          int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int32>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::Int32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_FIXED64, uint64>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kFixed64Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_FIXED32, uint64>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kFixed32Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_FIXED32, uint32>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kFixed32Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_BOOL, bool>(const Tensor& input,
                                                        int message_index,
                                                        int size) {
  return size * WireFormatLite::kBoolSize;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_UINT32, uint64>(const Tensor& input,
                                                            int message_index,
                                                            int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<uint64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::UInt32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_UINT32, uint32>(const Tensor& input,
                                                            int message_index,
                                                            int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<uint32>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::UInt32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_ENUM, int32>(const Tensor& input,
                                                         int message_index,
                                                         int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int32>();
  for (int64 i = 0; i < size; i++) {
    data_size +=
        WireFormatLite::EnumSize(input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SFIXED32, int32>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kSFixed32Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SFIXED32, int64>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kSFixed32Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SFIXED64, int64>(
    const Tensor& input, int message_index, int size) {
  return size * WireFormatLite::kSFixed64Size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SINT32, int32>(const Tensor& input,
                                                           int message_index,
                                                           int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int32>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::SInt32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SINT32, int64>(const Tensor& input,
                                                           int message_index,
                                                           int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::SInt32Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

template <>
size_t TotalPackedSize<WireFormatLite::TYPE_SINT64, int64>(const Tensor& input,
                                                           int message_index,
                                                           int size) {
  size_t data_size = 0;
  auto input_t = input.flat_inner_dims<int64>();
  for (int64 i = 0; i < size; i++) {
    data_size += WireFormatLite::SInt64Size(
        input_t(static_cast<int64>(message_index), i));
  }
  return data_size;
}

// Writes a possibly repeated primitive field. TensorFlow does not have unsigned
// types, so we decode them to signed and encode them back to unsigned.
template <typename TensorT, typename ProtoT,
          WireFormatLite::FieldType FieldType,
          void Writer(ProtoT, CodedOutputStream*)>
Status WriteField(const FieldDescriptor& field_desc, const Tensor& input,
                  int message_index, int size, CodedOutputStream* output) {
  auto wire_type = WireFormatLite::WireTypeForFieldType(
      WireFormatLite::FieldType(field_desc.type()));

  auto input_t = input.flat_inner_dims<TensorT>();
  if (field_desc.options().packed()) {
    // Write the tag for the packed field.
    WireFormatLite::WriteTag(field_desc.number(),
                             WireFormatLite::WIRETYPE_LENGTH_DELIMITED, output);

    // Write the total packed length.
    size_t data_size =
        TotalPackedSize<FieldType, TensorT>(input, message_index, size);
    output->WriteVarint32(data_size);

    // Write individual values.
    for (int64 i = 0; i < size; i++) {
      // Note implicit cast from signed to unsigned.
      const ProtoT& value = input_t(static_cast<int64>(message_index), i);
      Writer(value, output);
    }
  } else {
    for (int64 i = 0; i < size; i++) {
      WireFormatLite::WriteTag(field_desc.number(), wire_type, output);

      // Note implicit cast from signed to unsigned.
      const ProtoT& value = input_t(static_cast<int64>(message_index), i);
      Writer(value, output);
    }
  }
  return Status::OK();
}

// Writes a possibly repeated string, bytes, or message field.
template <typename T, void Writer(int, const T&, CodedOutputStream*)>
Status WriteVarLenField(const FieldDescriptor& field_desc, const Tensor& input,
                        int message_index, int size,
                        CodedOutputStream* output) {
  auto input_t = input.flat_inner_dims<T>();
  for (int64 i = 0; i < size; i++) {
    const T& value = input_t(static_cast<int64>(message_index), i);
    // TODO(nix): there doesn't seem to be an inlined version of
    // WireFormatLite::WriteString or its relatives, which might allow a
    // small speedup.
    Writer(field_desc.number(), value, output);
  }
  return Status::OK();
}

// Writes a group field. Groups are treated like submessages, but tag-delimited
// instead of length-delimited. WireFormatLite handles this differently so we
// code it ourselves.
Status WriteGroup(const FieldDescriptor& field_desc, const Tensor& input,
                  int message_index, int size, CodedOutputStream* output) {
  auto input_t = input.flat_inner_dims<string>();
  for (int64 i = 0; i < size; i++) {
    const string& value = input_t(static_cast<int64>(message_index), i);
    WireFormatLite::WriteTag(field_desc.number(),
                             WireFormatLite::WIRETYPE_START_GROUP, output);
    // Note the use of WriteRaw instead of WriteString to skip the length.
    output->WriteRaw(value.data(), value.size());
    WireFormatLite::WriteTag(field_desc.number(),
                             WireFormatLite::WIRETYPE_END_GROUP, output);
  }
  return Status::OK();
}

// Writes a (possibly repeated) field into an output stream. It is the caller's
// responsibility to ensure that the type of the input tensor is compatible with
// the type of the proto field descriptor, and that (message_index, size-1) is
// within bounds.
Status WriteField(const FieldDescriptor& field_desc, const Tensor& input,
                  int message_index, int size, CodedOutputStream* output) {
  DataType dtype = input.dtype();

  switch (field_desc.type()) {
    case WireFormatLite::TYPE_DOUBLE:
      return WriteField<double, double, WireFormatLite::TYPE_DOUBLE,
                        WireFormatLite::WriteDoubleNoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_FLOAT:
      switch (dtype) {
        case DataType::DT_FLOAT:
          return WriteField<float, float, WireFormatLite::TYPE_FLOAT,
                            WireFormatLite::WriteFloatNoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_DOUBLE:
          return WriteField<double, float, WireFormatLite::TYPE_FLOAT,
                            WireFormatLite::WriteFloatNoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_FLOAT for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_INT64:
      return WriteField<int64, protobuf_int64, WireFormatLite::TYPE_INT64,
                        WireFormatLite::WriteInt64NoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_UINT64:
      return WriteField<uint64, protobuf_uint64, WireFormatLite::TYPE_UINT64,
                        WireFormatLite::WriteUInt64NoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_INT32:
      switch (dtype) {
        case DataType::DT_INT64:
          return WriteField<int64, int32, WireFormatLite::TYPE_INT32,
                            WireFormatLite::WriteInt32NoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_INT32:
          return WriteField<int32, int32, WireFormatLite::TYPE_INT32,
                            WireFormatLite::WriteInt32NoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_INT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_FIXED64:
      return WriteField<uint64, protobuf_uint64, WireFormatLite::TYPE_FIXED64,
                        WireFormatLite::WriteFixed64NoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_FIXED32:
      switch (dtype) {
        case DataType::DT_UINT64:
          return WriteField<uint64, uint32, WireFormatLite::TYPE_FIXED32,
                            WireFormatLite::WriteFixed32NoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_UINT32:
          return WriteField<uint32, uint32, WireFormatLite::TYPE_FIXED32,
                            WireFormatLite::WriteFixed32NoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_FIXED32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_BOOL:
      return WriteField<bool, bool, WireFormatLite::TYPE_BOOL,
                        WireFormatLite::WriteBoolNoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_STRING:
      return WriteVarLenField<string, WireFormatLite::WriteString>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_GROUP:
      return WriteGroup(field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_MESSAGE:
      return WriteVarLenField<string, WireFormatLite::WriteBytes>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_BYTES:
      return WriteVarLenField<string, WireFormatLite::WriteBytes>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_UINT32:
      switch (dtype) {
        case DataType::DT_UINT64:
          return WriteField<uint64, uint32, WireFormatLite::TYPE_UINT32,
                            WireFormatLite::WriteUInt32NoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_UINT32:
          return WriteField<uint32, uint32, WireFormatLite::TYPE_UINT32,
                            WireFormatLite::WriteUInt32NoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_UINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_ENUM:
      return WriteField<int32, int32, WireFormatLite::TYPE_ENUM,
                        WireFormatLite::WriteEnumNoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_SFIXED32:
      switch (dtype) {
        case DataType::DT_INT64:
          return WriteField<int64, int32, WireFormatLite::TYPE_SFIXED32,
                            WireFormatLite::WriteSFixed32NoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_INT32:
          return WriteField<int32, int32, WireFormatLite::TYPE_SFIXED32,
                            WireFormatLite::WriteSFixed32NoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_SFIXED32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SFIXED64:
      return WriteField<int64, protobuf_int64, WireFormatLite::TYPE_SFIXED64,
                        WireFormatLite::WriteSFixed64NoTag>(
          field_desc, input, message_index, size, output);
    case WireFormatLite::TYPE_SINT32:
      switch (dtype) {
        case DataType::DT_INT64:
          return WriteField<int64, int32, WireFormatLite::TYPE_SINT32,
                            WireFormatLite::WriteSInt32NoTag>(
              field_desc, input, message_index, size, output);
        case DataType::DT_INT32:
          return WriteField<int32, int32, WireFormatLite::TYPE_SINT32,
                            WireFormatLite::WriteSInt32NoTag>(
              field_desc, input, message_index, size, output);
        default:
          return errors::DataLoss("Failed writing TYPE_SINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SINT64:
      return WriteField<int64, protobuf_int64, WireFormatLite::TYPE_SINT64,
                        WireFormatLite::WriteSInt64NoTag>(
          field_desc, input, message_index, size, output);
      // default: intentionally omitted in order to enable static checking.
  }
}

class EncodeProtoOp : public OpKernel {
 public:
  explicit EncodeProtoOp(OpKernelConstruction* context) : OpKernel(context) {
    string descriptor_source;
    OP_REQUIRES_OK(context,
                   context->GetAttr("descriptor_source", &descriptor_source));
    // We always get back a desc_pool, but we may not own it. If we own it,
    // owned_desc_pool_ will be filled in.
    DescriptorPool const* desc_pool;
    OP_REQUIRES_OK(context, GetDescriptorPool(context->env(), descriptor_source,
                                              &desc_pool, &owned_desc_pool_));

    string message_type;
    OP_REQUIRES_OK(context, context->GetAttr("message_type", &message_type));
    const Descriptor* message_desc =
        desc_pool->FindMessageTypeByName(message_type);
    OP_REQUIRES(context, message_desc != nullptr,
                errors::InvalidArgument("No descriptor found for message type ",
                                        message_type));

    OP_REQUIRES_OK(context, context->GetAttr("field_names", &field_names_));

    // Gather the field descriptors for the given field_names.
    field_descs_.resize(field_names_.size());
    for (int i = 0; i < field_names_.size(); i++) {
      const string& name = field_names_[i];
      auto field_desc = message_desc->FindFieldByName(name);
      OP_REQUIRES(context, field_desc != nullptr,
                  errors::InvalidArgument("Unknown field: ", name,
                                          " in message type ", message_type));

      field_descs_[i] = field_desc;
    }

    // Build a list of indices into field_descs sorted by increasing
    // field_number. This will be used to output fields in sorted order,
    // which is strongly encouraged when serializing protobufs.
    sorted_field_index_.resize(field_names_.size());
    // Start with the fields sorted by current index.
    for (int i = 0; i < field_names_.size(); i++) sorted_field_index_[i] = i;
    // Then sort the field indices by their proto field number.
    std::sort(sorted_field_index_.begin(), sorted_field_index_.end(),
              [this](int a, int b) -> bool {
                return field_descs_[a]->number() < field_descs_[b]->number();
              });
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sizes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sizes", &sizes_tensor));

    OpInputList values;
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values));

    OP_REQUIRES(ctx, field_descs_.size() == values.size(),
                errors::InvalidArgument(
                    "Length of inputs list must match field_names"));

    // Check the arguments for consistency.
    TensorShape common_prefix;
    int message_count = 0;
    for (int i = 0; i < field_descs_.size(); i++) {
      const Tensor& v = values[i];

      // The type of each value tensor must match the corresponding field.
      OP_REQUIRES(
          ctx,
          proto_utils::IsCompatibleType(field_descs_[i]->type(), v.dtype()),
          errors::InvalidArgument(
              "Incompatible type for field ", field_names_[i],
              ".  Saw dtype: ", DataTypeString(v.dtype()),
              " but field type is: ", field_descs_[i]->type_name()));

      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrixOrHigher(v.shape()),
          errors::InvalidArgument("Invalid shape for field ", field_names_[i],
                                  ".  Saw shape ", v.shape().DebugString(),
                                  " but it should be at least a matrix."));

      // All value tensors must have the same shape prefix (i.e. batch size).
      TensorShape shape_prefix = v.shape();
      shape_prefix.RemoveDim(shape_prefix.dims() - 1);

      // Do some initialization on the first input value. The rest will
      // have to match this one.
      if (i == 0) {
        OP_REQUIRES(ctx, v.dims() >= 1,
                    errors::InvalidArgument(
                        "Expected value to be at least a vector, saw shape: ",
                        v.shape().DebugString()));
        common_prefix = shape_prefix;
        message_count = common_prefix.num_elements();
      } else {
        OP_REQUIRES(ctx, shape_prefix == common_prefix,
                    errors::InvalidArgument(
                        "Values must match up to the last dimension"));
      }
    }

    TensorShape expected_sizes_shape = common_prefix;
    expected_sizes_shape.AddDim(field_descs_.size());

    OP_REQUIRES(ctx, sizes_tensor->shape() == expected_sizes_shape,
                errors::InvalidArgument(
                    "sizes should be batch_size + [len(field_names)].  Saw: ",
                    sizes_tensor->shape().DebugString(),
                    " but expected: ", expected_sizes_shape.DebugString()));

    auto sizes = sizes_tensor->flat_inner_dims<int32>();

    for (int i = 0; i < field_descs_.size(); ++i) {
      const Tensor& v = values[i];
      int max_size = v.dim_size(v.dims() - 1);

      // The last dimension of a value tensor must be greater than the
      // corresponding size in the sizes tensor.
      for (int message_index = 0; message_index < message_count;
           message_index++) {
        OP_REQUIRES(
            ctx, sizes(message_index, i) <= max_size,
            errors::InvalidArgument(
                "Size to write must not be larger than value tensor; but saw: ",
                sizes(message_index, i), " > ", max_size, " at message ",
                message_index, " field ", i));
      }
    }

    // This pointer is owned by the context.
    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, common_prefix, &output_tensor));

    auto bufs = output_tensor->flat<string>();
    for (int message_index = 0; message_index < message_count;
         message_index++) {
      // TODO(nix): possibly optimize allocation here by calling
      // `bufs(message_index).reserve(DEFAULT_BUF_SIZE)`.
      StringOutputStream output_string(&bufs(message_index));
      CodedOutputStream out(&output_string);
      // Write fields in ascending field_number order.
      for (int i : sorted_field_index_) {
        auto& field_desc = *field_descs_[i];
        const Tensor& v = values[i];
        int size = sizes(message_index, i);
        if (!size) continue;
        OP_REQUIRES_OK(ctx,
                       WriteField(field_desc, v, message_index, size, &out));
      }
    }
  }

 private:
  std::vector<string> field_names_;
  std::vector<const FieldDescriptor*> field_descs_;

  // Owned_desc_pool_ is null when using descriptor_source=local.
  std::unique_ptr<DescriptorPool> owned_desc_pool_;

  // Contains indices into field_names_, sorted by field number since that's the
  // order of writing.
  std::vector<int> sorted_field_index_;

  TF_DISALLOW_COPY_AND_ASSIGN(EncodeProtoOp);
};

REGISTER_KERNEL_BUILDER(Name("EncodeProto").Device(DEVICE_CPU), EncodeProtoOp);

}  // namespace
}  // namespace tensorflow
