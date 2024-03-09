/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/gradients/tape/tape_operation.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/gradients.h"

namespace tensorflow {
namespace gradients {
TapeOperation::TapeOperation(AbstractOperation* parent_op, Tape* tape,
                             const GradientRegistry& registry)
    : AbstractOperation(kTape),
      parent_op_(parent_op),
      tape_(tape),
      registry_(registry) {
  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  // parent_op_->Ref();
}
void TapeOperation::Release() {
  // TODO(srbs): Change to Unref().
  delete this;
}
TapeOperation::~TapeOperation() {
  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  // parent_op->Unref();
}
Status TapeOperation::Reset(const char* op, const char* raw_device_name) {
  forward_op_.op_name = op;
  forward_op_.attrs.Reset(op);
  forward_op_.inputs.clear();
  forward_op_.outputs.clear();
  return parent_op_->Reset(op, raw_device_name);
}
const string& TapeOperation::Name() const { return parent_op_->Name(); }
const string& TapeOperation::DeviceName() const {
  return parent_op_->DeviceName();
}
Status TapeOperation::SetDeviceName(const char* name) {
  return parent_op_->SetDeviceName(name);
}
Status TapeOperation::AddInput(AbstractTensorHandle* input) {
  TF_RETURN_IF_ERROR(parent_op_->AddInput(input));
  forward_op_.inputs.push_back(input);
  return absl::OkStatus();
}
Status TapeOperation::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
  TF_RETURN_IF_ERROR(parent_op_->AddInputList(inputs));
  for (auto input : inputs) {
    forward_op_.inputs.push_back(input);
  }
  return absl::OkStatus();
}
Status TapeOperation::SetAttrString(const char* attr_name, const char* data,
                                    size_t length) {
  forward_op_.attrs.Set(attr_name, StringPiece(data, length));
  return parent_op_->SetAttrString(attr_name, data, length);
}
Status TapeOperation::SetAttrInt(const char* attr_name, int64_t value) {
  forward_op_.attrs.Set(attr_name, static_cast<int64_t>(value));
  return parent_op_->SetAttrInt(attr_name, value);
}
Status TapeOperation::SetAttrFloat(const char* attr_name, float value) {
  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrFloat(attr_name, value);
}
Status TapeOperation::SetAttrBool(const char* attr_name, bool value) {
  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrBool(attr_name, value);
}
Status TapeOperation::SetAttrType(const char* attr_name, DataType value) {
  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrType(attr_name, value);
}
Status TapeOperation::SetAttrShape(const char* attr_name, const int64_t* dims,
                                   const int num_dims) {
  if (num_dims > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Value specified for `", attr_name, "` has ",
                                   num_dims,
                                   " dimensions which is over the limit of ",
                                   TensorShape::MaxDimensions(), ".");
  }
  TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }

  forward_op_.attrs.Set(attr_name, proto);
  return parent_op_->SetAttrShape(attr_name, dims, num_dims);
}
Status TapeOperation::SetAttrFunction(const char* attr_name,
                                      const AbstractOperation* value) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunction has not been implemented yet.");
}
Status TapeOperation::SetAttrFunctionName(const char* attr_name,
                                          const char* value, size_t length) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionName has not been implemented "
      "yet.");
}
Status TapeOperation::SetAttrTensor(const char* attr_name,
                                    AbstractTensorInterface* tensor) {
  return tensorflow::errors::Unimplemented(
      "SetAttrTensor has not been implemented yet.");
}
Status TapeOperation::SetAttrStringList(const char* attr_name,
                                        const void* const* values,
                                        const size_t* lengths, int num_values) {
  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  forward_op_.attrs.Set(attr_name, v);
  return parent_op_->SetAttrStringList(attr_name, values, lengths, num_values);
}
Status TapeOperation::SetAttrFloatList(const char* attr_name,
                                       const float* values, int num_values) {
  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const float>(values, num_values));
  return parent_op_->SetAttrFloatList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrIntList(const char* attr_name,
                                     const int64_t* values, int num_values) {
  forward_op_.attrs.Set(
      attr_name, gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));
  return parent_op_->SetAttrIntList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrTypeList(const char* attr_name,
                                      const DataType* values, int num_values) {
  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const DataType>(values, num_values));
  return parent_op_->SetAttrTypeList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrBoolList(const char* attr_name,
                                      const unsigned char* values,
                                      int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const bool>(b.get(), num_values));
  return parent_op_->SetAttrBoolList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrShapeList(const char* attr_name,
                                       const int64_t** dims,
                                       const int* num_dims, int num_values) {
  std::unique_ptr<TensorShapeProto[]> proto(new TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > TensorShape::MaxDimensions()) {
      return errors::InvalidArgument(
          strings::StrCat("Value specified for `", attr_name, "` has ",
                          num_dims_i, " dimensions which is over the limit of ",
                          TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  forward_op_.attrs.Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return parent_op_->SetAttrShapeList(attr_name, dims, num_dims, num_values);
}
Status TapeOperation::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionList has not been "
      "implemented yet.");
}
AbstractOperation* TapeOperation::GetBackingOperation() { return parent_op_; }
Status TapeOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                              int* num_retvals) {
  TF_RETURN_IF_ERROR(parent_op_->Execute(retvals, num_retvals));
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(srbs): Manage refcount of ForwardOperation's inputs/outputs.
    forward_op_.outputs.push_back(retvals[i]);
  }
  // TODO(b/166669239): This is needed to support AttrBuilder::Get for string
  // attributes. Number type attrs and DataType attrs work fine without this.
  // Consider getting rid of this and making the behavior between number types
  // and string consistent.
  forward_op_.attrs.BuildNodeDef();
  // TODO(b/170307493): Populate skip_input_indices here.
  std::unique_ptr<GradientFunction> backward_fn;
  TF_RETURN_IF_ERROR(registry_.Lookup(forward_op_, &backward_fn));
  tape_->RecordOperation(forward_op_.inputs, forward_op_.outputs,
                         backward_fn.release(), parent_op_->Name());
  return absl::OkStatus();
}

}  // namespace gradients
}  // namespace tensorflow
