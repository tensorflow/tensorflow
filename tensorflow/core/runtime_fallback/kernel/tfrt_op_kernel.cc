/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"

#include <optional>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime

namespace tensorflow {

//////////////////////////////////////////////////////////////////////
// OpKernel interface.
//////////////////////////////////////////////////////////////////////
TFRTOpKernelConstruction::TFRTOpKernelConstruction(
    const tfrt::OpAttrsRef& attributes)
    : attributes_(std::move(attributes)) {}

Status MissingAttributeError(StringPiece attr_name) {
  return errors::InvalidArgument("Missing attribute: ", attr_name);
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::string* value) const {
  tfrt::string_view view;
  bool success = attributes_.GetString(
      llvm::StringRef(attr_name.data(), attr_name.size()), &view);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = view.str();
  return OkStatus();
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         DataType* value) const {
  tfrt::OpAttrType attrtype;
  bool success = attributes_.Get<tfrt::OpAttrType>(
      llvm::StringRef(attr_name.data(), attr_name.size()), &attrtype);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = tfd::ConvertToTfDataType(attrtype);
  return OkStatus();
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         Padding* value) const {
  std::string padding_str;
  TF_RETURN_IF_ERROR(GetAttr<std::string>(attr_name, &padding_str));
  return GetPaddingFromString(padding_str, value);
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::vector<int32>* value) const {
  llvm::ArrayRef<int32> arrayref;
  bool success = attributes_.GetArray<int32>(
      llvm::StringRef(attr_name.data(), attr_name.size()), &arrayref);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = arrayref;
  return OkStatus();
}

void TFRTOpKernelConstruction::CtxFailure(const Status& s) {
  error_ = tfrt::MakeStatusString(s);
}

void TFRTOpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  CtxFailure(s);
}

namespace {
std::string FillFailureMessage(const char* file, int line, const Status& s) {
  std::string error;
  llvm::raw_string_ostream sstr(error);
  sstr << "OP_REQUIRES failed at " << file << ":" << line << " : "
       << tfrt::MakeStatusString(s);
  sstr.str();
  return error;
}
}  // namespace

void TFRTOpKernelConstruction::CtxFailure(const char* file, int line,
                                          const Status& s) {
  error_ = FillFailureMessage(file, line, s);
}

void TFRTOpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                     const Status& s) {
  CtxFailure(file, line, s);
}

const std::optional<std::string>& TFRTOpKernelConstruction::error() {
  return error_;
}

TFRTOpKernelContext::TFRTOpKernelContext(
    llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs, int num_outputs,
    const TFRTOpMeta* op_meta, tfrt::HostContext* host)
    : inputs_(inputs),
      op_meta_(op_meta),
      outputs_(num_outputs),
      eigen_host_context_(host) {}

const Tensor& TFRTOpKernelContext::output(int index) { return outputs_[index]; }

const std::optional<std::string>& TFRTOpKernelContext::error() {
  return error_;
}

bool TFRTOpKernelContext::ValidateInputsAreSameShape(TFRTOpKernel* op) {
  // TODO(lauj) Check shapes.
  return true;
}

const Tensor& TFRTOpKernelContext::input(int index) {
  return inputs_[index]->get<Tensor>();
}

int TFRTOpKernelContext::num_inputs() const { return inputs_.size(); }

int TFRTOpKernelContext::num_outputs() const { return outputs_.size(); }

void TFRTOpKernelContext::set_output(int index, const Tensor& tensor) {
  outputs_[index] = tensor;
}

Status TFRTOpKernelContext::allocate_temp(DataType type,
                                          const TensorShape& shape,
                                          Tensor* out_temp) {
  *out_temp = Tensor(type, shape);
  return OkStatus();
}

Status TFRTOpKernelContext::allocate_output(int index, const TensorShape& shape,
                                            Tensor** tensor) {
  // Fetch output DataType from the op's TFRTOpMeta.
  DataType output_type = op_meta_->output_type(index);
  outputs_[index] = Tensor(output_type, shape);
  *tensor = &outputs_[index];
  return OkStatus();
}

DataType TFRTOpKernelContext::expected_output_dtype(int i) const {
  return op_meta_->output_type(i);
}

void TFRTOpKernelContext::CtxFailure(const Status& s) { error_ = s.message(); }
void TFRTOpKernelContext::CtxFailureWithWarning(const Status& s) {
  CtxFailure(s);
}
void TFRTOpKernelContext::CtxFailure(const char* file, int line,
                                     const Status& s) {
  error_ = FillFailureMessage(file, line, s);
}
void TFRTOpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                                const Status& s) {
  CtxFailure(file, line, s);
}

template <>
const Eigen::ThreadPoolDevice& TFRTOpKernelContext::eigen_device() const {
  return eigen_host_context_.Device();
}

//////////////////////////////////////////////////////////////////////
// Forwarding op metadata.
//////////////////////////////////////////////////////////////////////
TFRTOpMeta::TFRTOpMeta(std::vector<DataType> output_types)
    : output_types_(std::move(output_types)) {}

DataType TFRTOpMeta::output_type(int index) const {
  return output_types_[index];
}

TFRTOpMetaBuilder::TFRTOpMetaBuilder(StringPiece op_name) : op_name_(op_name) {}

namespace {

DataType ParseInputOutputSpec(StringPiece spec) {
  std::vector<absl::string_view> name_type =
      absl::StrSplit(spec, absl::MaxSplits(':', 2));
  DataType data_type;
  bool success =
      DataTypeFromString(absl::StripAsciiWhitespace(name_type[1]), &data_type);
  assert(success && "Failed to parse DataType");
  (void)success;
  return data_type;
}

}  // anonymous namespace

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Output(StringPiece output_spec) {
  output_types_.push_back(ParseInputOutputSpec(output_spec));
  return *this;
}

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Input(StringPiece input_spec) {
  return *this;
}

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Attr(StringPiece attr_spec) {
  return *this;
}

const string& TFRTOpMetaBuilder::op_name() const { return op_name_; }

TFRTOpMeta TFRTOpMetaBuilder::BuildMeta() const {
  return TFRTOpMeta(output_types_);
}

TFRTOpMetaMap::TFRTOpMetaMap() {}

void TFRTOpMetaMap::RegisterOpMeta(const TFRTOpMetaBuilder& op_builder) {
  auto insert_result = op_metas_.insert(
      std::make_pair(op_builder.op_name(), op_builder.BuildMeta()));
  assert(insert_result.second && "Multiple registrations for the same op_name");
  (void)insert_result;
}

const TFRTOpMeta* TFRTOpMetaMap::GetOpMeta(StringPiece op_name) const {
  auto it = op_metas_.find(llvm::StringRef(op_name.data(), op_name.size()));
  if (it == op_metas_.end()) return nullptr;

  return &it->second;
}

TFRTOpRegisterer::TFRTOpRegisterer(const TFRTOpMetaBuilder& op_builder) {
  tfrt_forwarding_op_meta_map->RegisterOpMeta(op_builder);
}

llvm::ManagedStatic<TFRTOpMetaMap> tfrt_forwarding_op_meta_map;

llvm::ManagedStatic<TFRTOpKernelFactories> tfrt_forwarding_kernel_factories;

//////////////////////////////////////////////////////////////////////
// Forwarding kernel registration.
//////////////////////////////////////////////////////////////////////

TFRTOpKernelFactories::TFRTOpKernelFactories() {}

void TFRTOpKernelFactories::RegisterFactory(StringPiece kernel_class_name,
                                            TFRTOpKernelReg kernel_info) {
  factories_[std::string(kernel_class_name)].push_back(kernel_info);
}

// Returns true if kernel attributes match given type constraints.
Status ValidKernelAttr(StringPiece kernel_class_name,
                       TFRTOpKernelConstruction* construction,
                       const llvm::StringMap<DataType>& constraints) {
  for (const auto& constraint : constraints) {
    auto attr_name = std::string(constraint.first());
    DataType type;
    Status s = construction->GetAttr(attr_name, &type);
    if (!s.ok()) {
      return errors::InvalidArgument(
          "Kernel ", kernel_class_name,
          " has constraint for unset tfdtype attribute ", attr_name, ".");
    }
    if (type != constraint.second) {
      return errors::InvalidArgument(
          "Kernel ", kernel_class_name, " with type constraint ", attr_name,
          ": ", DataTypeString(constraint.second),
          " does not match attribute type ", DataTypeString(type), ".");
    }
  }
  return OkStatus();
}

std::unique_ptr<TFRTOpKernel> TFRTOpKernelFactories::CreateKernel(
    StringPiece kernel_class_name,
    TFRTOpKernelConstruction* op_kernel_construction) const {
  auto it = factories_.find(std::string(kernel_class_name));
  if (it == factories_.end()) {
    // Could not find kernel in the registry
    op_kernel_construction->CtxFailure(errors::NotFound(
        "Could not find kernel ", kernel_class_name, " in the registry."));
    return std::unique_ptr<TFRTOpKernel>(nullptr);
  }
  Status status;
  for (const auto& kernel_info : it->second) {
    Status s = ValidKernelAttr(kernel_class_name, op_kernel_construction,
                               kernel_info.type_constraints);
    if (s.ok()) {
      return kernel_info.callback(op_kernel_construction);
    }
    status.Update(s);
  }
  // No valid kernel found
  op_kernel_construction->CtxFailure(status);
  return std::unique_ptr<TFRTOpKernel>(nullptr);
}

}  // namespace tensorflow
