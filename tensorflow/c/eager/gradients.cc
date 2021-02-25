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
#include "tensorflow/c/eager/gradients.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace gradients {
namespace {

// TODO(b/172558015): Using the pointer address as the identifier for the tensor
// may lead to collisions. Introduce another way to get a unique id for this
// tensor.
int64 ToId(const AbstractTensorHandle* t) {
  return static_cast<int64>(reinterpret_cast<uintptr_t>(t));
}

Status ZerosLike(AbstractContext* ctx, AbstractTensorHandle* t,
                 AbstractTensorHandle** result) {
  AbstractOperationPtr op(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(op->Reset("ZerosLike", /*raw_device_name=*/nullptr));
  if (isa<tracing::TracingOperation>(op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(op.get())->SetOpName(
        absl::StrCat("ZerosLike", ToId(t)).c_str()));
  }
  TF_RETURN_IF_ERROR(op->AddInput(t));
  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs));
  *result = outputs[0];
  return Status::OK();
}
}  // namespace

Status GradientRegistry::Register(
    const string& op_name, GradientFunctionFactory gradient_function_factory) {
  auto iter = registry_.find(op_name);
  if (iter != registry_.end()) {
    const string error_msg = "Gradient already exists for op: " + op_name + ".";
    return errors::AlreadyExists(error_msg);
  }
  registry_.insert({op_name, gradient_function_factory});
  return Status::OK();
}
Status GradientRegistry::Lookup(
    const ForwardOperation& op,
    std::unique_ptr<GradientFunction>* gradient_function) const {
  auto iter = registry_.find(op.op_name);
  if (iter == registry_.end()) {
    const string error_msg = "No gradient defined for op: " + op.op_name + ".";
    return errors::NotFound(error_msg);
  }
  gradient_function->reset(iter->second(op));
  return Status::OK();
}

TapeTensor::TapeTensor(AbstractTensorHandle* handle) : handle_(handle) {
  handle_->Ref();
}
TapeTensor::TapeTensor(const TapeTensor& other) {
  handle_ = other.handle_;
  handle_->Ref();
}
TapeTensor::~TapeTensor() { handle_->Unref(); }

tensorflow::int64 TapeTensor::GetID() const { return ToId(handle_); }

tensorflow::DataType TapeTensor::GetDType() const {
  return handle_->DataType();
}
AbstractTensorHandle* TapeTensor::GetHandle() const { return handle_; }

AbstractTensorHandle* TapeTensor::ZerosLike() const { return nullptr; }

class TapeVSpace
    : public eager::VSpace<AbstractTensorHandle, GradientFunction, TapeTensor> {
 public:
  explicit TapeVSpace(AbstractContext* ctx) : ctx_(ctx) {}
  ~TapeVSpace() override {}

  // Returns the number of elements in the gradient tensor.
  int64 NumElements(AbstractTensorHandle* tensor) const override;

  // Consumes references to the tensors in the gradient_tensors list and returns
  // a tensor with the result.
  AbstractTensorHandle* AggregateGradients(
      gtl::ArraySlice<AbstractTensorHandle*> gradient_tensors) const override;

  // Calls the passed-in backward function.
  // op_type is the op's name provided in RecordOperation.
  Status CallBackwardFunction(
      const string& op_type, GradientFunction* gradient_function,
      const std::vector<int64>& unneeded_gradients,
      gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
      absl::Span<AbstractTensorHandle*> result) const override;

  // Builds a tensor filled with ones with the same shape and dtype as `t`.
  Status BuildOnesLike(const TapeTensor& t,
                       AbstractTensorHandle** result) const override;

  // Looks up the ID of a Gradient.
  int64 TensorId(AbstractTensorHandle* tensor) const override;

  // Converts a Gradient to a TapeTensor.
  TapeTensor TapeTensorFromGradient(AbstractTensorHandle* g) const override;

  void MarkAsResult(AbstractTensorHandle* gradient) const override;

  void DeleteGradient(AbstractTensorHandle* gradient) const override;

 private:
  // The context where the aggregation op `Add` is to be created.
  AbstractContext* ctx_;
};

// Returns the number of elements in the gradient tensor.
int64 TapeVSpace::NumElements(AbstractTensorHandle* tensor) const {
  // TODO(srbs): It seems like this is used only for performance optimization
  // and not for correctness. The only downside of keeping this 1 seems to be
  // that the gradient accumulation is unbounded and we will never
  // aggressively aggregate accumulated gradients to recover memory.
  // Revisit and fix.
  return 1;
}

// Consumes references to the tensors in the gradient_tensors list and returns
// a tensor with the result.
AbstractTensorHandle* TapeVSpace::AggregateGradients(
    gtl::ArraySlice<AbstractTensorHandle*> gradient_tensors) const {
  if (gradient_tensors.size() == 1) {
    return gradient_tensors[0];
  }

  AbstractOperationPtr op(ctx_->CreateOperation());
  Status s = op->Reset("AddN", /*raw_device_name=*/nullptr);
  if (!s.ok()) {
    return nullptr;
  }
  s = op->AddInputList(gradient_tensors);
  if (!s.ok()) {
    return nullptr;
  }

  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  s = op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs);
  if (!s.ok()) {
    return nullptr;
  }
  return outputs[0];
}

// Calls the passed-in backward function.
// op_type is the op's name provided in RecordOperation.
Status TapeVSpace::CallBackwardFunction(
    const string& op_type, GradientFunction* gradient_function,
    const std::vector<int64>& unneeded_gradients,
    gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
    absl::Span<AbstractTensorHandle*> result) const {
  if (gradient_function == nullptr) {
    return errors::InvalidArgument(
        "Provided null gradient_function for '", op_type, "'.\n",
        "If the intent is to treat this op as non-differentiable consider "
        "using RegisterNotDifferentiable or "
        "NotDifferentiableGradientFunction.");
  }
  return gradient_function->Compute(ctx_, output_gradients, result);
}

Status TapeVSpace::BuildOnesLike(const TapeTensor& t,
                                 AbstractTensorHandle** result) const {
  AbstractOperationPtr op(ctx_->CreateOperation());
  TF_RETURN_IF_ERROR(op->Reset("OnesLike", /*raw_device_name=*/nullptr));
  if (isa<tracing::TracingOperation>(op.get())) {
    TF_RETURN_IF_ERROR(dyn_cast<tracing::TracingOperation>(op.get())->SetOpName(
        absl::StrCat("OnesLike", ToId(t.GetHandle())).c_str()));
  }
  TF_RETURN_IF_ERROR(op->AddInput(t.GetHandle()));
  int num_outputs = 1;
  std::vector<AbstractTensorHandle*> outputs(num_outputs);
  TF_RETURN_IF_ERROR(
      op->Execute(absl::Span<AbstractTensorHandle*>(outputs), &num_outputs));
  *result = outputs[0];
  return Status::OK();
}

// Looks up the ID of a Gradient.
int64 TapeVSpace::TensorId(AbstractTensorHandle* tensor) const {
  return ToId(tensor);
}

// Converts a Gradient to a TapeTensor.
TapeTensor TapeVSpace::TapeTensorFromGradient(AbstractTensorHandle* g) const {
  return TapeTensor(g);
}

void TapeVSpace::MarkAsResult(AbstractTensorHandle* gradient) const {}

void TapeVSpace::DeleteGradient(AbstractTensorHandle* gradient) const {
  gradient->Unref();
}

void Tape::Watch(const AbstractTensorHandle* t) {
  GradientTape::Watch(ToId(t));
}
void Tape::RecordOperation(absl::Span<AbstractTensorHandle* const> inputs,
                           absl::Span<AbstractTensorHandle* const> outputs,
                           GradientFunction* gradient_function,
                           const string& op_name) {
  std::vector<int64> input_ids(inputs.size());
  std::vector<tensorflow::DataType> input_dtypes(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    input_ids[i] = ToId(inputs[i]);
    input_dtypes[i] = inputs[i]->DataType();
  }
  std::vector<TapeTensor> tape_tensors;
  for (auto t : outputs) {
    tape_tensors.push_back(TapeTensor(t));
  }
  GradientTape::RecordOperation(
      op_name, tape_tensors, input_ids, input_dtypes,
      [gradient_function]() -> GradientFunction* { return gradient_function; },
      [](GradientFunction* ptr) {
        if (ptr) {
          delete ptr;
        }
      });
}
bool Tape::ShouldRecord(
    absl::Span<const AbstractTensorHandle* const> tensors) const {
  std::vector<int64> tensor_ids(tensors.size());
  std::vector<tensorflow::DataType> tensor_dtypes(tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    tensor_ids[i] = ToId(tensors[i]);
    tensor_dtypes[i] = tensors[i]->DataType();
  }
  return GradientTape::ShouldRecord(tensor_ids, tensor_dtypes);
}
void Tape::DeleteTrace(const AbstractTensorHandle* t) {
  GradientTape::DeleteTrace(ToId(t));
}

std::vector<int64> MakeTensorIDList(
    absl::Span<AbstractTensorHandle* const> tensors) {
  std::vector<int64> ids(tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    ids[i] = ToId(tensors[i]);
  }
  return ids;
}

Status Tape::ComputeGradient(
    AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> targets,
    absl::Span<AbstractTensorHandle* const> sources,
    absl::Span<AbstractTensorHandle* const> output_gradients,
    absl::Span<AbstractTensorHandle*> result) {
  TapeVSpace vspace(ctx);
  std::vector<int64> target_tensor_ids = MakeTensorIDList(targets);
  std::vector<int64> source_tensor_ids = MakeTensorIDList(sources);
  tensorflow::gtl::FlatSet<tensorflow::int64> sources_set(
      source_tensor_ids.begin(), source_tensor_ids.end());
  std::unordered_map<int64, TapeTensor> sources_that_are_targets;
  for (int i = 0; i < target_tensor_ids.size(); ++i) {
    int64 target_id = target_tensor_ids[i];
    if (sources_set.find(target_id) != sources_set.end()) {
      auto tensor = targets[i];
      sources_that_are_targets.insert(
          std::make_pair(target_id, TapeTensor(tensor)));
    }
  }

  TF_RETURN_IF_ERROR(GradientTape::ComputeGradient(
      vspace, target_tensor_ids, source_tensor_ids, sources_that_are_targets,
      output_gradients, result, /*build_default_zeros_grads*/ false));
  return Status::OK();
}

// Helper functions which delegate to `AbstractOperation`, update
// the state of the ForwardOperation and call the tape as appropriate.
// These APIs are mainly to facilitate testing and are subject to change.
namespace internal {
Status Reset(AbstractOperation* op_, const char* op,
             const char* raw_device_name, ForwardOperation* forward_op_) {
  forward_op_->op_name = op;
  forward_op_->attrs.Reset(op);
  return op_->Reset(op, raw_device_name);
}
Status AddInput(AbstractOperation* op_, AbstractTensorHandle* input,
                ForwardOperation* forward_op_) {
  TF_RETURN_IF_ERROR(op_->AddInput(input));
  forward_op_->inputs.push_back(input);
  return Status::OK();
}
Status AddInputList(AbstractOperation* op_,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    ForwardOperation* forward_op_) {
  TF_RETURN_IF_ERROR(op_->AddInputList(inputs));
  for (auto input : inputs) {
    forward_op_->inputs.push_back(input);
  }
  return Status::OK();
}

Status SetAttrString(AbstractOperation* op_, const char* attr_name,
                     const char* data, size_t length,
                     ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name, StringPiece(data, length));
  return op_->SetAttrString(attr_name, data, length);
}
Status SetAttrInt(AbstractOperation* op_, const char* attr_name, int64_t value,
                  ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name, static_cast<int64>(value));
  return op_->SetAttrInt(attr_name, value);
}
Status SetAttrFloat(AbstractOperation* op_, const char* attr_name, float value,
                    ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrFloat(attr_name, value);
}
Status SetAttrBool(AbstractOperation* op_, const char* attr_name, bool value,
                   ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrBool(attr_name, value);
}
Status SetAttrType(AbstractOperation* op_, const char* attr_name,
                   DataType value, ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name, value);
  return op_->SetAttrType(attr_name, value);
}
Status SetAttrShape(AbstractOperation* op_, const char* attr_name,
                    const int64_t* dims, const int num_dims,
                    ForwardOperation* forward_op_) {
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

  forward_op_->attrs.Set(attr_name, proto);
  return op_->SetAttrShape(attr_name, dims, num_dims);
}
Status SetAttrFunction(AbstractOperation* op_, const char* attr_name,
                       const AbstractOperation* value,
                       ForwardOperation* forward_op_) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunction has not been implemented yet.");
}
Status SetAttrFunctionName(AbstractOperation* op_, const char* attr_name,
                           const char* value, size_t length,
                           ForwardOperation* forward_op_) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionName has not been implemented "
      "yet.");
}
Status SetAttrTensor(AbstractOperation* op_, const char* attr_name,
                     AbstractTensorInterface* tensor,
                     ForwardOperation* forward_op_) {
  return tensorflow::errors::Unimplemented(
      "SetAttrTensor has not been implemented yet.");
}
Status SetAttrStringList(AbstractOperation* op_, const char* attr_name,
                         const void* const* values, const size_t* lengths,
                         int num_values, ForwardOperation* forward_op_) {
  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  forward_op_->attrs.Set(attr_name, v);
  return op_->SetAttrStringList(attr_name, values, lengths, num_values);
}
Status SetAttrFloatList(AbstractOperation* op_, const char* attr_name,
                        const float* values, int num_values,
                        ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const float>(values, num_values));
  return op_->SetAttrFloatList(attr_name, values, num_values);
}
Status SetAttrIntList(AbstractOperation* op_, const char* attr_name,
                      const int64_t* values, int num_values,
                      ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(
      attr_name, gtl::ArraySlice<const int64>(
                     reinterpret_cast<const int64*>(values), num_values));
  return op_->SetAttrIntList(attr_name, values, num_values);
}
Status SetAttrTypeList(AbstractOperation* op_, const char* attr_name,
                       const DataType* values, int num_values,
                       ForwardOperation* forward_op_) {
  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const DataType>(values, num_values));
  return op_->SetAttrTypeList(attr_name, values, num_values);
}
Status SetAttrBoolList(AbstractOperation* op_, const char* attr_name,
                       const unsigned char* values, int num_values,
                       ForwardOperation* forward_op_) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  forward_op_->attrs.Set(attr_name,
                         gtl::ArraySlice<const bool>(b.get(), num_values));
  return op_->SetAttrBoolList(attr_name, values, num_values);
}
Status SetAttrShapeList(AbstractOperation* op_, const char* attr_name,
                        const int64_t** dims, const int* num_dims,
                        int num_values, ForwardOperation* forward_op_) {
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
  forward_op_->attrs.Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return op_->SetAttrShapeList(attr_name, dims, num_dims, num_values);
}
Status SetAttrFunctionList(AbstractOperation* op_, const char* attr_name,
                           absl::Span<const AbstractOperation*> values,
                           ForwardOperation* forward_op_) {
  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionList has not been "
      "implemented yet.");
}
Status Execute(AbstractOperation* op_, AbstractContext* ctx,
               absl::Span<AbstractTensorHandle*> retvals, int* num_retvals,
               ForwardOperation* forward_op_, Tape* tape,
               const GradientRegistry& registry) {
  TF_RETURN_IF_ERROR(op_->Execute(retvals, num_retvals));
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(srbs): Manage refcount of ForwardOperation's inputs/outputs.
    forward_op_->outputs.push_back(retvals[i]);
  }
  // TODO(b/166669239): This is needed to support AttrBuilder::Get for string
  // attributes. Number type attrs and DataType attrs work fine without this.
  // Consider getting rid of this and making the behavior between number types
  // and string consistent.
  forward_op_->attrs.BuildNodeDef();
  std::unique_ptr<GradientFunction> gradient_fn;
  TF_RETURN_IF_ERROR(registry.Lookup(*forward_op_, &gradient_fn));
  tape->RecordOperation(forward_op_->inputs, retvals, gradient_fn.release(),
                        op_->Name());
  return Status::OK();
}
}  // namespace internal

}  // namespace gradients
}  // namespace tensorflow
