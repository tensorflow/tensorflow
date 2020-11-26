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

namespace tensorflow {
namespace gradients {

namespace {
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

class IncomingGradientsImpl : public IncomingGradients {
 public:
  explicit IncomingGradientsImpl(
      absl::Span<AbstractTensorHandle* const> grad_inputs, Context* ctx,
      DefaultGradientFunction* default_gradients)
      : grad_inputs_(grad_inputs),
        ctx_(ctx),
        default_gradients_(default_gradients) {}
  AbstractTensorHandle* operator[](int i) const override {
    return default_gradients_->get(ctx_, grad_inputs_, i);
  }
  size_t size() const override { return grad_inputs_.size(); }

 private:
  absl::Span<AbstractTensorHandle* const> grad_inputs_;
  Context* ctx_;
  DefaultGradientFunction* default_gradients_;
};

AllZerosDefaultGradients::AllZerosDefaultGradients(const ForwardOperation& op)
    : outputs_(op.outputs) {
  for (auto output : outputs_) {
    output->Ref();
  }
}
AbstractTensorHandle* AllZerosDefaultGradients::get(
    Context* ctx, absl::Span<AbstractTensorHandle* const> grad_inputs, int i) {
  if (grad_inputs[i]) {
    return grad_inputs[i];
  }
  if (cached_default_grads_[i]) {
    return cached_default_grads_[i].get();
  }
  AbstractTensorHandle* result = nullptr;
  Status s = ZerosLike(ctx->ctx, outputs_[i], &result);
  if (!s.ok()) {
    if (result) {
      result->Unref();
    }
    VLOG(1) << "Failed to create ZerosLike for index " << i;
    return nullptr;
  }
  cached_default_grads_[i].reset(result);
  return result;
}

PassThroughDefaultGradients::PassThroughDefaultGradients(
    const ForwardOperation& op) {}
AbstractTensorHandle* PassThroughDefaultGradients::get(
    Context* ctx, absl::Span<AbstractTensorHandle* const> grad_inputs, int i) {
  return grad_inputs[i];
}

Status GradientRegistry::Register(
    const string& op_name, BackwardFunctionFactory backward_function_factory) {
  auto iter = registry_.find(op_name);
  if (iter != registry_.end()) {
    const string error_msg = "Gradient already exists for op: " + op_name + ".";
    return errors::AlreadyExists(error_msg);
  }
  registry_.insert({op_name, backward_function_factory});
  return Status::OK();
}
Status GradientRegistry::Lookup(
    const ForwardOperation& op,
    std::unique_ptr<BackwardFunction>* backward_function) const {
  auto iter = registry_.find(op.op_name);
  if (iter == registry_.end()) {
    const string error_msg = "No gradient defined for op: " + op.op_name + ".";
    return errors::NotFound(error_msg);
  }
  backward_function->reset(iter->second(op));
  return Status::OK();
}

int64 ToId(AbstractTensorHandle* t) {
  return static_cast<int64>(reinterpret_cast<uintptr_t>(t));
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
Status TapeVSpace::CallBackwardFunction(
    BackwardFunction* backward_function,
    const std::vector<int64>& unneeded_gradients,
    gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
    std::vector<AbstractTensorHandle*>* result) const {
  if (backward_function == nullptr) return Status::OK();
  Context ctx = {ctx_};
  IncomingGradientsImpl incoming_gradients(
      output_gradients, &ctx, backward_function->GetDefaultGradientFunction());
  return backward_function->GetGradientFunction()->Compute(
      &ctx, incoming_gradients, result);
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
  std::vector<int64> input_ids(forward_op_->inputs.size());
  std::vector<tensorflow::DataType> input_dtypes(forward_op_->inputs.size());
  for (int i = 0; i < forward_op_->inputs.size(); i++) {
    input_ids[i] = ToId(forward_op_->inputs[i]);
    input_dtypes[i] = forward_op_->inputs[i]->DataType();
  }
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(srbs): Manage refcount of ForwardOperation's inputs/outputs.
    forward_op_->outputs.push_back(retvals[i]);
  }
  // TODO(b/166669239): This is needed to support AttrBuilder::Get for string
  // attributes. Number type attrs and DataType attrs work fine without this.
  // Consider getting rid of this and making the behavior between number types
  // and string consistent.
  forward_op_->attrs.BuildNodeDef();
  std::vector<TapeTensor> tape_tensors;
  for (auto t : retvals) {
    tape_tensors.push_back(TapeTensor(t));
  }
  tape->RecordOperation(
      op_->Name(), tape_tensors, input_ids, input_dtypes,
      [registry, forward_op_]() -> BackwardFunction* {
        std::unique_ptr<BackwardFunction> backward_fn;
        Status s = registry.Lookup(*forward_op_, &backward_fn);
        if (!s.ok()) {
          return nullptr;
        }
        return backward_fn.release();
      },
      [](BackwardFunction* ptr) {
        if (ptr) {
          delete ptr;
        }
      });
  return Status::OK();
}
}  // namespace internal

}  // namespace gradients
}  // namespace tensorflow
