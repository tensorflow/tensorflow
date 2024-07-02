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

#include "tensorflow/cc/experimental/libtf/function.h"

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/cc/experimental/libtf/value_iostream.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {

namespace tf = tensorflow;

// Helper function to retrieve the function name from an AbstractFunctionPtr.
tf::StatusOr<std::string> GetFunctionName(AbstractFunctionPtr trace) {
  const tf::FunctionDef* fdef = nullptr;
  TF_RETURN_IF_ERROR(trace->GetFunctionDef(&fdef));
  return fdef->signature().name();
}

// Function to execute a traced function with the given context and inputs.
tf::Status ExecuteFunction(
    AbstractFunctionPtr trace, AbstractContext* ctx,
    absl::Span<tf::AbstractTensorHandle* const> inputs,
    absl::Span<tf::AbstractTensorHandle*> outputs) {
  TF_ASSIGN_OR_RETURN(std::string fname, GetFunctionName(trace));
  
  TF_RETURN_IF_ERROR(ctx->RegisterFunction(trace.get()));
  auto cleanup = absl::MakeCleanup([&]() { ctx->RemoveFunction(fname).IgnoreError(); });

  auto call_op = AbstractOperationPtr(ctx->CreateOperation());
  TF_RETURN_IF_ERROR(call_op->Reset(fname.c_str(), /*raw_device_name=*/nullptr));
  for (auto t : inputs) {
    TF_RETURN_IF_ERROR(call_op->AddInput(t));
  }

  int num_outputs = outputs.size();
  return call_op->Execute(outputs, &num_outputs);
}

// Verify if the given signature is supported.
tf::Status VerifySupportedSignature(const TaggedValue& signature) {
  if (signature.type() == TaggedValue::Type::TENSOR_SPEC) {
    return tf::OkStatus();
  }
  if (signature.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : signature.tuple()) {
      if (t.type() != TaggedValue::Type::TENSOR_SPEC) {
        break;
      }
    }
    return tf::OkStatus();
  }
  return tf::errors::Unimplemented("Only functions with inputs/outputs containing a single tensor or a tuple of tensors are supported right now.");
}

// Verify if the given arguments are supported.
tf::Status VerifySupportedArgs(const TaggedValue& args) {
  if (args.type() == TaggedValue::Type::TENSOR) {
    return tf::OkStatus();
  }
  if (args.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : args.tuple()) {
      if (t.type() != TaggedValue::Type::TENSOR) {
        break;
      }
    }
    return tf::OkStatus();
  }
  return tf::errors::Unimplemented("Only functions with inputs/outputs containing a single tensor or a tuple of tensors are supported right now.");
}

// Register a traced function with its input and output signatures.
tf::Status Function::RegisterTrace(AbstractFunctionPtr fn, TaggedValue input_signature, TaggedValue output_signature) {
  TF_RETURN_IF_ERROR(VerifySupportedSignature(input_signature));
  TF_RETURN_IF_ERROR(VerifySupportedSignature(output_signature));
  concrete_fns_.emplace_back(fn, input_signature, output_signature);
  return tf::OkStatus();
}

// Helper function to match a value against a signature.
bool Match(const TaggedValue& signature, const TaggedValue& value) {
  switch (signature.type()) {
    case TaggedValue::Type::TENSOR_SPEC: {
      if (value.type() != TaggedValue::Type::TENSOR) {
        return false;
      }
      const auto& spec = signature.tensor_spec();
      const auto& tensor = value.tensor();
      if (tensor->DataType() != spec.dtype) {
        return false;
      }
      tf::PartialTensorShape tensor_shape;
      DCHECK(tensor->Shape(&tensor_shape).ok());
      if (!tensor_shape.IsCompatibleWith(spec.shape)) {
        return false;
      }
      break;
    }
    case TaggedValue::Type::TUPLE: {
      if (value.type() != TaggedValue::Type::TUPLE || value.tuple().size() != signature.tuple().size()) {
        return false;
      }
      for (size_t i = 0; i < value.tuple().size(); ++i) {
        if (!Match(signature.tuple()[i], value.tuple()[i])) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  return true;
}

// Flatten the given value into a list of tensor handles.
void Flatten(const TaggedValue& value, std::vector<AbstractTensorHandle*>* flat_args) {
  if (value.type() == TaggedValue::Type::TENSOR) {
    flat_args->emplace_back(value.tensor().get());
  } else if (value.type() == TaggedValue::Type::TUPLE) {
    for (const auto& t : value.tuple()) {
      Flatten(t, flat_args);
    }
  } else {
    LOG(ERROR) << "Unimplemented";
  }
}

// Unflatten the given tensor handles into a structured TaggedValue.
tf::StatusOr<TaggedValue> Unflatten(absl::Span<AbstractTensorHandle* const> flat_args, const TaggedValue& structure) {
  if (structure.type() == TaggedValue::Type::TENSOR_SPEC) {
    if (flat_args.size() != 1) {
      return tf::errors::Internal("Expected single tensor but found ", flat_args.size());
    }
    TaggedValue wrapped_t = TaggedValue(impl::TaggedValueTensor(flat_args[0], /*add_ref=*/true));
    if (!Match(structure, wrapped_t)) {
      std::stringstream stream;
      stream << "Shape and dtype of tensor " << wrapped_t << " does not match that in signature " << structure;
      return tf::errors::Internal(stream.str());
    }
    return wrapped_t;
  }
  if (structure.type() == TaggedValue::Type::TUPLE) {
    if (flat_args.size() != structure.tuple().size()) {
      return tf::errors::InvalidArgument("Tuple length ", structure.tuple().size(), " does not match length of flat args ", flat_args.size());
    }
    auto result = impl::TaggedValue::Tuple();
    for (size_t i = 0; i < structure.tuple().size(); ++i) {
      TF_ASSIGN_OR_RETURN(TaggedValue ele, Unflatten({flat_args[i]}, structure.tuple()[i]));
      result.tuple().emplace_back(std::move(ele));
    }
    return result;
  }
  return tf::errors::Unimplemented("Only tensors and tuples of tensors are supported right now.");
}

// Get the flat size of a TaggedValue.
size_t GetFlatSize(const TaggedValue& value) {
  if (value.type() == TaggedValue::Type::TUPLE) {
    size_t result = 0;
    for (const auto& t : value.tuple()) {
      result += GetFlatSize(t);
    }
    return result;
  }
  if (value.type() == TaggedValue::Type::LIST) {
    size_t result = 0;
    for (const auto& t : value.list()) {
      result += GetFlatSize(t);
    }
    return result;
  }
  if (value.type() == TaggedValue::Type::DICT) {
    size_t result = 0;
    for (const auto& t : value.dict()) {
      result += GetFlatSize(t.second);
    }
    return result;
  }
  return 1;
}

// Execute a function with the given context and value.
tf::StatusOr<TaggedValue> Function::Execute(AbstractContext* ctx, const TaggedValue& value) const {
  TF_RETURN_IF_ERROR(VerifySupportedArgs(value));
  TF_ASSIGN_OR_RETURN(auto concrete_fn, GetConcreteFunction(value));
  
  std::vector<AbstractTensorHandle*> args;
  Flatten(value, &args);
  
  std::vector<AbstractTensorHandle*> outs(GetFlatSize(concrete_fn.output_signature));
  TF_RETURN_IF_ERROR(ExecuteFunction(concrete_fn.trace, ctx, args, absl::MakeSpan(outs)));

  auto cleanup_tensors = absl::MakeCleanup([&outs]() {
    for (auto t : outs) {
      t->Unref();
    }
  });

  return Unflatten(outs, concrete_fn.output_signature);
}

// Get the concrete function matching the given value.
tf::StatusOr<Function::ConcreteFunction> Function::GetConcreteFunction(const TaggedValue& value) const {
  if (concrete_fns_.empty()) {
    return tf::errors::FailedPrecondition("No registered ConcreteFunctions.");
  }
  for (const auto& spec : concrete_fns_) {
    if (Match(spec.input_signature, value)) {
      return spec;
    }
  }
  return tf::errors::InvalidArgument("No match found.");
}

}  // namespace libtf
}  // namespace tf
