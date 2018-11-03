/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/model.h"
#ifndef TFLITE_MCU
#include "tensorflow/lite/nnapi_delegate.h"
#endif
#include "tensorflow/lite/version.h"

namespace tflite {

namespace {
// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e ? e : DefaultErrorReporter();
}
}  // namespace

const char* kEmptyTensorName = "";

// Normally we'd use ABSL_HAVE_ATTRIBUTE_WEAK and ABSL_ATTRIBUTE_WEAK, but
// we avoid the absl dependency for binary size reasons.
#ifdef __has_attribute
#define TFLITE_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define TFLITE_HAS_ATTRIBUTE(x) 0
#endif

#if TFLITE_HAS_ATTRIBUTE(weak) || (defined(__GNUC__) && !defined(__clang__))
// Using weak symbols for the flex delegate allows automatic injection of the
// delegate simply by adding it as a dependency. See also the strong override in
// lite/delegates/flex/delegate.cc.
__attribute__((weak)) Interpreter::TfLiteDelegatePtr AcquireFlexDelegate() {
  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}
#else
Interpreter::TfLiteDelegatePtr (*AcquireFlexDelegate)() = nullptr;
#endif

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(const char* filename,
                                                  bool mmap_file,
                                                  ErrorReporter* error_reporter,
                                                  bool use_nnapi) {
  std::unique_ptr<Allocation> allocation;
  if (mmap_file && MMAPAllocation::IsSupported()) {
    if (use_nnapi && NNAPIDelegate::IsSupported())
      allocation.reset(new NNAPIAllocation(filename, error_reporter));
    else
      allocation.reset(new MMAPAllocation(filename, error_reporter));
  } else {
    allocation.reset(new FileCopyAllocation(filename, error_reporter));
  }
  return allocation;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model;
  auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
                                          error_reporter, /*use_nnapi=*/true);
  model.reset(new FlatBufferModel(allocation.release(), error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(
    const char* filename, TfLiteVerifier* verifier,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model;
  auto allocation = GetAllocationFromFile(filename, /*mmap_file=*/true,
                                          error_reporter, /*use_nnapi=*/true);
  if (verifier &&
      !verifier->Verify(static_cast<const char*>(allocation->base()),
                        allocation->bytes(), error_reporter)) {
    return model;
  }
  model.reset(new FlatBufferModel(allocation.release(), error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}
#endif

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromBuffer(
    const char* buffer, size_t buffer_size, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model;
  Allocation* allocation =
      new MemoryAllocation(buffer, buffer_size, error_reporter);
  model.reset(new FlatBufferModel(allocation, error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromModel(
    const tflite::Model* model_spec, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model;
  model.reset(new FlatBufferModel(model_spec, error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}

bool FlatBufferModel::CheckModelIdentifier() const {
  if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
    const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
    error_reporter_->Report(
        "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
        ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
    return false;
  }
  return true;
}

FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)) {
  model_ = model;
}

FlatBufferModel::FlatBufferModel(Allocation* allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)) {
  allocation_ = allocation;
  if (!allocation_->valid() || !CheckModelIdentifier()) return;

  model_ = ::tflite::GetModel(allocation_->base());
}

FlatBufferModel::~FlatBufferModel() { delete allocation_; }

InterpreterBuilder::InterpreterBuilder(const FlatBufferModel& model,
                                       const OpResolver& op_resolver)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(model.error_reporter())),
      allocation_(model.allocation()) {}

InterpreterBuilder::InterpreterBuilder(const ::tflite::Model* model,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(ValidateErrorReporter(error_reporter)) {}

InterpreterBuilder::~InterpreterBuilder() {}

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping() {
  TfLiteStatus status = kTfLiteOk;
  auto opcodes = model_->operator_codes();
  for (const OperatorCode* opcode : *opcodes) {
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      return status;
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}

namespace {
template <class T>
std::vector<int> FlatBufferIntArrayToVector(T* flat_array) {
  // Initialize shape of tensors with null shape. Empty vectors are converted
  // to nullptr for models that are constructed via flatbuffers::Pack.
  if (flat_array == nullptr) {
    return {};
  }
  std::vector<int> ret(flat_array->Length());
  for (int i = 0; i < flat_array->Length(); i++) {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

// Used to determine how the op data parsing function creates its working space.
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size) override { return malloc(size); }
  void Deallocate(void* data) override { free(data); }
};

}  // namespace

TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Interpreter* interpreter) {
  TfLiteStatus status = kTfLiteOk;

  // Reduce the number of redundant allocations
  interpreter->ReserveNodes(operators->Length());

  for (int i = 0; i < operators->Length(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      status = kTfLiteError;
      continue;
    }

    const TfLiteRegistration* registration =
        flatbuffer_op_index_to_registration_[index];
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }

    if (op->custom_options()) {
      interpreter->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()),
          reinterpret_cast<const char*>(op->custom_options()->data()),
          op->custom_options()->size(), nullptr, registration);
    } else {
      void* builtin_data = nullptr;
      MallocDataAllocator malloc_allocator;
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &malloc_allocator, &builtin_data));
      interpreter->AddNodeWithParameters(
          FlatBufferIntArrayToVector(op->inputs()),
          FlatBufferIntArrayToVector(op->outputs()), nullptr, 0, builtin_data,
          registration);
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Interpreter* interpreter) {
  TfLiteStatus status = kTfLiteOk;

  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the interpreter.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  for (int i = 0; i < tensors->Length(); ++i) {
    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    TfLiteQuantizationParams quantization;
    quantization.scale = 0;
    quantization.zero_point = 0;
    auto* q_params = tensor->quantization();
    if (q_params) {
      // Note that the schema could hold per-channel quantization parameters
      // but we really only support one value for the whole tensor.
      // TODO(aselle): This breaks as well if these are nullptr's.
      // TODO(aselle): This assumes non per-channel quantization.

      if (q_params->scale()) {
        if (q_params->scale()->size() != 1) {
          error_reporter_->Report(
              "QuantizationParam has %d scale values (only 1 is supported).",
              q_params->scale()->size());
          return kTfLiteError;
        }
        quantization.scale = q_params->scale()->Get(0);
      }

      if (q_params->zero_point()) {
        if (q_params->zero_point()->size() != 1) {
          error_reporter_->Report(
              "QuantizationParam has %d zero_point values"
              " (only 1 is supported).",
              q_params->zero_point()->size());
          return kTfLiteError;
        }
        quantization.zero_point = q_params->zero_point()->Get(0);
      }
    }

    TfLiteType type;
    if (ConvertTensorType(tensor->type(), &type, error_reporter_) !=
        kTfLiteOk) {
      status = kTfLiteError;
      continue;
    }
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        error_reporter_->Report(
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        if (auto* array = buffer->data()) {
          if (size_t size = array->size()) {
            *buffer_size = size;
            *buffer_data = reinterpret_cast<const char*>(array->data());
            return kTfLiteOk;
          }
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    bool is_variable = tensor->is_variable();
    if (buffer_ptr) {
      if (is_variable) {
        error_reporter_->Report(
            "Tensor %d is a variable tensor with buffer. "
            "It's not supported now.\n",
            i);
        status = kTfLiteError;
      }

      if (interpreter->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    } else {
      if (interpreter->SetTensorParametersReadWrite(i, type, get_name(tensor),
                                                    dims, quantization,
                                                    is_variable) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ApplyDelegates(Interpreter* interpreter) {
  // TODO(b/117561550): Move flex delegate application to the OpResolver.
  if (AcquireFlexDelegate == nullptr) {
    return kTfLiteOk;
  }

  bool has_flex_op = false;
  for (const auto* registration : flatbuffer_op_index_to_registration_) {
    if ((registration->builtin_code == BuiltinOperator_CUSTOM) &&
        IsFlexOp(registration->custom_name)) {
      has_flex_op = true;
      break;
    }
  }

  if (!has_flex_op) {
    return kTfLiteOk;
  }

  if (auto flex_delegate = AcquireFlexDelegate()) {
    return interpreter->ModifyGraphWithDelegate(std::move(flex_delegate));
  }

  return kTfLiteOk;
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter) {
  return operator()(interpreter, /*num_threads=*/-1);
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter, int num_threads) {
  if (!interpreter) {
    error_reporter_->Report(
        "Null output pointer passed to InterpreterBuilder.");
    return kTfLiteError;
  }

  // Safe exit by deleting partially created interpreter, to reduce verbosity
  // on error conditions. Use by return cleanup_on_error();
  auto cleanup_and_error = [&interpreter]() {
    interpreter->reset();
    return kTfLiteError;
  };

  if (!model_) {
    error_reporter_->Report("Null pointer passed in as model.");
    return cleanup_and_error();
  }

  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter_->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return cleanup_and_error();
  }

  if (BuildLocalIndexToRegistrationMapping() != kTfLiteOk) {
    error_reporter_->Report("Registration failed.\n");
    return cleanup_and_error();
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();
  if (subgraphs->size() != 1) {
    error_reporter_->Report("Only 1 subgraph is currently supported.\n");
    return cleanup_and_error();
  }
  const tflite::SubGraph* subgraph = (*subgraphs)[0];
  auto operators = subgraph->operators();
  auto tensors = subgraph->tensors();
  if (!operators || !tensors || !buffers) {
    error_reporter_->Report(
        "Did not get operators, tensors, or buffers in input flat buffer.\n");
    return cleanup_and_error();
  }
  interpreter->reset(new Interpreter(error_reporter_));
  if ((**interpreter).AddTensors(tensors->Length()) != kTfLiteOk) {
    return cleanup_and_error();
  }
  // Set num threads
  (**interpreter).SetNumThreads(num_threads);
  // Parse inputs/outputs
  (**interpreter).SetInputs(FlatBufferIntArrayToVector(subgraph->inputs()));
  (**interpreter).SetOutputs(FlatBufferIntArrayToVector(subgraph->outputs()));

  // Finally setup nodes and tensors
  if (ParseNodes(operators, interpreter->get()) != kTfLiteOk)
    return cleanup_and_error();
  if (ParseTensors(buffers, tensors, interpreter->get()) != kTfLiteOk)
    return cleanup_and_error();

  std::vector<int> variables;
  for (int i = 0; i < (*interpreter)->tensors_size(); ++i) {
    auto* tensor = (*interpreter)->tensor(i);
    if (tensor->is_variable) {
      variables.push_back(i);
    }
  }
  (**interpreter).SetVariables(std::move(variables));

  if (ApplyDelegates(interpreter->get()) != kTfLiteOk)
    return cleanup_and_error();

  return kTfLiteOk;
}

}  // namespace tflite
