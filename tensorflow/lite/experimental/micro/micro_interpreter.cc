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
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/experimental/micro/compatibility.h"

namespace tflite {
namespace {
const int kStackDataAllocatorSize = 128;
class StackDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size) override {
    if (size > kStackDataAllocatorSize) {
      return nullptr;
    } else {
      return data_;
    }
  }
  void Deallocate(void* data) override {
    // Do nothing.
  }

 private:
  uint8_t data_[kStackDataAllocatorSize];

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

const char* OpNameFromRegistration(const TfLiteRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

void ReportOpError(struct TfLiteContext* context, const char* format, ...) {
  MicroInterpreter* interpreter =
      static_cast<MicroInterpreter*>(context->impl_);
  va_list args;
  va_start(args, format);
  interpreter->error_reporter()->Report(format, args);
  va_end(args);
}

}  // namespace

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const OpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      context_(),
      allocator_(&context_, model_, tensor_arena, tensor_arena_size,
                 error_reporter_),
      tensors_allocated_(false) {
  auto* subgraphs = model->subgraphs();
  if (subgraphs->size() != 1) {
    error_reporter->Report("Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];
  tensors_ = subgraph_->tensors();
  operators_ = subgraph_->operators();

  context_.impl_ = static_cast<void*>(this);
  context_.ReportError = ReportOpError;
  context_.recommended_num_threads = 1;

  // If the system is big endian then convert weights from the flatbuffer from
  // little to big endian on startup so that it does not need to be done during
  // inference.
  // NOTE: This requires that the flatbuffer is held in memory which can be
  // modified by this process.
  if (!FLATBUFFERS_LITTLEENDIAN) {
    for (int t = 0; t < tensors_size(); ++t) {
      TfLiteTensor* thisTensor = &context_.tensors[t];
      if (thisTensor->allocation_type == kTfLiteMmapRo)
        CorrectTensorEndianness(thisTensor);
    }
  }

  initialization_status_ = kTfLiteOk;
}

void MicroInterpreter::CorrectTensorEndianness(TfLiteTensor* tensorCorr) {
  int32_t tensorSize = 1;
  for (int d = 0; d < tensorCorr->dims->size; ++d)
    tensorSize *= reinterpret_cast<const int32_t*>(tensorCorr->dims->data)[d];

  switch (tensorCorr->type) {
    case TfLiteType::kTfLiteFloat32:
      CorrectTensorDataEndianness(tensorCorr->data.f, tensorSize);
      break;
    case TfLiteType::kTfLiteFloat16:
      CorrectTensorDataEndianness(tensorCorr->data.f16, tensorSize);
      break;
    case TfLiteType::kTfLiteInt64:
      CorrectTensorDataEndianness(tensorCorr->data.i64, tensorSize);
      break;
    case TfLiteType::kTfLiteInt32:
      CorrectTensorDataEndianness(tensorCorr->data.i32, tensorSize);
      break;
    case TfLiteType::kTfLiteInt16:
      CorrectTensorDataEndianness(tensorCorr->data.i16, tensorSize);
      break;
    case TfLiteType::kTfLiteComplex64:
      CorrectTensorDataEndianness(tensorCorr->data.c64, tensorSize);
      break;
    default:
      // Do nothing for other data types.
      break;
  }
}

template <class T>
void MicroInterpreter::CorrectTensorDataEndianness(T* data, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    data[i] = flatbuffers::EndianScalar(data[i]);
  }
}

TfLiteStatus MicroInterpreter::RegisterPreallocatedInput(uint8_t* buffer,
                                                         size_t input_index) {
  return allocator_.RegisterPreallocatedInput(buffer, input_index);
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  TfLiteStatus status = allocator_.AllocateTensors();
  TF_LITE_ENSURE_OK(&context_, status);
  tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    error_reporter_->Report("Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    AllocateTensors();
  }
  TfLiteStatus status = kTfLiteOk;
  auto opcodes = model_->operator_codes();
  for (size_t i = 0; i < operators_->size(); ++i) {
    const auto* op = operators_->Get(i);
    size_t index = op->opcode_index();
    if (index < 0 || index >= opcodes->size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      return kTfLiteError;
    }
    auto opcode = (*opcodes)[index];
    const TfLiteRegistration* registration = nullptr;
    status = GetRegistrationFromOpCode(opcode, op_resolver_, error_reporter_,
                                       &registration);
    if (status != kTfLiteOk) {
      return status;
    }
    if (registration == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      return kTfLiteError;
    }
    BuiltinOperator op_type =
        static_cast<BuiltinOperator>(registration->builtin_code);

    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Unsupported behavior: found builtin operator %s with custom "
          "options.\n",
          EnumNameBuiltinOperator(op_type));
      return kTfLiteError;
    }
    StackDataAllocator stack_data_allocator;
    const char* custom_data = nullptr;
    size_t custom_data_size = 0;
    unsigned char* builtin_data = nullptr;
    if (op->custom_options()) {
      custom_data = reinterpret_cast<const char*>(op->custom_options()->data());
      custom_data_size = op->custom_options()->size();
    } else {
      TF_LITE_ENSURE_STATUS(ParseOpData(op, op_type, error_reporter_,
                                        &stack_data_allocator,
                                        (void**)(&builtin_data)));
    }

    const char* init_data;
    size_t init_data_size;
    if (registration->builtin_code == BuiltinOperator_CUSTOM) {
      init_data = custom_data;
      init_data_size = custom_data_size;
    } else {
      init_data = reinterpret_cast<const char*>(builtin_data);
      init_data_size = 0;
    }
    void* user_data = nullptr;
    if (registration->init) {
      user_data = registration->init(&context_, init_data, init_data_size);
    }

    // Disregard const qualifier to workaround with existing API.
    TfLiteIntArray* inputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->inputs()));
    TfLiteIntArray* outputs_array = const_cast<TfLiteIntArray*>(
        reinterpret_cast<const TfLiteIntArray*>(op->outputs()));

    const int kMaxTemporaries = 16;
    int temporaries_data[kMaxTemporaries + 1];
    TfLiteIntArray* temporaries_array =
        reinterpret_cast<TfLiteIntArray*>(temporaries_data);
    temporaries_array->size = 0;

    TfLiteNode node;
    node.inputs = inputs_array;
    node.outputs = outputs_array;
    node.temporaries = temporaries_array;
    node.user_data = user_data;
    node.builtin_data = reinterpret_cast<void*>(builtin_data);
    node.custom_initial_data = custom_data;
    node.custom_initial_data_size = custom_data_size;
    node.delegate = nullptr;
    if (registration->prepare) {
      TfLiteStatus prepare_status = registration->prepare(&context_, &node);
      if (prepare_status != kTfLiteOk) {
        error_reporter_->Report(
            "Node %s (number %d) failed to prepare with status %d",
            OpNameFromRegistration(registration), i, prepare_status);
        return kTfLiteError;
      }
    }

    if (registration->invoke) {
      TfLiteStatus invoke_status = registration->invoke(&context_, &node);
      if (invoke_status != kTfLiteOk) {
        error_reporter_->Report(
            "Node %s (number %d) failed to invoke with status %d",
            OpNameFromRegistration(registration), i, invoke_status);
        return kTfLiteError;
      }
    }

    if (registration->free) {
      registration->free(&context_, user_data);
    }
  }
  return status;
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const flatbuffers::Vector<int32_t>* inputs = subgraph_->inputs();
  const size_t length = inputs->size();
  if ((index < 0) || (index >= length)) {
    error_reporter_->Report("Input index %d out of range (length is %d)", index,
                            length);
    return nullptr;
  }
  return &(context_.tensors[inputs->Get(index)]);
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const flatbuffers::Vector<int32_t>* outputs = subgraph_->outputs();
  const size_t length = outputs->size();
  if ((index < 0) || (index >= outputs->size())) {
    error_reporter_->Report("Output index %d out of range (length is %d)",
                            index, length);
    return nullptr;
  }
  return &(context_.tensors[outputs->Get(index)]);
}

TfLiteTensor* MicroInterpreter::tensor(size_t index) {
  const size_t length = tensors_size();
  if ((index < 0) || (index >= tensors_size())) {
    error_reporter_->Report("Tensor index %d out of range (length is %d)",
                            index, length);
    return nullptr;
  }
  return &context_.tensors[index];
}

}  // namespace tflite
