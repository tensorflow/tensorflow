// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_extended_interpreter.h"

#include <iostream>
#include <vector>

namespace tflite {
namespace micro {
namespace xcore {

constexpr int max_log_len = 256;

typedef TfLiteStatus (*invoke_function_t)(TfLiteContext*, TfLiteNode*);

//****************************
//****************************
//****************************
// Callback classes
//****************************
//****************************
//****************************
class CallbackContext {
 public:
  CallbackContext()
      : current_operator(0),
        preinvoke_callback(nullptr),
        postinvoke_callback(nullptr) {}
  void Reset() {
    current_operator = 0;
    preinvoke_callback = nullptr;
    postinvoke_callback = nullptr;
    invoke_functions.clear();
  }
  int current_operator;
  invoke_callback_t preinvoke_callback;
  invoke_callback_t postinvoke_callback;
  std::vector<invoke_function_t> invoke_functions;
};
static CallbackContext gCallbackContext;

TfLiteStatus CallbackInvoke(TfLiteContext* context, TfLiteNode* node) {
  int current_operator = gCallbackContext.current_operator;

  invoke_function_t invoke =
      gCallbackContext.invoke_functions[current_operator];

  if (gCallbackContext.preinvoke_callback)
    gCallbackContext.preinvoke_callback(current_operator);
  TfLiteStatus status = invoke(context, node);
  if (gCallbackContext.postinvoke_callback)
    gCallbackContext.postinvoke_callback(current_operator);
  gCallbackContext.current_operator++;

  return status;
}

//****************************
//****************************
//****************************
// BufferedErrorReporter
//****************************
//****************************
//****************************
int BufferedErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

int BufferedErrorReporter::Report(const char* format, va_list args) {
  char log_buffer[max_log_len];
  std::vsnprintf(log_buffer, max_log_len, format, args);
  log_stream_ << log_buffer << std::endl;
  return 0;
}

std::string BufferedErrorReporter::GetError() {
  std::string error = log_stream_.str();
  Clear();
  return error;
}

void BufferedErrorReporter::Clear() { log_stream_.str(""); }

//****************************
//****************************
//****************************
// ExtendedXCoreInterpreter
//****************************
//****************************
//****************************
ExtendedXCoreInterpreter::ExtendedXCoreInterpreter(
    const tflite::Model* model, const tflite::MicroOpResolver& resolver,
    uint8_t* arena, size_t arena_size, tflite::ErrorReporter* reporter,
    bool use_current_thread, tflite::Profiler* profiler)
    : XCoreInterpreter(model, resolver, arena, arena_size, reporter,
                       use_current_thread, profiler),
      reporter_(reporter) {}

ExtendedXCoreInterpreter::ExtendedXCoreInterpreter(
    const tflite::Model* model, const tflite::MicroOpResolver& resolver,
    tflite::MicroAllocator* allocator, tflite::ErrorReporter* reporter,
    bool use_current_thread, tflite::Profiler* profiler)
    : XCoreInterpreter(model, resolver, allocator, reporter, use_current_thread,
                       profiler),
      reporter_(reporter) {}

size_t ExtendedXCoreInterpreter::input_tensor_index(size_t input_index) {
  const TfLiteTensor* inputtensor_p = input(input_index);
  for (size_t i = 0; i < tensors_size(); i++) {
    const TfLiteTensor* tensor_p = tensor(i);
    if (tensor_p == inputtensor_p) return i;
  }
  return -1;
}

size_t ExtendedXCoreInterpreter::output_tensor_index(size_t output_index) {
  const TfLiteTensor* outputtensor_p = output(output_index);
  for (size_t i = 0; i < tensors_size(); i++) {
    const TfLiteTensor* tensor_p = tensor(i);
    if (tensor_p == outputtensor_p) return i;
  }
  return -1;
}

TfLiteStatus ExtendedXCoreInterpreter::Invoke(
    invoke_callback_t preinvoke_callback,
    invoke_callback_t postinvoke_callback) {
  if (preinvoke_callback || postinvoke_callback) {
    gCallbackContext.preinvoke_callback = preinvoke_callback;
    gCallbackContext.postinvoke_callback = postinvoke_callback;
    // Save the registered invoke functions
    for (size_t node_index = 0; node_index < operators_size(); node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          node_and_registration(static_cast<int>(node_index));

      gCallbackContext.invoke_functions.push_back(
          node_and_reg.registration->invoke);
    }
    // Set the invoke function to the CallbackInvoke
    for (size_t node_index = 0; node_index < operators_size(); node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          node_and_registration(static_cast<int>(node_index));
      const TfLiteRegistration* reg = node_and_reg.registration;

      // Disregard const qualifier to workaround existing API.
      (const_cast<TfLiteRegistration*>(reg))->invoke = CallbackInvoke;
    }
  }

  TfLiteStatus invoke_status = XCoreInterpreter::Invoke();

  // Set back the original invoke function
  if (preinvoke_callback || postinvoke_callback) {
    // Set the invoke function to the CallbackInvoke
    for (size_t node_index = 0; node_index < operators_size(); node_index++) {
      tflite::NodeAndRegistration node_and_reg =
          node_and_registration(static_cast<int>(node_index));
      const TfLiteRegistration* reg = node_and_reg.registration;

      // Disregard const qualifier to workaround existing API.
      (const_cast<TfLiteRegistration*>(reg))->invoke =
          gCallbackContext.invoke_functions[node_index];
    }
    gCallbackContext.Reset();
  }
  if (invoke_status != kTfLiteOk) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::SetTensor(size_t tensor_index,
                                                 const void* value,
                                                 const int size,
                                                 const int* shape,
                                                 const int type) {
  TfLiteTensor* tensor_p = tensor(tensor_index);
  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->dims->size != size) {
    reporter_->Report("tensor dims size %d != %d", tensor_p->dims->size, size);
    return kTfLiteError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor_p->dims->data[i] != shape[i]) {
      reporter_->Report("tensor dim %d != %d", tensor_p->dims->data[i],
                        shape[i]);
      return kTfLiteError;
    }
  }

  if (tensor_p->type != type) {
    reporter_->Report("tensor type %d != %d", tensor_p->type, type);
    return kTfLiteError;
  }

  std::memcpy(tensor_p->data.raw, value, tensor_p->bytes);
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensor(size_t tensor_index,
                                                 void* value, const int size,
                                                 const int* shape,
                                                 const int type) {
  TfLiteTensor* tensor_p = tensor(tensor_index);
  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->dims->size != size) {
    reporter_->Report("tensor dims size %d != %d", tensor_p->dims->size, size);
    return kTfLiteError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor_p->dims->data[i] != shape[i]) {
      reporter_->Report("tensor dim %d != %d", tensor_p->dims->data[i],
                        shape[i]);
      return kTfLiteError;
    }
  }

  if (tensor_p->type != type) {
    reporter_->Report("tensor type %d != %d", tensor_p->type, type);
    return kTfLiteError;
  }

  std::memcpy(value, tensor_p->data.raw, tensor_p->bytes);
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensorDetailsBufferSizes(
    size_t tensor_index, size_t* dims, size_t* scales, size_t* zero_points) {
  TfLiteTensor* tensor_p = tensor(tensor_index);
  if (tensor_p == nullptr) {
    return kTfLiteError;
  }
  *dims = tensor_p->dims->size;
  TfLiteAffineQuantization* quantization_params =
      static_cast<TfLiteAffineQuantization*>(tensor_p->quantization.params);
  if (quantization_params) {
    *scales = quantization_params->scale->size;
    *zero_points = quantization_params->zero_point->size;
  } else {
    *scales = 1;
    *zero_points = 1;
  }
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensorDetails(
    size_t tensor_index, char* name, int name_len, int* shape, int* type,
    float* scale, int32_t* zero_point) {
  TfLiteTensor* tensor_p = tensor(tensor_index);
  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->name) std::strncpy(name, tensor_p->name, name_len);

  for (int i = 0; i < tensor_p->dims->size; i++) {
    shape[i] = tensor_p->dims->data[i];
  }
  *type = tensor_p->type;
  TfLiteAffineQuantization* quantization_params =
      static_cast<TfLiteAffineQuantization*>(tensor_p->quantization.params);
  if (quantization_params) {
    for (int i = 0; i < quantization_params->scale->size; i++) {
      scale[i] = quantization_params->scale->data[i];
    }
    for (int i = 0; i < quantization_params->zero_point->size; i++) {
      zero_point[i] = quantization_params->zero_point->data[i];
    }
  } else {
    *scale = 0.0;
    *zero_point = 0;
  }
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetOperatorDetailsBufferSizes(
    size_t operator_index, size_t* inputs, size_t* outputs) {
  if (operator_index >= operators_size()) {
    reporter_->Report("Invalid operator index %d", operator_index);
    return kTfLiteError;
  }

  tflite::NodeAndRegistration node_and_reg =
      node_and_registration(static_cast<int>(operator_index));
  const TfLiteNode& node = node_and_reg.node;
  *inputs = node.inputs->size;
  *outputs = node.outputs->size;

  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetOperatorDetails(
    size_t operator_index, char* name, int name_len, int* version, int* inputs,
    int* outputs) {
  if (operator_index >= operators_size()) {
    reporter_->Report("Invalid operator index %d", operator_index);
    return kTfLiteError;
  }

  tflite::NodeAndRegistration node_and_reg =
      node_and_registration(static_cast<int>(operator_index));
  const TfLiteNode& node = node_and_reg.node;
  const TfLiteRegistration* reg = node_and_reg.registration;

  if (reg->custom_name != nullptr) {
    std::strncpy(name, reg->custom_name, name_len);
  } else {
    std::strncpy(name, tflite::EnumNamesBuiltinOperator()[reg->builtin_code],
                 name_len);
  }
  *version = reg->version;
  for (int i = 0; i < node.inputs->size; i++) {
    inputs[i] = node.inputs->data[i];
  }
  for (int i = 0; i < node.outputs->size; i++) {
    outputs[i] = node.outputs->data[i];
  }

  return kTfLiteOk;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite