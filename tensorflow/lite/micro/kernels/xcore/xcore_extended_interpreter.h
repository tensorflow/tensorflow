// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_interpreter.h"

namespace tflite {
namespace micro {
namespace xcore {

typedef void (*invoke_callback_t)(int);

//****************************
//****************************
//****************************
// BufferedErrorReporter
//    A tflite::ErrorReporter that stores the last error so it can be fetched
//    later
//****************************
//****************************
//****************************
class BufferedErrorReporter : public tflite::ErrorReporter {
 public:
  ~BufferedErrorReporter() {}

  int Report(const char* format, ...);
  int Report(const char* format, va_list args);
  std::string GetError();
  void Clear();

 private:
  std::stringstream log_stream_;
};

//****************************
//****************************
//****************************
// ExtendedXCoreInterpreter
//****************************
//****************************
//****************************
class ExtendedXCoreInterpreter : public XCoreInterpreter {
 public:
  ExtendedXCoreInterpreter(const tflite::Model* model,
                           const tflite::MicroOpResolver& resolver,
                           uint8_t* arena, size_t arena_size,
                           tflite::ErrorReporter* reporter,
                           bool use_current_thread = true,
                           tflite::Profiler* profiler = nullptr);

  ExtendedXCoreInterpreter(const tflite::Model* model,
                           const tflite::MicroOpResolver& resolver,
                           tflite::MicroAllocator* allocator,
                           tflite::ErrorReporter* reporter,
                           bool use_current_thread = true,
                           tflite::Profiler* profiler = nullptr);

  size_t input_tensor_index(size_t input_index);
  size_t output_tensor_index(size_t output_index);

  TfLiteStatus Invoke(invoke_callback_t preinvoke_callback,
                      invoke_callback_t postinvoke_callback);

  TfLiteStatus SetTensor(size_t tensor_index, const void* value, const int size,
                         const int* shape, const int type);

  TfLiteStatus GetTensor(size_t tensor_index, void* value, const int size,
                         const int* shape, const int type);

  TfLiteStatus GetTensorDetailsBufferSizes(size_t tensor_index, size_t* dims,
                                           size_t* scales, size_t* zero_points);

  TfLiteStatus GetTensorDetails(size_t tensor_index, char* name, int name_len,
                                int* shape, int* type, float* scale,
                                int32_t* zero_point);

  TfLiteStatus GetOperatorDetailsBufferSizes(size_t operator_index,
                                             size_t* inputs, size_t* outputs);

  TfLiteStatus GetOperatorDetails(size_t operator_index, char* name,
                                  int name_len, int* version, int* inputs,
                                  int* outputs);

 private:
  const tflite::Model* model_;
  tflite::ErrorReporter* reporter_;
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
