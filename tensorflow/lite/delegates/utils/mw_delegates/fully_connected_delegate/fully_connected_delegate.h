#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_FPGA_TEST_DELEGATE_ADD_FPGA_TEST_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_FPGA_TEST_DELEGATE_ADD_FPGA_TEST_DELEGATE_H_

#include <memory>
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} FullyConnectedDelegateOptions;

// Returns a structure with the default delegate options.
FullyConnectedDelegateOptions TfLiteFullyConnectedDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteFullyConnectedDelegateDelete` when delegate is no longer used by TFLite.
TfLiteDelegate* TfLiteFullyConnectedDelegateCreate(const FullyConnectedDelegateOptions* options);

// Destroys a delegate created with `TfLiteFullyConnectedDelegateCreate` call.
void TfLiteFullyConnectedDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteFullyConnectedDelegateCreateUnique(const FullyConnectedDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteFullyConnectedDelegateCreate(options), TfLiteFullyConnectedDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_FPGA_TEST_DELEGATE_ADD_FPGA_TEST_DELEGATE_H_