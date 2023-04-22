#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_VMSIM_DELEGATE_VMSIM_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_VMSIM_DELEGATE_VMSIM_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/common.h"

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
} VMSimDelegateOptions;

// Returns a structure with the default delegate options.
VMSimDelegateOptions TfLiteVMSimDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteVMSimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteVMSimDelegateCreate(const VMSimDelegateOptions* options);

// Destroys a delegate created with `TfLiteVMSimDelegateCreate` call.
void TfLiteVMSimDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteVMSimDelegateCreateUnique(const VMSimDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteVMSimDelegateCreate(options), TfLiteVMSimDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_VMSIM_DELEGATE_VMSIM_DELEGATE_H_
