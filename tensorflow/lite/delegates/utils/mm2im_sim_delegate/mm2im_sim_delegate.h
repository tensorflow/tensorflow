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
} MM2IMSimDelegateOptions;

// Returns a structure with the default delegate options.
MM2IMSimDelegateOptions TfLiteMM2IMSimDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteMM2IMSimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteMM2IMSimDelegateCreate(const MM2IMSimDelegateOptions* options);

// Destroys a delegate created with `TfLiteMM2IMSimDelegateCreate` call.
void TfLiteMM2IMSimDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteMM2IMSimDelegateCreateUnique(const MM2IMSimDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteMM2IMSimDelegateCreate(options), TfLiteMM2IMSimDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_VMSIM_DELEGATE_VMSIM_DELEGATE_H_
