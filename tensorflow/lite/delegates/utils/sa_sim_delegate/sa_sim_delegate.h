#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SA_SIM_DELEGATE_SA_SIM_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SA_SIM_DELEGATE_SA_SIM_DELEGATE_H_

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
} SASimDelegateOptions;

// Returns a structure with the default delegate options.
SASimDelegateOptions TfLiteSASimDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteSASimDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteSASimDelegateCreate(const SASimDelegateOptions* options);

// Destroys a delegate created with `TfLiteSASimDelegateCreate` call.
void TfLiteSASimDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteSASimDelegateCreateUnique(const SASimDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteSASimDelegateCreate(options), TfLiteSASimDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SA_SIM_DELEGATE_SA_SIM_DELEGATE_H_
