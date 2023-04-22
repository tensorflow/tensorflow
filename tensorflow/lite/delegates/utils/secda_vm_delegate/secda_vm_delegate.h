#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_VM_DELEGATE_SECDA_VM_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_VM_DELEGATE_SECDA_VM_DELEGATE_H_

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
} SecdaVMDelegateOptions;

// Returns a structure with the default delegate options.
SecdaVMDelegateOptions TfLiteSecdaVMDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteSecdaVMDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteSecdaVMDelegateCreate(const SecdaVMDelegateOptions* options);

// Destroys a delegate created with `TfLiteSecdaVMDelegateCreate` call.
void TfLiteSecdaVMDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteSecdaVMDelegateCreateUnique(const SecdaVMDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteSecdaVMDelegateCreate(options), TfLiteSecdaVMDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_VM_DELEGATE_SECDA_VM_DELEGATE_H_
