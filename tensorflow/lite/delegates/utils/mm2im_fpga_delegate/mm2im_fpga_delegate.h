#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_MM2IMFPGA_DELEGATE_MM2IMFPGA_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_MM2IMFPGA_DELEGATE_MM2IMFPGA_DELEGATE_H_

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
} MM2IMFPGADelegateOptions;

// Returns a structure with the default delegate options.
MM2IMFPGADelegateOptions TfLiteMM2IMFPGADelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteMM2IMFPGADelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteMM2IMFPGADelegateCreate(const MM2IMFPGADelegateOptions* options);

// Destroys a delegate created with `TfLiteMM2IMFPGADelegateCreate` call.
void TfLiteMM2IMFPGADelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteMM2IMFPGADelegateCreateUnique(const MM2IMFPGADelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteMM2IMFPGADelegateCreate(options), TfLiteMM2IMFPGADelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_MM2IMFPGA_DELEGATE_MM2IMFPGA_DELEGATE_H_
