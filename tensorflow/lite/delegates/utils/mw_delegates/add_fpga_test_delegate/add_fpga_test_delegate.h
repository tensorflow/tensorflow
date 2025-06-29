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
} AddFpgaTestDelegateOptions;

// Returns a structure with the default delegate options.
AddFpgaTestDelegateOptions TfLiteAddFpgaTestDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteAddFpgaTestDelegateDelete` when delegate is no longer used by TFLite.
TfLiteDelegate* TfLiteAddFpgaTestDelegateCreate(const AddFpgaTestDelegateOptions* options);

// Destroys a delegate created with `TfLiteAddFpgaTestDelegateCreate` call.
void TfLiteAddFpgaTestDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteAddFpgaTestDelegateCreateUnique(const AddFpgaTestDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteAddFpgaTestDelegateCreate(options), TfLiteAddFpgaTestDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_FPGA_TEST_DELEGATE_ADD_FPGA_TEST_DELEGATE_H_