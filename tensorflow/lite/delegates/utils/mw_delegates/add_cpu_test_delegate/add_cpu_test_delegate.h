#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_CPU_TEST_DELEGATE_ADD_CPU_TEST_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_CPU_TEST_DELEGATE_ADD_CPU_TEST_DELEGATE_H_

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
} AddCpuTestDelegateOptions;

// Returns a structure with the default delegate options.
AddCpuTestDelegateOptions TfLiteAddCpuTestDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteAddCpuTestDelegateDelete` when delegate is no longer used by TFLite.
TfLiteDelegate* TfLiteAddCpuTestDelegateCreate(const AddCpuTestDelegateOptions* options);

// Destroys a delegate created with `TfLiteAddCpuTestDelegateCreate` call.
void TfLiteAddCpuTestDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteAddCpuTestDelegateCreateUnique(const AddCpuTestDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteAddCpuTestDelegateCreate(options), TfLiteAddCpuTestDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_MW_DELEGATES_ADD_CPU_TEST_DELEGATE_ADD_CPU_TEST_DELEGATE_H_