#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_BERT_DELEGATE_SECDA_BERT_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_BERT_DELEGATE_SECDA_BERT_DELEGATE_H_

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
} SecdaBertDelegateOptions;

// Returns a structure with the default delegate options.
SecdaBertDelegateOptions TfLiteSecdaBertDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteSecdaBertDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteSecdaBertDelegateCreate(const SecdaBertDelegateOptions* options);

// Destroys a delegate created with `TfLiteSecdaBertDelegateCreate` call.
void TfLiteSecdaBertDelegateDelete(TfLiteDelegate* delegate);
#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteSecdaBertDelegateCreateUnique(const SecdaBertDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteSecdaBertDelegateCreate(options), TfLiteSecdaBertDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SECDA_BERT_DELEGATE_SECDA_BERT_DELEGATE_H_
