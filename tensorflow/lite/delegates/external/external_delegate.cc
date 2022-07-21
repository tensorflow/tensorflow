/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/delegates/external/external_delegate.h"

#include <locale>
#include <string>
#include <vector>

#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/shared_library.h"

namespace tflite {
namespace {

// External delegate library construct
struct ExternalLib {
  using CreateDelegatePtr = std::add_pointer<TfLiteDelegate*(
      const char**, const char**, size_t,
      void (*report_error)(const char*))>::type;
  using DestroyDelegatePtr = std::add_pointer<void(TfLiteDelegate*)>::type;
  struct wchar_codecvt : public std::codecvt<wchar_t, char, std::mbstate_t> {};
  std::wstring_convert<wchar_codecvt> converter;

  // Open a given delegate library and load the create/destroy symbols
  bool load(const std::string library) {
#if defined(_WIN32)
    void* handle = SharedLibrary::LoadLibrary(
        converter.from_bytes(library.c_str()).c_str());
#else
    void* handle = SharedLibrary::LoadLibrary(library.c_str());
#endif  // defined(_WIN32)
    if (handle == nullptr) {
      TFLITE_LOG(TFLITE_LOG_INFO, "Unable to load external delegate from : %s",
                 library.c_str());
    } else {
      create =
          reinterpret_cast<decltype(create)>(SharedLibrary::GetLibrarySymbol(
              handle, "tflite_plugin_create_delegate"));
      destroy =
          reinterpret_cast<decltype(destroy)>(SharedLibrary::GetLibrarySymbol(
              handle, "tflite_plugin_destroy_delegate"));
      return create && destroy;
    }
    return false;
  }

  CreateDelegatePtr create{nullptr};
  DestroyDelegatePtr destroy{nullptr};
};

// An ExternalDelegateWrapper is responsibile to manage a TFLite delegate
// initialized from a shared library. It creates a delegate from the given
// option and storages it to external_delegate_ member variable. On the
// destruction, it conducts necessary clean up process.
class ExternalDelegateWrapper {
 public:
  explicit ExternalDelegateWrapper(
      const TfLiteExternalDelegateOptions* options);
  ~ExternalDelegateWrapper();

  // Return a TfLiteDelegate which is created from
  // tflite_plugin_create_delegate() of an external delegate logic.
  TfLiteDelegate* tflite_external_delegate() { return external_delegate_; }

  // Return a TfLiteDelegate which is convertibile to this class.
  TfLiteDelegate* tflite_wrapper_delegate() { return &wrapper_delegate_; }

 private:
  ExternalLib external_lib_;

  // external delegate instance owned by external delegate logic.
  // It's created by "tflite_plugin_destroy_delegate()" function in the external
  // delegate logic And it should be released by
  // "tflite_plugin_destroy_delegate()" function.
  TfLiteDelegate* external_delegate_;

  // TfLiteDelegate representation of this ExternalDelegateWrapper object.
  TfLiteDelegate wrapper_delegate_;
};

// Converts the given TfLiteDelegate to an ExternalDelegateWrapper instance.
inline ExternalDelegateWrapper* GetExternalDelegateWrapper(
    TfLiteDelegate* delegate) {
  return reinterpret_cast<ExternalDelegateWrapper*>(delegate->data_);
}

// Relay Prepare() call to the associated external TfLiteDelegate object.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->Prepare(context, external_delegate);
}

// Relay CopyFromBufferHandle() call to the associated external TfLiteDelegate
// object.
TfLiteStatus DelegateCopyFromBufferHandle(TfLiteContext* context,
                                          struct TfLiteDelegate* delegate,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteTensor* tensor) {
  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->CopyFromBufferHandle(context, delegate,
                                                 buffer_handle, tensor);
}

// Relay CopyToBufferHandle() call to the associated external TfLiteDelegate
// object.
TfLiteStatus DelegateCopyToBufferHandle(TfLiteContext* context,
                                        struct TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) {
  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->CopyToBufferHandle(context, delegate, buffer_handle,
                                               tensor);
}

// Relay FreeBufferHandle() call to the associated external TfLiteDelegate
// object.
void DelegateFreeBufferHandle(TfLiteContext* context,
                              struct TfLiteDelegate* delegate,
                              TfLiteBufferHandle* handle) {
  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->FreeBufferHandle(context, delegate, handle);
}

ExternalDelegateWrapper::ExternalDelegateWrapper(
    const TfLiteExternalDelegateOptions* options) {
  external_delegate_ = nullptr;
  if (external_lib_.load(options->lib_path)) {
    std::vector<const char*> ckeys, cvalues;
    for (int i = 0; i < options->count; i++) {
      ckeys.push_back(options->keys[i]);
      cvalues.push_back(options->values[i]);
    }

    external_delegate_ = external_lib_.create(ckeys.data(), cvalues.data(),
                                              ckeys.size(), nullptr);
    if (external_delegate_) {
      wrapper_delegate_ = {
          reinterpret_cast<void*>(this),  // .data =
          DelegatePrepare,                // .Prepare =
          nullptr,                        // .CopyFromBufferHandle =
          nullptr,                        // .CopyToBufferHandle =
          nullptr,                        // .FreeBufferHandle =
          external_delegate_->flags,      // .flags =
      };
      if (external_delegate_->CopyFromBufferHandle) {
        wrapper_delegate_.CopyFromBufferHandle = DelegateCopyFromBufferHandle;
      }
      if (external_delegate_->CopyToBufferHandle) {
        wrapper_delegate_.CopyToBufferHandle = DelegateCopyToBufferHandle;
      }
      if (external_delegate_->FreeBufferHandle) {
        wrapper_delegate_.FreeBufferHandle = DelegateFreeBufferHandle;
      }
    }
  }
}

ExternalDelegateWrapper::~ExternalDelegateWrapper() {
  if (external_delegate_ != nullptr) {
    external_lib_.destroy(external_delegate_);
  }
}

}  // namespace
}  // namespace tflite

// TfLiteExternalDelegateOptionsInsert adds key/value to the given
// TfLiteExternalDelegateOptions instance.
TfLiteStatus TfLiteExternalDelegateOptionsInsert(
    TfLiteExternalDelegateOptions* options, const char* key,
    const char* value) {
  if (options->count >= kExternalDelegateMaxOptions) {
    return kTfLiteError;
  }
  options->keys[options->count] = key;
  options->values[options->count] = value;
  options->count++;
  return kTfLiteOk;
}

TfLiteExternalDelegateOptions TfLiteExternalDelegateOptionsDefault(
    const char* lib_path) {
  // As 'keys' and 'values' don't need to be set here, using designated
  // initializers may cause a compiling error as "non-trivial designated
  // initializers not supported" by some compiler.
  TfLiteExternalDelegateOptions options;
  options.lib_path = lib_path;
  options.count = 0;
  options.insert = TfLiteExternalDelegateOptionsInsert;
  return options;
}

TfLiteDelegate* TfLiteExternalDelegateCreate(
    const TfLiteExternalDelegateOptions* options) {
  auto* external_delegate_wrapper =
      new tflite::ExternalDelegateWrapper(options);
  if (external_delegate_wrapper) {
    return external_delegate_wrapper->tflite_wrapper_delegate();
  }
  return nullptr;
}

void TfLiteExternalDelegateDelete(TfLiteDelegate* delegate) {
  delete tflite::GetExternalDelegateWrapper(delegate);
}
