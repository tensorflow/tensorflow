/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_C_C_API_INTERNAL_H_
#define TENSORFLOW_LITE_C_C_API_INTERNAL_H_

#include <stdarg.h>

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/profiling/telemetry/c/profiler.h"
#include "tensorflow/lite/signature_runner.h"

// Internal structures and subroutines used by the C API. These are likely to
// change and should not be depended on directly by any C API clients.
//
// NOTE: This header does not follow C conventions and does not define a C API.
// It is effectively an (internal) implementation detail of the C API.

struct TfLiteModel {
  // Sharing is safe as FlatBufferModel is const.
  std::shared_ptr<const tflite::impl::FlatBufferModel> impl;
};

// The `TfLiteOpResolver` struct is an abstract callback interface that
// contains function pointers for callbacks that return a
// `TfLiteRegistration` given an op code or custom op name. This mechanism is
// used to map ops referenced in the flatbuffer model to executable function
// pointers (`TfLiteRegistration`s).
// This struct mirrors the tflite::OpResolver C++ abstract base class.
struct TfLiteOpResolverCallbacks {
  // Opaque data that gets passed down to the callback functions.
  void* user_data = nullptr;

  // Callback that finds the op registration for a builtin operator by enum
  // code.  The `user_data` parameter will be set to the
  // `op_resolver_user_data` value that was passed to
  // `TfLiteInterpreterOptionsSetOpResolver`.
  std::function<const TfLiteRegistration*(
      void* user_data, TfLiteBuiltinOperator op, int version)>
      find_builtin_op;
  // Callback that finds the op registration of a custom operator by op name.
  // The `user_data` parameter will be set to the `op_resolver_user_data` value
  // that was passed to `TfLiteInterpreterOptionsSetOpResolver`.
  std::function<const TfLiteRegistration*(void* user_data, const char* op,
                                          int version)>
      find_custom_op;

  // Variant of `find_builtin_op` which returns `TfLiteRegistration_V2`.
  std::function<const TfLiteRegistration_V2*(
      void* user_data, TfLiteBuiltinOperator op, int version)>
      find_builtin_op_v2;
  // Variant of `find_custom_op` which returns `TfLiteRegistration_V2`.
  std::function<const TfLiteRegistration_V2*(void* user_data, const char* op,
                                             int version)>
      find_custom_op_v2;

  // Varant of `find_builtin_op` which returns `TfLiteRegistration_V1`.
  std::function<const TfLiteRegistration_V1*(
      void* user_data, TfLiteBuiltinOperator op, int version)>
      find_builtin_op_v1;
  // Varant of `find_custom_op` which returns `TfLiteRegistration_V1`.
  std::function<const TfLiteRegistration_V1*(void* user_data, const char* op,
                                             int version)>
      find_custom_op_v1;

  // Variant of `find_builtin_op` which returns `TfLiteRegistrationExternal`.
  std::function<const TfLiteRegistrationExternal*(
      void* user_data, TfLiteBuiltinOperator op, int version)>
      find_builtin_op_external;
  // Variant of `find_custom_op` which returns `TfLiteRegistrationExternal`.
  std::function<const TfLiteRegistrationExternal*(void* user_data,
                                                  const char* op, int version)>
      find_custom_op_external;
};

// This struct mirrors the tflite::ErrorResolver C++ abstract base class.
struct TfLiteErrorReporterCallback {
  // Opaque data that gets passed down to the callback function.
  void* user_data = nullptr;

  // Callback function that reports an error.
  void (*error_reporter)(void* user_data, const char* format,
                         va_list args) = nullptr;
};

struct TfLiteInterpreterOptions {
  enum {
    kDefaultNumThreads = -1,
  };
  int num_threads = kDefaultNumThreads;

  tflite::MutableOpResolver mutable_op_resolver;

  TfLiteOpResolverCallbacks op_resolver_callbacks = {};

  std::vector<TfLiteDelegate*> delegates;

  TfLiteErrorReporterCallback error_reporter_callback;

  bool use_nnapi = false;

  // Determines whether to allow automatic fallback to CPU.
  // If true, and if one or more delegates were set,
  // then if Invoke with delegates fails, it will be
  // automatically retried without delegates.
  bool enable_delegate_fallback = false;

  // TfLiteRegistrationExternal objects owned by caller of
  // `TfLiteInterpreterOptionsAddRegistrationExternal` API.
  std::vector<TfLiteRegistrationExternal*> op_registrations;

  // Determines whether to allow to cancel invocations with
  // `Interpreter::Cancel` or `SignatureRunner::Cancel`.
  bool enable_cancellation = false;

  // If not nullptr, report telemetry metrics to profiler.
  TfLiteTelemetryProfilerStruct* telemetry_profiler = nullptr;
};

struct TfLiteInterpreter {
  // Taking a reference to the (const) model data avoids lifetime-related issues
  // and complexity with the TfLiteModel's existence.
  std::shared_ptr<const tflite::impl::FlatBufferModel> model;

  // The interpreter does not take ownership of the provided ErrorReporter
  // instance, so we ensure its validity here. Note that the interpreter may use
  // the reporter in its destructor, so the reporter should be declared first.
  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;

  std::unique_ptr<tflite::Interpreter> impl;

  bool enable_delegate_fallback;
};

struct TfLiteSignatureRunner {
  // The tflite::SignatureRunner runner object that this points to is owned by
  // the interpreter. So this pointer will become invalid when the interpreter
  // is destroyed.
  tflite::SignatureRunner* impl;
};

namespace tflite {
namespace internal {

/// `CallbackOpResolver` is a (C++) `tflite::OpResolver` that forwards the
/// methods to (C ABI) callback functions from a `TfLiteOpResolverCallbacks`
/// struct.
///
/// The SetCallbacks method must be called before calling any of the FindOp
/// methods.
class CallbackOpResolver : public ::tflite::OpResolver {
 public:
  CallbackOpResolver() = default;
  void SetCallbacks(
      const struct TfLiteOpResolverCallbacks& op_resolver_callbacks) {
    op_resolver_callbacks_ = op_resolver_callbacks;
  }
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;

  const TfLiteRegistration* FindOp(const char* op, int version) const override;

 private:
  CallbackOpResolver(const CallbackOpResolver&) = delete;
  CallbackOpResolver& operator=(const CallbackOpResolver&) = delete;

  // Builds a builtin op TfLiteRegistration from a legacy registration
  // (e.g. TfLiteRegistration_V1).
  // The legacy registration type must be a POD struct type whose field types
  // must be a prefix of the field types in TfLiteRegistration, and offset of
  // the first field in TfLiteRegistration that is not present in the legacy
  // registration type must be greater than or equal to the size of the legacy
  // registration type.
  // `legacy_find_builtin_op` is a callback that finds the
  // legacy registration for a builtin operator by enum code. The caller owns
  // the returned registration.
  template <class LegacyRegistrationT>
  TfLiteRegistration* BuildBuiltinOpFromLegacyRegistration(
      tflite::BuiltinOperator op, int version,
      std::function<const LegacyRegistrationT*(
          void* user_data, TfLiteBuiltinOperator op, int version)>
          legacy_find_builtin_op) const {
    if (legacy_find_builtin_op) {
      // Get a deprecated Registration object to create a Registration.
      const LegacyRegistrationT* legacy_registration = legacy_find_builtin_op(
          op_resolver_callbacks_.user_data,
          static_cast<TfLiteBuiltinOperator>(op), version);
      if (legacy_registration) {
        TfLiteRegistration* new_registration = new TfLiteRegistration();
        memcpy(new_registration, legacy_registration,
               sizeof(LegacyRegistrationT));
        new_registration->registration_external = nullptr;
        temporary_builtin_registrations_.push_back(
            std::unique_ptr<TfLiteRegistration>(new_registration));
        return new_registration;
      }
    }
    return nullptr;
  }

  // Builds a custom op TfLiteRegistration from a legacy registration
  // (e.g. TfLiteRegistration_V1).
  // The legacy registration type must be a POD struct type whose field types
  // must be a prefix of the field types in TfLiteRegistration, and offset of
  // the first field in TfLiteRegistration that is not present in the legacy
  // registration type must be greater than or equal to the size of the legacy
  // registration type.
  // `legacy_find_custom_op` is a callback that finds the legacy registration
  // for a builtin operator by op name.
  // The caller owns the returned registration.
  template <class LegacyRegistrationT>
  TfLiteRegistration* BuildCustomOpFromLegacyRegistration(
      const char* op, int version,
      std::function<const LegacyRegistrationT*(void* user_data, const char* op,
                                               int version)>
          legacy_find_custom_op) const {
    if (legacy_find_custom_op) {
      // Get a deprecated Registration object to create a Registration.
      const LegacyRegistrationT* legacy_registration =
          legacy_find_custom_op(op_resolver_callbacks_.user_data, op, version);
      if (legacy_registration) {
        TfLiteRegistration* new_registration = new TfLiteRegistration();
        memcpy(new_registration, legacy_registration,
               sizeof(LegacyRegistrationT));
        new_registration->registration_external = nullptr;
        temporary_custom_registrations_.push_back(
            std::unique_ptr<TfLiteRegistration>(new_registration));
        return new_registration;
      }
    }
    return nullptr;
  }

  struct TfLiteOpResolverCallbacks op_resolver_callbacks_ = {};

  // mutable objects to store temporary `TfLiteRegistration`.
  mutable std::mutex mutex_;
  mutable std::vector<std::unique_ptr<TfLiteRegistration>>
      temporary_builtin_registrations_;  // GUARDED_BY(mutex_)
  mutable std::vector<std::unique_ptr<TfLiteRegistration>>
      temporary_custom_registrations_;  // GUARDED_BY(mutex_)
};

// This adds the builtin and/or custom operators specified in options in
// `optional_options` (if any) to `mutable_resolver`, and then returns a newly
// created TfLiteInterpreter using `mutable_op_resolver` as the default
// OpResolver, and using any other options in `optional_options`, and using
// the provided `model`.
//
// * `model` must be a valid model instance. The caller retains ownership of the
//   object, and can destroy it immediately after creating the interpreter; the
//   interpreter will maintain its own reference to the underlying model data.
// * `optional_options` may be null. The caller retains ownership of the object,
//   and can safely destroy it immediately after creating the interpreter.
// * `mutable_resolver` must not be null. The caller retains ownership of the
//   MutableOpResolver object, and can safely destroy it immediately after
//   creating the interpreter.
//
// NOTE: The client *must* explicitly allocate tensors before attempting to
// access input tensor data or invoke the interpreter.

TfLiteInterpreter* InterpreterCreateWithOpResolver(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options,
    tflite::MutableOpResolver* mutable_resolver);

}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_C_C_API_INTERNAL_H_
