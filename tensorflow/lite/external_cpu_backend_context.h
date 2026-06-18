/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_

#include <memory>
#include <utility>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// This is the base class for TF Lite internal backend contexts (like a
// RUY-based cpu backend context class). A derived internal backend context is
// generally a collection of utilities (i.e. a thread pool etc.) for TF Lite to
// use certain kernel libraries, such as Gemmlowp, RUY, etc., to implement TF
// Lite operators.
class TfLiteInternalBackendContext {
 public:
  virtual ~TfLiteInternalBackendContext() {}

  // Set the maximum number of threads that could be used for parallelizing
  // TfLite computation.
  virtual void SetMaxNumThreads(int max_num_threads) = 0;

  // A context may internally cache prepacked versions of constant tensors for
  // faster computation. This function will clear any caches on the context.
  virtual void ClearCaches() = 0;
};

// This TfLiteExternalContext-derived class is the default
// 'kTfLiteCpuBackendContext'-typed context that's used internally in TF Lite
// framework. The primary purpose of having this class is to allow the same cpu
// backend context to be sharable among a set of TF Lite interpreters so that
// certain system costs are saved, like saving the cost of having multiple
// thread pools in each separate cpu backend context etc..
//
// Note: as of 2019/07/19, such context sharing among a set of interpreters will
// break the execution if these interpreters are invoked simultaneously. It
// works only when these context-sharing interpreters are invoked in a
// serialized way. Here's an example to illustrate the context sharing among 2
// TF Lite interpreters:
//
//  TfLiteExternalContext* global_ctxt = new ExternalCpuBackendContext();
//  interpreter1 = /*...*/;
//  interpreter1->SetExternalContext(kTfLiteCpuBackendContext, global_ctxt);
//  interpreter2 = /*...*/;
//  interpreter2->SetExternalContext(kTfLiteCpuBackendContext, global_ctxt);
//
//  interpreter1->SetNumThreads(2);
//  interpreter1->Invoke();
//
//  interpreter2->SetNumThreads(4);
//  interpreter2->Invoke();
//
// After sharing the context, calling 'SetNumThreads' on any of the
// context-sharing interpreters will have the global impact as it also refreshes
// the #thread info in the global cpu backend context (i.e. 'global_ctxt' above)
// that affects how much parallelism an interpreter invocation will use.
// Therefore, if different number of threads are used among different
// interpreters, don't call 'SetNumThreads' consecutively but call it
// separately between each interpreter's invocation as illustrated above.
//
// Note: it is the responsibility of the user of this context (i.e. a
// TFLiteInterpreter) to clear any state from the internal backend
// context if/when the interpreter no longer needs the shared context.
// See, e.g., TFLiteInterpreter destructor clears caches in the case of a
// shared ExternalCpuBackendContext.
class ExternalCpuBackendContext : public TfLiteExternalContext {
 public:
  ExternalCpuBackendContext();
  ~ExternalCpuBackendContext() {}

  void set_internal_backend_context(
      std::unique_ptr<TfLiteInternalBackendContext> internal_backend_context) {
    internal_backend_context_ = std::move(internal_backend_context);
  }

  TfLiteInternalBackendContext* internal_backend_context() const {
    return internal_backend_context_.get();
  }

 private:
  // Note the actual internal backend context object is lazily initialized.
  std::unique_ptr<TfLiteInternalBackendContext> internal_backend_context_;

  ExternalCpuBackendContext(const ExternalCpuBackendContext&) = delete;
  ExternalCpuBackendContext& operator=(const ExternalCpuBackendContext&) =
      delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_
