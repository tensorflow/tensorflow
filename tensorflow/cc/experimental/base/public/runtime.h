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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_H_

#include <memory>

#include "tensorflow/c/eager/c_api_experimental.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// Runtime represents an opaque instance of a Tensorflow runtime, with its own
// resources, threadpools, etc. Clients are expected to construct a Runtime
// object through tensorflow::cc::RuntimeBuilder::Build, after setting any
// relevant configuration options. Many Tensorflow functions take a reference to
// the runtime as an argument (eg: tensorflow::cc::SavedModelAPI::Load), and
// may have different implementations depending on the runtime. For many of
// these Runtime-attached objects (such as tensorflow::cc::TensorHandle), the
// Runtime must outlive these objects.
class Runtime {
 public:
  // Runtime is movable, but not copyable.
  Runtime(Runtime&&) = default;
  Runtime& operator=(Runtime&&) = default;

 private:
  friend class RuntimeBuilder;
  friend class SavedModelAPI;
  friend class TensorHandle;

  // Wraps a TFE_Context. Takes ownership of ctx.
  explicit Runtime(TFE_Context* ctx) : ctx_(ctx) {}

  // Deletes the currently wrapped TFE_Context, swaps it with ctx,
  // and takes ownership of ctx.
  void Reset(TFE_Context* ctx) { ctx_.reset(ctx); }

  // Returns the TFE_Context that this object wraps. This object
  // retains ownership of the pointer.
  TFE_Context* GetTFEContext() const { return ctx_.get(); }

  // Runtime is not copyable
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  struct TFEContextDeleter {
    void operator()(TFE_Context* p) const { TFE_DeleteContext(p); }
  };
  std::unique_ptr<TFE_Context, TFEContextDeleter> ctx_;
};

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_H_
