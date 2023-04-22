/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_

#include <sys/types.h>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/experimental/libtf/object.h"
#include "tensorflow/cc/experimental/libtf/value.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

namespace tf {
namespace libtf {
namespace runtime {

/// @brief A runtime object capable of loading modules and executing functions.
///
/// It is the responsibility of the owner of the Runtime to keep it alive longer
/// than all imported modules.
class Runtime : public Object {
 public:
  // TODO(b/191264214): Remove need for AbstractContext
  explicit Runtime(tensorflow::AbstractContext* ctx);
  /// @brief Loads the module indicated by `name` and returns it.
  ///
  /// @param name The name of the module / file path to load
  /// @return An `Object` representing the module, if successful.  Otherwise, a
  /// non-ok `absl::Status`.
  tensorflow::StatusOr<Object> Load(const String& name);
  // TODO(b/186787000): Loading a module with identically-named functions as
  // a previously loaded module results in undefined behavior. This
  // functionality will be supported in the future.

  // Create a host tensor and copy data into it.
  //
  // Raises an error if shape or dtype are incompatible with T.
  // TODO(b/189458441): Update this when we decide on the representation of
  // shape and dtype in this API.
  // Disclaimer: This API is subject to change as we add support for creating
  // device tensors b/187222691 and enable buffer re-use b/187223179.
  // TODO(b/190715501): Make this available via a soft API as well.
  template <class T>
  tensorflow::StatusOr<Tensor> CreateHostTensor(absl::Span<const int64_t> shape,
                                                int dtype,
                                                absl::Span<const T> data);
};

template <class T>
tensorflow::StatusOr<Tensor> Runtime::CreateHostTensor(
    absl::Span<const int64_t> shape, int dtype, absl::Span<const T> data) {
  size_t num_elements = 1;
  for (int dim = 0; dim < shape.size(); dim++) {
    if (shape[dim] < 0) {
      return tensorflow::errors::InvalidArgument(absl::StrCat(
          "Shape must be fully-defined, got: shape[", dim, "] = ", shape[dim]));
    }
    num_elements *= shape[dim];
  }
  if (data.size() != num_elements) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Mismatched shape and data size: \n", "Shape num_elements: ",
        num_elements, "\n", "Data size: ", data.size(), "\n"));
  }
  auto maybe_capsule = Get<internal::Capsule>(String("ctx"));
  if (!maybe_capsule.status().ok()) {
    return maybe_capsule.status();
  }
  auto capsule = maybe_capsule.ValueOrDie();
  auto ctx = capsule.cast<tensorflow::ImmediateExecutionContext*>();
  tensorflow::AbstractTensorPtr t(
      ctx->CreateTensor(static_cast<tensorflow::DataType>(dtype), shape));
  // TODO(srbs): This is still a weak check. Check that dtype and T are
  // compatible.
  if (t->ByteSize() != sizeof(T) * data.size()) {
    return tensorflow::errors::InvalidArgument(absl::StrCat(
        "Invalid number of bytes in data buffer\n", "Expected bytes: ",
        t->ByteSize(), "\n", "Actual bytes: ", sizeof(T) * data.size()));
  }
  memcpy(t->Data(), data.data(), t->ByteSize());
  return Tensor(Convert(TaggedValue(
      impl::TaggedValueTensor(ctx->CreateLocalHandle(t.get()), false))));
}

}  // namespace runtime
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_RUNTIME_H_
