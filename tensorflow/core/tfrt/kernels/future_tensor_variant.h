/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_KERNELS_FUTURE_TENSOR_VARIANT_H_
#define TENSORFLOW_CORE_TFRT_KERNELS_FUTURE_TENSOR_VARIANT_H_

#include <string>
#include <utility>

#include "xla/tsl/concurrency/future.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace tensorflow {

//   This C++ code defines the FutureTensorVariant class, which is a wrapper
//   around a tsl::Future<tensorflow::Tensor>.

// A tsl::Future<T> is a construct for asynchronous programming that represents
// a value that will become available in the future. In this case, the value is
// a tensorflow::Tensor.

// The FutureTensorVariant class allows a tsl::Future<tensorflow::Tensor> to be
// stored inside a tensorflow::Tensor of type DT_VARIANT. This is a key part of
// the asynchronous execution model being introduced in this changelist.

// Here's how it works with the other changes:

// The IfrtCallOp now executes an IFRT program asynchronously. Instead of
// blocking to wait for the result, it immediately returns a DT_VARIANT tensor
// that contains a FutureTensorVariant object. This FutureTensorVariant holds
// the "promise" of a result tensor that is being computed in the background.
// The new IfrtAwaitOp takes this variant tensor as input, extracts the
// FutureTensorVariant, and waits for the future to be resolved to get the
// actual tensorflow::Tensor. The Encode and Decode methods are not implemented
// because serializing a future is not meaningful; it's a runtime object tied to
// an active computation.
class FutureTensorVariant {
 public:
  FutureTensorVariant() = default;
  explicit FutureTensorVariant(tsl::Future<tensorflow::Tensor> future)
      : future_(std::move(future)) {}

  const tsl::Future<tensorflow::Tensor>& future() const { return future_; }

  std::string TypeName() const { return "FutureTensorVariant"; }
  void Encode(VariantTensorData* data) const {}
  bool Decode(const VariantTensorData& data) { return false; }
  std::string DebugString() const { return "FutureTensorVariant"; }

 private:
  tsl::Future<tensorflow::Tensor> future_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_KERNELS_FUTURE_TENSOR_VARIANT_H_
