/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file has utilities that facilitates creating new opaque delegates.
// - SimpleOpaqueDelegateKernelInterface: Represents a Kernel which handles a
// subgraph to be delegated. It has Init/Prepare/Invoke which are going to be
// called during inference, similar to TFLite Kernels. Delegate owner should
// implement this interface to build/prepare/invoke the delegated subgraph.
// - SimpleOpaqueDelegateInterface:
// This class wraps TFLiteOpaqueDelegate and users need to implement the
// interface and then call
// TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(...) to get a
// TfLiteOpaqueDelegate* that can be passed to
// TfLiteInterpreterOptionsAddDelegate and free it via
// TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(...),
// or call TfLiteOpaqueDelegateFactory::Create(...) to get a std::unique_ptr
// to TfLiteOpaqueDelegate that can also be passed to
// TfLiteInterpreterOptionsAddDelegate.
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_OPAQUE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_OPAQUE_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {

using TfLiteOpaqueDelegateUniquePtr =
    std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleOpaqueDelegateKernelInterface {
 public:
  virtual ~SimpleOpaqueDelegateKernelInterface() = default;

  // Initializes a delegated subgraph.
  // The nodes in the subgraph are inside
  // TfLiteOpaqueDelegateParams->nodes_to_replace
  virtual TfLiteStatus Init(TfLiteOpaqueContext* context,
                            const TfLiteOpaqueDelegateParams* params) = 0;

  // Will be called by the framework. Should handle any needed preparation
  // for the subgraph e.g. allocating buffers, compiling model.
  // Returns status, and signalling any errors.
  virtual TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                               TfLiteOpaqueNode* node) = 0;

  // Actual subgraph inference should happen on this call.
  // Returns status, and signalling any errors.
  // NOTE: Tensor data pointers (tensor->data) can change every inference, so
  // the implementation of this method needs to take that into account.
  virtual TfLiteStatus Eval(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate's capabilities and provides a factory
// for SimpleDelegateKernelInterface.
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate
// - Initialize
// - Name
// - CreateDelegateKernelInterface
// - DelegateOptions
class SimpleOpaqueDelegateInterface {
 public:
  virtual ~SimpleOpaqueDelegateInterface() = default;

  // Returns true if 'node' is supported by the delegate. False otherwise.
  virtual bool IsNodeSupportedByDelegate(
      const TfLiteRegistrationExternal* registration_external,
      const TfLiteOpaqueNode* node, TfLiteOpaqueContext* context) const = 0;

  // Initialize the delegate before finding and replacing TfLite nodes with
  // delegate kernels, for example, retrieving some TFLite settings from
  // 'context'.
  virtual TfLiteStatus Initialize(TfLiteOpaqueContext* context) = 0;

  // Returns a name that identifies the delegate.
  // This name is used for debugging/logging/profiling.
  virtual const char* Name() const = 0;

  // Returns instance of an object that implements the interface
  // SimpleDelegateKernelInterface.
  // An instance of SimpleDelegateKernelInterface represents one subgraph to
  // be delegated.
  // Caller takes ownership of the returned object.
  virtual std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() = 0;
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteOpaqueDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleOpaqueDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
  // A simple usage of the flags bit mask:
  // CreateSimpleDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
  // kTfLiteDelegateFlagsRequirePropagatedShapes)
  static TfLiteOpaqueDelegate* CreateSimpleDelegate(
      std::unique_ptr<SimpleOpaqueDelegateInterface> simple_delegate,
      int64_t flags = kTfLiteDelegateFlagsNone);

  // Deletes 'delegate' the passed pointer must be the one returned from
  // CreateSimpleDelegate. This function will destruct the SimpleDelegate object
  // too.
  static void DeleteSimpleDelegate(TfLiteOpaqueDelegate* opaque_delegate);

  // A convenient function wrapping the above two functions and returning a
  // std::unique_ptr type for auto memory management.
  inline static TfLiteOpaqueDelegateUniquePtr Create(
      std::unique_ptr<SimpleOpaqueDelegateInterface> simple_delegate) {
    return TfLiteOpaqueDelegateUniquePtr(
        CreateSimpleDelegate(std::move(simple_delegate)), DeleteSimpleDelegate);
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_OPAQUE_DELEGATE_H_
