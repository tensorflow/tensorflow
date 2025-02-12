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

// This file has utilities that facilitates creating new delegates.
// - SimpleDelegateKernelInterface: Represents a Kernel which handles a subgraph
// to be delegated. It has Init/Prepare/Invoke which are going to be called
// during inference, similar to TFLite Kernels. Delegate owner should implement
// this interface to build/prepare/invoke the delegated subgraph.
// - SimpleDelegateInterface:
// This class wraps TFLiteDelegate and users need to implement the interface and
// then call TfLiteDelegateFactory::CreateSimpleDelegate(...) to get
// TfLiteDelegate* that can be passed to ModifyGraphWithDelegate and free it via
// TfLiteDelegateFactory::DeleteSimpleDelegate(...).
// or call TfLiteDelegateFactory::Create(...) to get a std::unique_ptr
// TfLiteDelegate that can also be passed to ModifyGraphWithDelegate, in which
// case TfLite interpereter takes the memory ownership of the delegate.
#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_

#include <stdint.h>

#include <memory>
#include <utility>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

using TfLiteDelegateUniquePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleDelegateKernelInterface {
 public:
  virtual ~SimpleDelegateKernelInterface() = default;

  // Initializes a delegated subgraph.
  // The nodes in the subgraph are inside TfLiteDelegateParams->nodes_to_replace
  virtual TfLiteStatus Init(TfLiteContext* context,
                            const TfLiteDelegateParams* params) = 0;

  // Will be called by the framework. Should handle any needed preparation
  // for the subgraph e.g. allocating buffers, compiling model.
  // Returns status, and signalling any errors.
  virtual TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) = 0;

  // Actual subgraph inference should happen on this call.
  // Returns status, and signalling any errors.
  // NOTE: Tensor data pointers (tensor->data) can change every inference, so
  // the implementation of this method needs to take that into account.
  virtual TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) = 0;
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
class SimpleDelegateInterface {
 public:
  // Properties of a delegate.  These are used by TfLiteDelegateFactory to
  // help determine how to partition the graph, i.e. which nodes each delegate
  // will get applied to.
  struct Options {
    // Maximum number of delegated subgraph, values <=0 means unlimited.
    int max_delegated_partitions = 0;

    // The minimum number of nodes allowed in a delegated graph, values <=0
    // means unlimited.
    int min_nodes_per_partition = 0;
  };

  virtual ~SimpleDelegateInterface() = default;

  // Returns true if 'node' is supported by the delegate. False otherwise.
  virtual bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                         const TfLiteNode* node,
                                         TfLiteContext* context) const = 0;

  // Initialize the delegate before finding and replacing TfLite nodes with
  // delegate kernels, for example, retrieving some TFLite settings from
  // 'context'.
  virtual TfLiteStatus Initialize(TfLiteContext* context) = 0;

  // Returns a name that identifies the delegate.
  // This name is used for debugging/logging/profiling.
  virtual const char* Name() const = 0;

  // Returns instance of an object that implements the interface
  // SimpleDelegateKernelInterface.
  // An instance of SimpleDelegateKernelInterface represents one subgraph to
  // be delegated.
  // Caller takes ownership of the returned object.
  virtual std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() = 0;

  // Returns SimpleDelegateInterface::Options which has delegate properties
  // relevant for graph partitioning.
  virtual SimpleDelegateInterface::Options DelegateOptions() const = 0;

  /// Optional method for supporting hardware buffers.
  /// Copies the data from delegate buffer handle into raw memory of the given
  /// `tensor`. Note that the delegate is allowed to allocate the raw bytes as
  /// long as it follows the rules for kTfLiteDynamic tensors.
  virtual TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                            TfLiteBufferHandle buffer_handle,
                                            TfLiteTensor* tensor) {
    return kTfLiteError;
  }

  /// Optional method for supporting hardware buffers.
  /// Copies the data from raw memory of the given `tensor` to delegate buffer
  /// handle.
  virtual TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                          TfLiteBufferHandle buffer_handle,
                                          const TfLiteTensor* tensor) {
    return kTfLiteError;
  }

  /// Optional method for supporting hardware buffers.
  /// Frees the Delegate Buffer Handle. Note: This only frees the handle, but
  /// this doesn't release the underlying resource (e.g. textures). The
  /// resources are either owned by application layer or the delegate.
  virtual void FreeBufferHandle(TfLiteContext* context,
                                TfLiteBufferHandle* handle) {}
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
  // A simple usage of the flags bit mask:
  // CreateSimpleDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
  // kTfLiteDelegateFlagsRequirePropagatedShapes)
  static TfLiteDelegate* CreateSimpleDelegate(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate,
      int64_t flags = kTfLiteDelegateFlagsNone);

  // Deletes 'delegate' the passed pointer must be the one returned
  // from CreateSimpleDelegate.
  // This function will destruct the SimpleDelegate object too.
  static void DeleteSimpleDelegate(TfLiteDelegate* delegate);

  // A convenient function wrapping the above two functions and returning a
  // std::unique_ptr type for auto memory management.
  inline static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate) {
    return TfLiteDelegateUniquePtr(
        CreateSimpleDelegate(std::move(simple_delegate)), DeleteSimpleDelegate);
  }
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_SIMPLE_DELEGATE_H_
