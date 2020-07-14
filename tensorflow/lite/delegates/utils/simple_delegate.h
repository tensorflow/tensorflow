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

#include <memory>

#include "tensorflow/lite/c/common.h"

namespace tflite {

using TfLiteDelegateUniquePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleDelegateKernelInterface {
 public:
  virtual ~SimpleDelegateKernelInterface() {}

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
  virtual TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate capabilities and provide factory
// for SimpleDelegateKernelInterface
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate
// - Initialize
// - name
// - CreateDelegateKernelInterface
class SimpleDelegateInterface {
 public:
  // Options for configuring a delegate.
  struct Options {
    // Maximum number of delegated subgraph, values <=0 means unlimited.
    int max_delegated_partitions = 0;

    // The minimum number of nodes allowed in a delegated graph, values <=0
    // means unlimited.
    int min_nodes_per_partition = 0;
  };

  virtual ~SimpleDelegateInterface() {}

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

  // Returns SimpleDelegateInterface::Options which has the delegate options.
  virtual SimpleDelegateInterface::Options DelegateOptions() const = 0;
};

// Factory class that provides static methods to deal with SimpleDelegate
// creation and deletion.
class TfLiteDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteSimpleDelegate.
  static TfLiteDelegate* CreateSimpleDelegate(
      std::unique_ptr<SimpleDelegateInterface> simple_delegate);

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
