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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_H_

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Use TfLiteHexagonDelegateOptionsDefault() for Default options.
struct TFL_CAPI_EXPORT TfLiteHexagonDelegateOptions {
  // This corresponds to the debug level in the hexagon SDK. 0 (default)
  // means no debug.
  int debug_level;

  // This corresponds to powersave_level in the hexagon SDK.
  // where 0 (default) means high performance which means more power
  // consumption.
  int powersave_level;

  // If set to true, performance information about the graph will be dumped
  // to Standard output, this includes cpu cycles.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_profile;

  // If set to true, graph structure will be dumped to Standard output.
  // This is usually beneficial to see what actual nodes executed on
  // the DSP. Combining with 'debug_level' more information will be printed.
  // WARNING: Experimental and subject to change anytime.
  bool print_graph_debug;

  // This sets the maximum number of Hexagon graphs created with
  // hexagon_nn_init. Each graph corresponds to one delegated node subset in the
  // TFLite model.
  int max_delegated_partitions;
  // This sets the minimum number of nodes per graph created with
  // hexagon_nn_init. Defaults to 2.
  int min_nodes_per_partition;

  // If true, then the hexagon graph will adapt for inputs with dynamic batch.
  // See below options are needed to be set.
  // Currently, Only supported when the whole graph is delegated, and
  // with batch as index 0.
  // WARNING: Experimental and subject to change anytime.
  bool enable_dynamic_batch_size;

  // Maximum value for a batch dimension when evaluating graphs with
  // dynamic batch. The input to the graph can have value for batch bigger than
  // this number, internally the graph will run multiple times each with
  // batch dimension <= max_batch_size. you should decide the value of this
  // based on memory/latency tradeoffs.
  // This needs to be set only if 'enable_dynamic_batch_size' is true.
  // Not needed for fixed graphs.
  // WARNING: Experimental and subject to change anytime.
  int max_batch_size;

  // Each element identifies the index of the batch dimension in a single input.
  // input_batch_dimensions->data[i] is the index of the batch dimension for
  // input[i]. If the graph has 1 input then the size of the array should be 1,
  // and so on. This needs to be set only if 'enable_dynamic_batch_size' is
  // true. Not needed for fixed graphs.
  // If input[i] doesn't have dynamic batch, then input_batch_dimensions[i]
  // should be -1.
  // Delegate will take ownership of the pointer.
  // WARNING: Experimental and subject to change anytime.
  TfLiteIntArray* input_batch_dimensions;

  // Each element identifies the index of the batch dimension in a single
  // output. output_batch_dimensions->data[i] is the index of the batch
  // dimension for output[i]. If the graph has 1 output then the size of the
  // array should be 1, and so on. This needs to be set only if
  // 'enable_dynamic_batch_size' is true. Not needed for fixed graphs. If
  // output[i] has doesn't have dynamic batch, then output_batch_dimensions[i]
  // should be -1. Delegate will take ownership of the pointer. WARNING:
  // Experimental and subject to change anytime.
  TfLiteIntArray* output_batch_dimensions;
};

// Return a delegate that uses Hexagon SDK for ops execution.
// Must outlive the interpreter.
TfLiteDelegate* TFL_CAPI_EXPORT
TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);

// Returns TfLiteHexagonDelegateOptions populated with default values.
TFL_CAPI_EXPORT TfLiteHexagonDelegateOptions
TfLiteHexagonDelegateOptionsDefault();

// Do any needed cleanup and delete 'delegate'.
void TFL_CAPI_EXPORT TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);

// Initializes the DSP connection.
// This should be called before doing any usage of the delegate.
// "lib_directory_path": Path to the directory which holds the
// shared libraries for the Hexagon NN libraries on the device.
void TFL_CAPI_EXPORT TfLiteHexagonInitWithPath(const char* lib_directory_path);

// Same as above method but doesn't accept the path params.
// Assumes the environment setup is already done. Only initialize Hexagon.
void TFL_CAPI_EXPORT TfLiteHexagonInit();

// Clean up and switch off the DSP connection.
// This should be called after all processing is done and delegate is deleted.
void TFL_CAPI_EXPORT TfLiteHexagonTearDown();
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_H_
