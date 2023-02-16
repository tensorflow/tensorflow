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
#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_PLUGIN_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_PLUGIN_H_

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The mapping utils intended for vendor plugin to track:
//   - TFLite tensor indices to NN API tensor indices mapping.
//   - TFLite node indices to NN API operation indices mapping.
// WARNING: This is an experimental interface that is subject to change.
typedef struct NnapiMappingUtilCInterface {
  // Given a TFLite index, return the ANN index. If it doesn't exist
  // return -1.
  int (*TfLiteIndexToNnIndex)(NnapiMappingUtilCInterface* mapping, int index);

  // When adding a non-tensor TFLite node parameter to NNAPI as an
  // ANeuralNetworksOperand, notify NNAPI delegate to increment the operand
  // count.
  int (*AddNewNonTensorOperand)(NnapiMappingUtilCInterface* mapping);

  // When adding a TFLite tensor to NNAPI as an ANeuralNetworksOperand, notify
  // NNAPI delegate to add a new mapping from `tflite_index` and return the NN
  // API tensor index.
  int (*AddNewNnTensorIndex)(NnapiMappingUtilCInterface* mapping,
                             int tflite_index);

  // When adding a TFLite tensor to NNAPI as multiple ANeuralNetworksOperand
  // objects, for example when splitting one input into several ones, notify
  // NNAPI delegate to increment the operand count.
  int (*AddDelegateGeneratedInputAnnTensorOperand)(
      NnapiMappingUtilCInterface* mapping);

  // Given a TFLite index returns a TFLite type to which a tensor must be
  // converted during copying the data to the memory allocated for NN API.
  // kTfLiteNoType means no conversion is needed.
  TfLiteType (*TfLiteIndexToNnTypeConversion)(
      NnapiMappingUtilCInterface* mapping, int index);

  // Add a new mapping from TFLite tensor index to a type conversion.
  void (*AddTypeConversion)(NnapiMappingUtilCInterface* mapping,
                            int tflite_index, TfLiteType tflite_type);

  // Add a new mapping from TFLite node index to NNAPI op index.
  void (*AddNnapiToTfliteOpMapping)(NnapiMappingUtilCInterface* mapping,
                                    int tflite_node_index);

  // opaque handle for the mapping context. Only intended for the NNAPI Delegate
  // to use.
  void* context;
} NnapiMappingUtilCInterface;

// The interface for NNAPI Vendor Plugin.
// The interface exposes necessary functionalities for NNAPI delegate to
// interact with the vendor plugin.
// WARNING: This is an experimental interface that is subject to change.
typedef struct NnapiDelegateVendorPlugin {
  // Validate whether the given TFLite node is supported by the plugin.
  bool (*ValidateNode)(const TfLiteContext* context,
                       const TfLiteRegistration* registration,
                       const TfLiteNode* node);

  // Translate a TFLite node into corresponding NNAPI operands and operation.
  // It assumes that the call to Validate for has been successful for
  // the operation. In case of success it returns kTfLiteOk and stores the
  // corresponding NNAPI operand indices and operation code through the mapping
  // utility interface. Returns kTfLiteError in case of failures during mapping.
  TfLiteStatus (*MapNode)(TfLiteContext* context, const TfLiteNode* node,
                          int node_index, NnapiMappingUtilCInterface* mapping,
                          ANeuralNetworksModel* model);

  // Parse the provided compilation_hints string and configure it for the given
  // ANeuralNetworksCompilation handle.
  TfLiteStatus (*ConfigureCompilationHints)(
      const char* compilation_hints, ANeuralNetworksCompilation* compilation);

  // Parse the provided execution_hints string and configure it for the given
  // ANeuralNetworksExecution handle.
  TfLiteStatus (*ConfigureExecutionHints)(const char* execution_hints,
                                          ANeuralNetworksExecution* execution);
} NnapiDelegateVendorPlugin;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_PLUGIN_H_
