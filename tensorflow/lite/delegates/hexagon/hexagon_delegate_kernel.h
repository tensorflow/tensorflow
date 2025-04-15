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
#ifndef TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_KERNEL_H_

#include <time.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hexagon/hexagon_nn_ops.h"
#include "hexagon/hexagon_nn.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Represents an abstraction of a Hexagon NNLib graph with functionality to
// initialize, prepare and invoke it based on the TFLite subgraph to be
// delegated.
class HexagonDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit HexagonDelegateKernel(const ::TfLiteHexagonDelegateOptions& params)
      : params_(params) {}

  // Initialize the Hexagon graph and add required nodes.
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;

  // Prepare the Hexagon graph with hexagon_nn_prepare.
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

  // Allocate Hexagon tensordefs for graph I/O & execute it.
  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

  ~HexagonDelegateKernel() override;

  // Sets the environment required for Hexagon execution: DSP attributes,
  // rpcmem, etc.
  static void InitState();

  // Teardown the environment initialized in InitState.
  static void Teardown();

 private:
  // Builds the Hexagon graph based on delegated TFLite subgraph.
  TfLiteStatus BuildGraph(TfLiteContext* context,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors);

  void ReportError(TfLiteContext* context, const std::string& msg);

  // Resizes output tensors in case the delegate has dynamic batch enabled.
  // Returns Error otherwise or if the requested size is invalid.
  TfLiteStatus ResizeOutputTensors(TfLiteContext* context, TfLiteNode* node);

  void PrintLog();

  // Prints performance information about the graph including cycles per node.
  // If 'profiler' is not nullptr data will be added to it.
  void PrintPerformanceData(Profiler* profiler);

  // Print debugging information about the graph constructed.
  // Amount of information can be increased with debug level.
  void PrintDebuggingGraph();

  const HexagonNN* hexagon_nn_ = nullptr;  // Not owned.
  std::unique_ptr<delegates::hexagon::GraphBuilder> builder_;
  hexagon_nn_nn_id graph_id_ = -1;
  // Indices of nodes in the delegated TfLite subgraph.
  std::vector<int> nodes_;
  ::TfLiteHexagonDelegateOptions params_;

  // Whether the Hexagon graph is prepared or not.
  bool graph_prepared_ = false;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_HEXAGON_HEXAGON_DELEGATE_KERNEL_H_
