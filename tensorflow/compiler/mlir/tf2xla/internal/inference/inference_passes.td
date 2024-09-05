/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

// Internal TF2XLA Bridge Passes used during interence only.
def InferenceMetricsPass : Pass<"tf2xla-inference-metrics-pass", "ModuleOp"> {
  let summary = "A pass to go over a Module and collect various metrics";
  let description = [{
    Collects metrics about a Module during inference, such as which ops are used.
  }];
  let constructor = "internal::CreateInferenceMetricsPass()";
}