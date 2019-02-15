/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/api_def/excluded_ops.h"

namespace tensorflow {

const std::unordered_set<std::string>* GetExcludedOps() {
  static std::unordered_set<std::string>* excluded_ops =
      new std::unordered_set<std::string>(
          {"BigQueryReader", "GenerateBigQueryReaderPartitions",
           "GcsConfigureBlockCache", "GcsConfigureCredentials",
#ifdef INTEL_MKL
           // QuantizedFusedOps for Intel CPU
           "QuantizedConcatV2", "QuantizedConv2DAndRequantize",
           "QuantizedConv2DWithBias", "QuantizedConv2DWithBiasAndRequantize",
           "QuantizedConv2DAndRelu", "QuantizedConv2DAndReluAndRequantize",
           "QuantizedConv2DWithBiasAndRelu",
           "QuantizedConv2DWithBiasAndReluAndRequantize",
           "QuantizedConv2DWithBiasSumAndRelu",
           "QuantizedConv2DWithBiasSumAndReluAndRequantize",
           "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
#endif  // INTEL_MKL
          });
  return excluded_ops;
}
}  // namespace tensorflow
