/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPH_QUANTIZE_TRAINING_H_
#define TENSORFLOW_GRAPH_QUANTIZE_TRAINING_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
// Rewrites graph for quantized training.
// Rewrites the forward pass to include the precision loss with quantization so
// the model can learn to deal with such loss and achieve better accuracy when
// it is quantized later for inference.
// Note that the num_bits should be in [1, 63] and 'g' must be not null.
//
// On success, returns OK.
//
// On failure, returns the error status. Possible errors include:
//    - num_bits out of range.
//    - g is null.
//    - More than 1 unknown ops encountered.
Status DoQuantizeTraining(int32 num_bits, Graph* g);

// Converts a input GraphDef and returns a rewritten GraphDef with the
// quantized training.
Status DoQuantizeTrainingOnSerializedGraphDef(const string& input_graph,
                                              int32 num_bits,
                                              string* result_graph);
}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_QUANTIZE_TRAINING_H_
