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

#ifndef TENSORFLOW_C_EAGER_MNIST_UTIL_H_
#define TENSORFLOW_C_EAGER_MNIST_UTIL_H_

#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include <cstdlib>
#include <string>
using tensorflow::string;

// Graph Tracing for abstract operation MatMul
TF_AbstractTensor* AbstractMatMul(TF_AbstractTensor* A, TF_AbstractTensor* B, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s);

// Graph Tracing for abstract operation Add
TF_AbstractTensor* AbstractAdd(TF_AbstractTensor* A, TF_AbstractTensor* B, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s);

// Graph Tracing for abstract operation Relu
TF_AbstractTensor* AbstractRelu(TF_AbstractTensor* A, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s);

// Graph Tracing for abstract Softmax Cross Entropy Loss. Returns scalar loss
TF_AbstractTensor* AbstractSparseSoftmaxCrossEntropyLoss(TF_AbstractTensor* scores, TF_AbstractTensor* y_labels, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s);

// Returns abstract function and frees memory associated with the graph context
TF_AbstractFunction* AbstractFinalizeFunction(TF_OutputList* func_outputs, TF_ExecutionContext* graph_ctx, TF_Status* s);

// Returns a 2-layer AbstractFunction* that traces a 2-layer MNIST Model
TF_AbstractFunction* getAbstractMNISTForward(TF_Status* s, string fn_name);
#endif  // TENSORFLOW_C_EAGER_C_API_TEST_UTIL_H_