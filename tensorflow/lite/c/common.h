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

// This file defines common C types and APIs for implementing operations,
// delegates and other constructs in TensorFlow Lite. The actual operations and
// delegates can be defined using C++, but the interface between the interpreter
// and the operations are C.
//
// Summary of abstractions
// TF_LITE_ENSURE - Self-sufficient error checking
// TfLiteStatus - Status reporting
// TfLiteIntArray - stores tensor shapes (dims),
// TfLiteContext - allows an op to access the tensors
// TfLiteTensor - tensor (a multidimensional array)
// TfLiteNode - a single node or operation
// TfLiteRegistration - the implementation of a conceptual operation.
// TfLiteDelegate - allows delegation of nodes to alternative backends.
//
// Some abstractions in this file are created and managed by Interpreter.
//
// NOTE: The order of values in these structs are "semi-ABI stable". New values
// should be added only to the end of structs and never reordered.

#ifndef TENSORFLOW_LITE_C_COMMON_H_
#define TENSORFLOW_LITE_C_COMMON_H_

#include "tensorflow/lite/core/c/common.h"

#endif  // TENSORFLOW_LITE_C_COMMON_H_
