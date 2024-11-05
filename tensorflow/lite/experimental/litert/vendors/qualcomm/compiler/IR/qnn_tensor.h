// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_IR_QNN_TENSOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_IR_QNN_TENSOR_H_

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

namespace litert::qnn {

//
// Initialize QNN Tensors.
//

// NOTE: Within LiteRt land, all Qnn Tensors are treated as "v2". Any
// referential data (like dimensions : uint32_t*) within a QNN Tensor
// is allocated with "new" and must be explicitly cleaned up with ResetTensor.

// Construct a "blank" QNN Tensor.
Qnn_Tensor_t BuildDefaultTensor();

// Construct a "blank" QNN Tensor with given id.
Qnn_Tensor_t BuildDefaultTensor(uint32_t id);

// Constructa a "blank" QNN Tensor meant to be used as a graph input.
Qnn_Tensor_t BuildInputTensor();

// Constructa a "blank" QNN Tensor meant to be used as a graph output.
Qnn_Tensor_t BuildOutputTensor();

Qnn_ClientBuffer_t BuildDefaultClientBuffer();

// Adds attributes to given tensor making it amenable for use as graph input.
void SetInputTensorAttrs(Qnn_Tensor_t& tensor);

// Adds attributes to given tensor making it amenable for use as graph output.
void SetOutputTensorAttrs(Qnn_Tensor_t& tensor);

// Adds attributes to given tensor making it amenable for uses a intermediate
// output.
void SetResultTensorAttrs(Qnn_Tensor_t& tensor);

// Reset the given tensor, deallocating anything on the heap that it points to.
void ResetTensor(Qnn_Tensor_t& tensor);

// Resets all fields other than id in the given tensor and returns the id for
// convenience. Only the id is needed to traffic QNN Tensors after they have
// been registered with the context.
uint32_t MoveToId(Qnn_Tensor_t& tensor);

//
// Legalize LiteRt Tensors to Analogous QNN Construct.
//

// Map src tensor onto dest. Resets dest before doing anything.
LiteRtStatus LegalizeTensor(const litert::Tensor& src, Qnn_Tensor_t& dest);

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_IR_QNN_TENSOR_H_
