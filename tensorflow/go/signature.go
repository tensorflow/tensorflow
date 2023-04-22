/*
Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

import corepb "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto"

// #include "tensorflow/c/c_api.h"
import "C"

// A Signature defines the signature of a computation supported by a TensorFlow
// graph.
//
// For example, a model with two loss computations, sharing a single input,
// might have the following signature_def map.
//
// Note that across the two Signatures "loss_A" and "loss_B", the input key,
// output key, and method_name are identical, and will be used by system(s) that
// implement or rely upon this particular loss method. The output tensor names
// differ, demonstrating how different outputs can exist for the same method.
//
// signature_def {
//   key: "loss_A"
//   value {
//     inputs {
//       key: "input"
//       value {
//         name: "input:0"
//         dtype: DT_STRING
//         tensor_shape: ...
//       }
//     }
//     outputs {
//       key: "loss_output"
//       value {
//         name: "loss_output_A:0"
//         dtype: DT_FLOAT
//         tensor_shape: ...
//       }
//     }
//   }
//   ...
//   method_name: "some/package/compute_loss"
// }
// signature_def {
//   key: "loss_B"
//   value {
//     inputs {
//       key: "input"
//       value {
//         name: "input:0"
//         dtype: DT_STRING
//         tensor_shape: ...
//       }
//     }
//     outputs {
//       key: "loss_output"
//       value {
//         name: "loss_output_B:0"
//         dtype: DT_FLOAT
//         tensor_shape: ...
//       }
//     }
//   }
//   ...
//   method_name: "some/package/compute_loss"
// }
type Signature struct {
	Inputs, Outputs map[string]TensorInfo
	MethodName      string
}

// A TensorInfo contains the information about a Tensor necessary for feeding or retrieval.
type TensorInfo struct {
	Name  string
	DType DataType
	Shape Shape
}

func signatureDefFromProto(pb *corepb.SignatureDef) Signature {
	inputs := make(map[string]TensorInfo)
	for name, input := range pb.GetInputs() {
		inputs[name] = tensorInfoFromProto(input)
	}
	outputs := make(map[string]TensorInfo)
	for name, output := range pb.GetOutputs() {
		outputs[name] = tensorInfoFromProto(output)
	}
	return Signature{
		Inputs:     inputs,
		Outputs:    outputs,
		MethodName: pb.GetMethodName(),
	}
}

func tensorInfoFromProto(pb *corepb.TensorInfo) TensorInfo {
	var dims []int64
	for _, d := range pb.GetTensorShape().GetDim() {
		dims = append(dims, d.GetSize())
	}
	return TensorInfo{
		Name:  pb.GetName(),
		DType: DataType(C.TF_DataType(pb.GetDtype())),
		Shape: MakeShape(dims...),
	}
}
