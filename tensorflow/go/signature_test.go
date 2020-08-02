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

import (
	"fmt"
	"testing"

	tspb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto"
	typb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/types_go_proto"
	corepb "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto"
)

func TestSignatureFromProto(t *testing.T) {
	got := signatureDefFromProto(&corepb.SignatureDef{
		Inputs: map[string]*corepb.TensorInfo{
			"input_1": &corepb.TensorInfo{
				Encoding: &corepb.TensorInfo_Name{
					Name: "tensor_1",
				},
				Dtype: typb.DataType_DT_INT8,
				TensorShape: &tspb.TensorShapeProto{
					Dim: []*tspb.TensorShapeProto_Dim{
						{Size: 1},
						{Size: 2},
						{Size: 3},
					},
				},
			},
			"input_2": &corepb.TensorInfo{
				Encoding: &corepb.TensorInfo_Name{
					Name: "tensor_2",
				},
				Dtype: typb.DataType_DT_FLOAT,
				TensorShape: &tspb.TensorShapeProto{
					Dim: []*tspb.TensorShapeProto_Dim{
						{Size: 4},
						{Size: 5},
						{Size: 6},
					},
				},
			},
		},
		Outputs: map[string]*corepb.TensorInfo{
			"output_1": &corepb.TensorInfo{
				Encoding: &corepb.TensorInfo_Name{
					Name: "tensor_3",
				},
				Dtype: typb.DataType_DT_STRING,
				TensorShape: &tspb.TensorShapeProto{
					Dim: []*tspb.TensorShapeProto_Dim{
						{Size: 1},
						{Size: 2},
						{Size: 3},
					},
				},
			},
			"output_2": &corepb.TensorInfo{
				Encoding: &corepb.TensorInfo_Name{
					Name: "tensor_4",
				},
				Dtype: typb.DataType_DT_BOOL,
				TensorShape: &tspb.TensorShapeProto{
					Dim: []*tspb.TensorShapeProto_Dim{
						{Size: 4},
						{Size: 5},
						{Size: 6},
					},
				},
			},
		},
		MethodName: "method",
	})

	want := Signature{
		Inputs: map[string]TensorInfo{
			"input_1": TensorInfo{
				Name:  "tensor_1",
				DType: Int8,
				Shape: MakeShape(1, 2, 3),
			},
			"input_2": TensorInfo{
				Name:  "tensor_2",
				DType: Float,
				Shape: MakeShape(4, 5, 6),
			},
		},
		Outputs: map[string]TensorInfo{
			"output_1": TensorInfo{
				Name:  "tensor_3",
				DType: String,
				Shape: MakeShape(1, 2, 3),
			},
			"output_2": TensorInfo{
				Name:  "tensor_4",
				DType: Bool,
				Shape: MakeShape(4, 5, 6),
			},
		},
		MethodName: "method",
	}

	for k, input := range want.Inputs {
		diff, err := diffTensorInfos(got.Inputs[k], input)
		if err != nil {
			t.Fatalf("Signature.Inputs[%s]: unable to diff TensorInfos: %v", k, err)
		}
		if diff != "" {
			t.Errorf("Signature.Inputs[%s] diff:\n%s", k, diff)
		}
	}

	for k, output := range want.Outputs {
		diff, err := diffTensorInfos(got.Outputs[k], output)
		if err != nil {
			t.Fatalf("Signature.Outputs[%s]: unable to diff TensorInfos: %v", k, err)
		}
		if diff != "" {
			t.Errorf("Signature.Outputs[%s] diff:\n%s", k, diff)
		}
	}

	if got.MethodName != want.MethodName {
		t.Errorf("Signature.MethodName: got %q, want %q", got.MethodName, want.MethodName)
	}
}

func TestTensorInfoFromProto(t *testing.T) {
	got := tensorInfoFromProto(&corepb.TensorInfo{
		Encoding: &corepb.TensorInfo_Name{
			Name: "tensor",
		},
		Dtype: typb.DataType_DT_INT8,
		TensorShape: &tspb.TensorShapeProto{
			Dim: []*tspb.TensorShapeProto_Dim{
				{Size: 1},
				{Size: 2},
				{Size: 3},
			},
		},
	})
	want := TensorInfo{
		Name:  "tensor",
		DType: Int8,
		Shape: MakeShape(1, 2, 3),
	}

	diff, err := diffTensorInfos(got, want)
	if err != nil {
		t.Fatalf("Unable to diff TensorInfos: %v", err)
	}
	if diff != "" {
		t.Errorf("tensorInfoFromProto produced a diff (got -> want): %s", diff)
	}
}

func diffTensorInfos(a, b TensorInfo) (string, error) {
	diff := ""
	if a.Name != b.Name {
		diff += fmt.Sprintf("Name: %q -> %q\n", a.Name, b.Name)
	}
	if a.DType != b.DType {
		diff += fmt.Sprintf("DType: %v -> %v\n", a.DType, b.DType)
	}

	aShape, err := a.Shape.ToSlice()
	if err != nil {
		return "", err
	}
	bShape, err := b.Shape.ToSlice()
	if err != nil {
		return "", err
	}
	shapeLen := len(aShape)
	if len(bShape) > shapeLen {
		shapeLen = len(bShape)
	}
	for i := 0; i < shapeLen; i++ {
		if i >= len(aShape) {
			diff += fmt.Sprintf("+Shape[%d]: %d\n", i, bShape[i])
			continue
		}
		if i >= len(bShape) {
			diff += fmt.Sprintf("-Shape[%d]: %d\n", i, aShape[i])
			continue
		}
		if aShape[i] != bShape[i] {
			diff += fmt.Sprintf("Shape[%d]: %d -> %d\n", i, aShape[i], bShape[i])
		}
	}

	return diff, nil
}
