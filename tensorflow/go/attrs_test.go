/*
Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
	"reflect"
	"testing"
)

func TestOperationAttrs(t *testing.T) {
	g := NewGraph()

	i := 0
	makeConst := func(v interface{}) Output {
		op, err := Const(g, fmt.Sprintf("const/%d/%+v", i, v), v)
		i++
		if err != nil {
			t.Fatal(err)
		}
		return op
	}

	makeTensor := func(v interface{}) *Tensor {
		tensor, err := NewTensor(v)
		if err != nil {
			t.Fatal(err)
		}
		return tensor
	}

	cases := []OpSpec{
		{
			Name: "type",
			Type: "Placeholder",
			Attrs: map[string]interface{}{
				"dtype": Float,
			},
		},
		{
			Name: "list(float)",
			Type: "Bucketize",
			Input: []Input{
				makeConst([]float32{1, 2, 3, 4}),
			},
			Attrs: map[string]interface{}{
				"boundaries": []float32{0, 1, 2, 3, 4, 5},
			},
		},
		{
			Name: "list(float) empty",
			Type: "Bucketize",
			Input: []Input{
				makeConst([]float32{}),
			},
			Attrs: map[string]interface{}{
				"boundaries": []float32(nil),
			},
		},
    /* TODO(ashankar): debug this issue and add it back later.
		{
			Name: "list(type),list(shape)",
			Type: "InfeedEnqueueTuple",
			Input: []Input{
				OutputList([]Output{
					makeConst(float32(1)),
					makeConst([][]int32{{2}}),
				}),
			},
			Attrs: map[string]interface{}{
				"dtypes": []DataType{Float, Int32},
				"shapes": []Shape{ScalarShape(), MakeShape(1, 1)},
			},
		},
		{
			Name: "list(type),list(shape) empty",
			Type: "InfeedEnqueueTuple",
			Input: []Input{
				OutputList([]Output{
					makeConst([][]int32{{2}}),
				}),
			},
			Attrs: map[string]interface{}{
				"dtypes": []DataType{Int32},
				"shapes": []Shape(nil),
			},
		},
		{
			Name: "list(type) empty,string empty,int",
			Type: "_XlaSendFromHost",
			Input: []Input{
				OutputList([]Output{}),
				makeConst(""),
			},
			Attrs: map[string]interface{}{
				"Tinputs":        []DataType(nil),
				"key":            "",
				"device_ordinal": int64(0),
			},
		},
    */
		{
			Name: "list(int),int",
			Type: "StringToHashBucketStrong",
			Input: []Input{
				makeConst(""),
			},
			Attrs: map[string]interface{}{
				"num_buckets": int64(2),
				"key":         []int64{1, 2},
			},
		},
		{
			Name: "list(int) empty,int",
			Type: "StringToHashBucketStrong",
			Input: []Input{
				makeConst(""),
			},
			Attrs: map[string]interface{}{
				"num_buckets": int64(2),
				"key":         ([]int64)(nil),
			},
		},
		{
			Name: "list(string),type",
			Type: "TensorSummary",
			Input: []Input{
				makeConst(""),
			},
			Attrs: map[string]interface{}{
				"T":      String,
				"labels": []string{"foo", "bar"},
			},
		},
		{
			Name: "list(string) empty,type",
			Type: "TensorSummary",
			Input: []Input{
				makeConst(""),
			},
			Attrs: map[string]interface{}{
				"T":      String,
				"labels": ([]string)(nil),
			},
		},
		{
			Name: "tensor",
			Type: "Const",
			Attrs: map[string]interface{}{
				"dtype": String,
				"value": makeTensor("foo"),
			},
		},
	}

	for i, spec := range cases {
		op, err := g.AddOperation(spec)
		if err != nil {
			t.Fatal(err)
		}
		for key, want := range spec.Attrs {
			out, err := op.Attr(key)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out, want) {
				t.Fatalf("%d. %q: Got %#v, wanted %#v", i, key, out, want)
			}
			wantT, ok := want.(*Tensor)
			if ok {
				wantVal := wantT.Value()
				outVal := out.(*Tensor).Value()
				if !reflect.DeepEqual(outVal, wantVal) {
					t.Fatalf("%d. %q: Got %#v, wanted %#v", i, key, outVal, wantVal)
				}
			}
		}
	}
}
