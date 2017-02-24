// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"bytes"
	"go/format"
	"testing"

	"github.com/golang/protobuf/proto"
	pb "github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/tensorflow/core/framework"
)

func TestGenerateOp(t *testing.T) {
	// TestGenerateOp validates the generated source code for an op.
	// The OpDef for the test cases are simplified forms of real ops.
	testdata := []struct {
		tag    string
		opdef  string
		wanted string
	}{
		{
			tag: "NoOp",
			opdef: `
name: "NoOp"
summary: "No. Op."
`,
			wanted: `
// No. Op.
//
// Returns the created operation.
func NoOp(scope *Scope) (o *tf.Operation) {
	if scope.Err() != nil {
		return
	}
	opspec := tf.OpSpec{
		Type: "NoOp",
	}
	return scope.AddOperation(opspec)
}
`,
		},
		{
			tag: "NoAttributes",
			opdef: `
name: "Add"
input_arg: <
  name: "x"
  type_attr: "T"
>
input_arg: <
  name: "y"
  type_attr: "T"
>
output_arg: <
  name: "z"
  type_attr: "T"
>
attr: <
  name: "T"
  type: "type"
  allowed_values: <
    list: <
      type: DT_FLOAT
      type: DT_INT64
    >
  >
>
summary: "Returns x + y element-wise."
description: "Blah blah",
`,
			wanted: `
// Returns x + y element-wise.
//
// Blah blah
func Add(scope *Scope, x tf.Output, y tf.Output) (z tf.Output) {
	if scope.Err() != nil {
		return
	}
	opspec := tf.OpSpec{
		Type: "Add",
		Input: []tf.Input{
			x, y,
		},
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
}
`,
		},
		{
			tag: "RequiredAttributes",
			opdef: `
name: "Cast"
input_arg: <
  name: "x"
  type_attr: "SrcT"
>
output_arg: <
  name: "y"
  type_attr: "DstT"
>
attr: <
  name: "SrcT"
  type: "type"
>
attr: <
  name: "DstT"
  type: "type"
>
summary: "Cast x of type SrcT to y of DstT."
`,
			wanted: `
// Cast x of type SrcT to y of DstT.
func Cast(scope *Scope, x tf.Output, DstT tf.DataType) (y tf.Output) {
	if scope.Err() != nil {
		return
	}
	attrs := map[string]interface{}{"DstT": DstT}
	opspec := tf.OpSpec{
		Type: "Cast",
		Input: []tf.Input{
			x,
		},
		Attrs: attrs,
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
}
`,
		},
		{
			tag: "OptionalAttributes",
			opdef: `
name: "DecodeJpeg"
input_arg: <
  name: "contents"
  description: "0-D.  The JPEG-encoded image."
  type: DT_STRING
>
output_arg: <
  name: "image"
  description: "3-D with shape [height, width, channels]"
  type: DT_UINT8
>
attr: <
  name: "channels"
  type: "int"
  default_value: <
    i: 0
  >
  description: "Number of color channels for the decoded image."
>
attr: <
  name: "fancy_upscaling"
  type: "bool"
  default_value: <
    b: true
  >
  description: "If true use a slower but nicer upscaling of the\nchroma planes (yuv420/422 only)."
>
attr: <
  name: "acceptable_fraction"
  type: "float"
  default_value: <
    f: 1
  >
  description: "The minimum required fraction of lines before a truncated\ninput is accepted."
>
summary: "Decode a JPEG-encoded image to a uint8 tensor."
description: "Norna dorna fjord\nkajorna\nhahaha"
`,
			wanted: `
// DecodeJpegAttr is an optional argument to DecodeJpeg.
type DecodeJpegAttr func(optionalAttr)

// DecodeJpegChannels sets the optional channels attribute to value.
//
// value: Number of color channels for the decoded image.
// If not specified, defaults to i:0
func DecodeJpegChannels(value int64) DecodeJpegAttr {
	return func(m optionalAttr) {
		m["channels"] = value
	}
}

// DecodeJpegFancyUpscaling sets the optional fancy_upscaling attribute to value.
//
// value: If true use a slower but nicer upscaling of the
// chroma planes (yuv420/422 only).
// If not specified, defaults to b:true
func DecodeJpegFancyUpscaling(value bool) DecodeJpegAttr {
	return func(m optionalAttr) {
		m["fancy_upscaling"] = value
	}
}

// DecodeJpegAcceptableFraction sets the optional acceptable_fraction attribute to value.
//
// value: The minimum required fraction of lines before a truncated
// input is accepted.
// If not specified, defaults to f:1
func DecodeJpegAcceptableFraction(value float32) DecodeJpegAttr {
	return func(m optionalAttr) {
		m["acceptable_fraction"] = value
	}
}

// Decode a JPEG-encoded image to a uint8 tensor.
//
// Norna dorna fjord
// kajorna
// hahaha
//
// Arguments:
//	contents: 0-D.  The JPEG-encoded image.
//
// Returns 3-D with shape [height, width, channels]
func DecodeJpeg(scope *Scope, contents tf.Output, optional ...DecodeJpegAttr) (image tf.Output) {
	if scope.Err() != nil {
		return
	}
	attrs := map[string]interface{}{}
	for _, a := range optional {
		a(attrs)
	}
	opspec := tf.OpSpec{
		Type: "DecodeJpeg",
		Input: []tf.Input{
			contents,
		},
		Attrs: attrs,
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
}
`,
		},
		{
			tag: "MultipleOutputs",
			opdef: `
name: "TwoOutputs"
input_arg: <
  name: "input"
  type_attr: "T"
>
output_arg <
  name: "x"
  type_attr: "T"
>
output_arg <
  name: "y"
  type_attr: "T"
>
attr: <
  name: "T"
  type: "type"
>
summary: "Op that produces multiple outputs"
`,
			wanted: `
// Op that produces multiple outputs
func TwoOutputs(scope *Scope, input tf.Output) (x tf.Output, y tf.Output) {
        if scope.Err() != nil {
                return
        }
        opspec := tf.OpSpec{
                Type: "TwoOutputs",
                Input: []tf.Input{
                        input,
                },
        }
        op := scope.AddOperation(opspec)
        return op.Output(0), op.Output(1)
}
`,
		},
		{
			tag: "ListOutput",
			opdef: `
name: "ShapeN"
input_arg: <
  name: "input"
  type_attr: "T"
  number_attr: "N"
>
output_arg: <
  name: "output"
  type_attr: "out_type"
  number_attr: "N"
>
attr: <
  name: "N"
  type: "int"
  has_minimum: true
  minimum: 1
>
attr: <
  name: "T"
  type: "type"
>
attr: <
  name: "out_type"
  type: "type"
  default_value: <
    type: DT_INT32
  >
  allowed_values: <
    list: <
      type: DT_INT32
      type: DT_INT64
    >
  >
>
summary: "Returns shape of tensors."
description: "Some description here."
`,
			wanted: `
// ShapeNAttr is an optional argument to ShapeN.
type ShapeNAttr func(optionalAttr)

// ShapeNOutType sets the optional out_type attribute to value.
// If not specified, defaults to type:DT_INT32
func ShapeNOutType(value tf.DataType) ShapeNAttr {
	return func(m optionalAttr) {
		m["out_type"] = value
	}
}

// Returns shape of tensors.
//
// Some description here.
func ShapeN(scope *Scope, input []tf.Output, optional ...ShapeNAttr) (output []tf.Output) {
	if scope.Err() != nil {
		return
	}
	attrs := map[string]interface{}{}
	for _, a := range optional {
		a(attrs)
	}
	opspec := tf.OpSpec{
		Type: "ShapeN",
		Input: []tf.Input{
			tf.OutputList(input),
		},
		Attrs: attrs,
	}
	op := scope.AddOperation(opspec)
	if scope.Err() != nil {
		return
	}
	var idx int
	var err error
	if output, idx, err = makeOutputList(op, idx, "output"); err != nil {
		scope.UpdateErr("ShapeN", err)
		return
	}
	return output
}
`,
		},
	}

	for _, test := range testdata {
		t.Run(test.tag, func(t *testing.T) {
			var opdef pb.OpDef
			var buf bytes.Buffer
			if err := proto.UnmarshalText(test.opdef, &opdef); err != nil {
				t.Fatal(err)
			}
			if err := generateFunctionForOp(&buf, &opdef); err != nil {
				t.Fatal(err)
			}
			got, err := format.Source(buf.Bytes())
			if err != nil {
				t.Fatalf("Unable to format: %v\n%s", err, buf.Bytes())
			}
			want, err := format.Source([]byte(test.wanted))
			if err != nil {
				t.Fatalf("Unable to format: %v\n%s", err, test.wanted)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("Got:\n%s\nWant:\n%s\n", got, want)
			}
		})
	}
}
