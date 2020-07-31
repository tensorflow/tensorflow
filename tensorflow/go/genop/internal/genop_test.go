/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package internal

import (
	"bytes"
	"go/format"
	"testing"

	"github.com/golang/protobuf/proto"
	adpb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/api_def_go_proto"
	odpb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/op_def_go_proto"
)

// Creates an ApiDef based on opdef and applies overrides
// from apidefText (ApiDef text proto).
func GetAPIDef(t *testing.T, opdef *odpb.OpDef, apidefText string) *adpb.ApiDef {
	opdefList := &odpb.OpList{Op: []*odpb.OpDef{opdef}}
	apimap, err := newAPIDefMap(opdefList)
	if err != nil {
		t.Fatal(err)
	}
	err = apimap.Put(apidefText)
	if err != nil {
		t.Fatal(err)
	}
	apidef, err := apimap.Get(opdef.Name)
	if err != nil {
		t.Fatal(err)
	}
	return apidef
}

func TestGenerateOp(t *testing.T) {
	// TestGenerateOp validates the generated source code for an op.
	// The OpDef for the test cases are simplified forms of real ops.
	testdata := []struct {
		tag    string
		opdef  string
		apidef string
		wanted string
	}{
		{
			tag: "NoOp",
			opdef: `
name: "NoOp"
`,
			apidef: `
op: <
graph_op_name: "NoOp"
summary: "No. Op."
>
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
`,
			apidef: `
op: <
graph_op_name: "Add"
summary: "Returns x + y element-wise."
description: "Blah blah",
>
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
`,
			apidef: `
op: <
graph_op_name: "Cast"
summary: "Cast x of type SrcT to y of DstT."
>
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
  type: DT_STRING
>
output_arg: <
  name: "image"
  type: DT_UINT8
>
attr: <
  name: "channels"
  type: "int"
  default_value: <
    i: 0
  >
>
attr: <
  name: "fancy_upscaling"
  type: "bool"
  default_value: <
    b: true
  >
>
attr: <
  name: "acceptable_fraction"
  type: "float"
  default_value: <
    f: 1
  >
>
`,
			apidef: `
op: <
graph_op_name: "DecodeJpeg"
in_arg: <
  name: "contents"
  description: "0-D.  The JPEG-encoded image."
>
out_arg: <
  name: "image"
  description: "3-D with shape [height, width, channels]"
>
attr: <
  name: "channels"
  description: "Number of color channels for the decoded image."
>
attr: <
  name: "fancy_upscaling"
  description: "If true use a slower but nicer upscaling of the\nchroma planes (yuv420/422 only)."
>
attr: <
  name: "acceptable_fraction"
  description: "The minimum required fraction of lines before a truncated\ninput is accepted."
>
summary: "Decode a JPEG-encoded image to a uint8 tensor."
description: "Norna dorna fjord\nkajorna\nhahaha"
>
`,
			wanted: `
// DecodeJpegAttr is an optional argument to DecodeJpeg.
type DecodeJpegAttr func(optionalAttr)

// DecodeJpegChannels sets the optional channels attribute to value.
//
// value: Number of color channels for the decoded image.
// If not specified, defaults to 0
func DecodeJpegChannels(value int64) DecodeJpegAttr {
	return func(m optionalAttr) {
		m["channels"] = value
	}
}

// DecodeJpegFancyUpscaling sets the optional fancy_upscaling attribute to value.
//
// value: If true use a slower but nicer upscaling of the
// chroma planes (yuv420/422 only).
// If not specified, defaults to true
func DecodeJpegFancyUpscaling(value bool) DecodeJpegAttr {
	return func(m optionalAttr) {
		m["fancy_upscaling"] = value
	}
}

// DecodeJpegAcceptableFraction sets the optional acceptable_fraction attribute to value.
//
// value: The minimum required fraction of lines before a truncated
// input is accepted.
// If not specified, defaults to 1
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
`,
			apidef: `
op: <
graph_op_name: "TwoOutputs"
summary: "Op that produces multiple outputs"
>
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
`,
			apidef: `
op: <
graph_op_name: "ShapeN"
summary: "Returns shape of tensors."
description: "Some description here."
>
`,
			wanted: `
// ShapeNAttr is an optional argument to ShapeN.
type ShapeNAttr func(optionalAttr)

// ShapeNOutType sets the optional out_type attribute to value.
// If not specified, defaults to DT_INT32
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
		{
			tag: "ApiDefOverrides",
			opdef: `
name: "TestOp"
input_arg: <
  name: "a"
  type: DT_STRING
>
input_arg: <
  name: "b"
  type: DT_STRING
>
output_arg: <
  name: "c"
  type: DT_UINT8
>
attr: <
  name: "d"
  type: "int"
  default_value: <
    i: 0
  >
>
`,
			apidef: `
op: <
graph_op_name: "TestOp"
in_arg: <
  name: "a"
  rename_to: "aa"
  description: "Description for aa."
>
in_arg: <
  name: "b"
  rename_to: "bb"
  description: "Description for bb."
>
arg_order: "b"
arg_order: "a"
out_arg: <
  name: "c"
  rename_to: "cc"
  description: "Description for cc."
>
attr: <
  name: "d"
  rename_to: "dd"
  description: "Description for dd."
>
summary: "Summary for TestOp."
description: "Description for TestOp."
>
`,
			wanted: `
// TestOpAttr is an optional argument to TestOp.
type TestOpAttr func(optionalAttr)

// TestOpDd sets the optional dd attribute to value.
//
// value: Description for dd.
// If not specified, defaults to 0
func TestOpDd(value int64) TestOpAttr {
	return func(m optionalAttr) {
		m["d"] = value
	}
}

// Summary for TestOp.
//
// Description for TestOp.
//
// Arguments:
//	bb: Description for bb.
//	aa: Description for aa.
//
// Returns Description for cc.
func TestOp(scope *Scope, bb tf.Output, aa tf.Output, optional ...TestOpAttr) (cc tf.Output) {
	if scope.Err() != nil {
		return
	}
	attrs := map[string]interface{}{}
	for _, a := range optional {
		a(attrs)
	}
	opspec := tf.OpSpec{
		Type: "TestOp",
		Input: []tf.Input{
			aa, bb,
		},
		Attrs: attrs,
	}
	op := scope.AddOperation(opspec)
	return op.Output(0)
}
`,
		},
		{
			tag: "SampleDistortedBoundingBox",
			opdef: `
name: "SampleDistortedBoundingBox"
input_arg {
	name: "image_size"
	type_attr: "T"
}
input_arg {
	name: "bounding_boxes"
	type: DT_FLOAT
}
output_arg {
	name: "begin"
	type_attr: "T"
}
output_arg {
	name: "size"
	type_attr: "T"
}
output_arg {
	name: "bboxes"
	type: DT_FLOAT
}
attr {
	name: "T"
	type: "type"
	allowed_values {
		list {
			type: DT_UINT8
			type: DT_INT8
			type: DT_INT16
			type: DT_INT32
			type: DT_INT64
		}
	}
}
attr {
	name: "seed"
	type: "int"
	default_value {
		i: 0
	}
}
attr {
	name: "seed2"
	type: "int"
	default_value {
		i: 0
	}
}
attr {
	name: "min_object_covered"
	type: "float"
	default_value {
		f: 0.1
	}
}
attr {
	name: "aspect_ratio_range"
	type: "list(float)"
	default_value {
		list {
			f: 0.75
			f: 1.33
		}
	}
}
attr {
	name: "area_range"
	type: "list(float)"
	default_value {
		list {
			f: 0.05
			f: 1
		}
	}
}
attr {
	name: "max_attempts"
	type: "int"
	default_value {
		i: 100
	}
}
attr {
	name: "use_image_if_no_bounding_boxes"
	type: "bool"
	default_value {
		b: false
	}
}
is_stateful: true
`,
			apidef: `
op {
  graph_op_name: "SampleDistortedBoundingBox"
  in_arg {
    name: "image_size"
    description: "Blah blah"
  }
  in_arg {
    name: "bounding_boxes"
    description: "Blah blah"
  }
  out_arg {
    name: "begin"
    description: "Blah blah"
  }
  out_arg {
    name: "size"
    description: "Blah blah"
  }
  out_arg {
    name: "bboxes"
    description: "Blah blah"
  }
  attr {
    name: "seed"
    description: "Blah blah"
  }
  attr {
    name: "seed2"
    description: "Blah blah"
  }
  attr {
    name: "min_object_covered"
    description: "Blah blah"
  }
  attr {
    name: "aspect_ratio_range"
    description: "Blah blah"
  }
  attr {
    name: "area_range"
    description: "Blah blah"
  }
  attr {
    name: "max_attempts"
    description: "Blah blah"
  }
  attr {
    name: "use_image_if_no_bounding_boxes"
    description: "Blah blah"
  }
  summary: "Generate a single randomly distorted bounding box for an image."
	description: "Blah blah"
}
`,
			wanted: `
// SampleDistortedBoundingBoxAttr is an optional argument to SampleDistortedBoundingBox.
type SampleDistortedBoundingBoxAttr func(optionalAttr)

// SampleDistortedBoundingBoxSeed sets the optional seed attribute to value.
//
// value: Blah blah
// If not specified, defaults to 0
func SampleDistortedBoundingBoxSeed(value int64) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["seed"] = value
	}
}

// SampleDistortedBoundingBoxSeed2 sets the optional seed2 attribute to value.
//
// value: Blah blah
// If not specified, defaults to 0
func SampleDistortedBoundingBoxSeed2(value int64) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["seed2"] = value
	}
}

// SampleDistortedBoundingBoxMinObjectCovered sets the optional min_object_covered attribute to value.
//
// value: Blah blah
// If not specified, defaults to 0.1
func SampleDistortedBoundingBoxMinObjectCovered(value float32) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["min_object_covered"] = value
	}
}

// SampleDistortedBoundingBoxAspectRatioRange sets the optional aspect_ratio_range attribute to value.
//
// value: Blah blah
// If not specified, defaults to <f:0.75 f:1.33 >
func SampleDistortedBoundingBoxAspectRatioRange(value []float32) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["aspect_ratio_range"] = value
	}
}

// SampleDistortedBoundingBoxAreaRange sets the optional area_range attribute to value.
//
// value: Blah blah
// If not specified, defaults to <f:0.05 f:1 >
func SampleDistortedBoundingBoxAreaRange(value []float32) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["area_range"] = value
	}
}

// SampleDistortedBoundingBoxMaxAttempts sets the optional max_attempts attribute to value.
//
// value: Blah blah
// If not specified, defaults to 100
func SampleDistortedBoundingBoxMaxAttempts(value int64) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["max_attempts"] = value
	}
}

// SampleDistortedBoundingBoxUseImageIfNoBoundingBoxes sets the optional use_image_if_no_bounding_boxes attribute to value.
//
// value: Blah blah
// If not specified, defaults to false
func SampleDistortedBoundingBoxUseImageIfNoBoundingBoxes(value bool) SampleDistortedBoundingBoxAttr {
	return func(m optionalAttr) {
		m["use_image_if_no_bounding_boxes"] = value
	}
}

// Generate a single randomly distorted bounding box for an image.
//
// Blah blah
//
// Arguments:
//	image_size: Blah blah
//	bounding_boxes: Blah blah
//
// Returns:
//	begin: Blah blah
//	size: Blah blah
//	bboxes: Blah blah
func SampleDistortedBoundingBox(scope *Scope, image_size tf.Output, bounding_boxes tf.Output, optional ...SampleDistortedBoundingBoxAttr) (begin tf.Output, size tf.Output, bboxes tf.Output) {
	if scope.Err() != nil {
		return
	}
	attrs := map[string]interface{}{}
	for _, a := range optional {
		a(attrs)
	}
	opspec := tf.OpSpec{
		Type: "SampleDistortedBoundingBox",
		Input: []tf.Input{
			image_size, bounding_boxes,
		},
		Attrs: attrs,
	}
	op := scope.AddOperation(opspec)
	return op.Output(0), op.Output(1), op.Output(2)
}
`,
		},
	}

	for _, test := range testdata {
		t.Run(test.tag, func(t *testing.T) {
			var opdef odpb.OpDef
			var apidef *adpb.ApiDef
			var buf bytes.Buffer
			if err := proto.UnmarshalText(test.opdef, &opdef); err != nil {
				t.Fatal(err)
			}
			apidef = GetAPIDef(t, &opdef, test.apidef)
			if err := generateFunctionForOp(&buf, &opdef, apidef); err != nil {
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
