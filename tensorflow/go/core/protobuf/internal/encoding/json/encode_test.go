// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json_test

import (
	"math"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/encoding/json"
)

// Disable detrand to enable direct comparisons on outputs.
func init() { detrand.Disable() }

// splitLines is a cmpopts.Option for comparing strings with line breaks.
var splitLines = cmpopts.AcyclicTransformer("SplitLines", func(s string) []string {
	return strings.Split(s, "\n")
})

func TestEncoder(t *testing.T) {
	tests := []struct {
		desc          string
		write         func(*json.Encoder)
		wantOut       string
		wantOutIndent string
	}{
		{
			desc: "null",
			write: func(e *json.Encoder) {
				e.WriteNull()
			},
			wantOut: `null`,
		},
		{
			desc: "true",
			write: func(e *json.Encoder) {
				e.WriteBool(true)
			},
			wantOut: `true`,
		},
		{
			desc: "false",
			write: func(e *json.Encoder) {
				e.WriteBool(false)
			},
			wantOut: `false`,
		},
		{
			desc: "string",
			write: func(e *json.Encoder) {
				e.WriteString("hello world")
			},
			wantOut: `"hello world"`,
		},
		{
			desc: "string contains escaped characters",
			write: func(e *json.Encoder) {
				e.WriteString("\u0000\"\\/\b\f\n\r\t")
			},
			wantOut: `"\u0000\"\\/\b\f\n\r\t"`,
		},
		{
			desc: "float64",
			write: func(e *json.Encoder) {
				e.WriteFloat(1.0199999809265137, 64)
			},
			wantOut: `1.0199999809265137`,
		},
		{
			desc: "float64 max value",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.MaxFloat64, 64)
			},
			wantOut: `1.7976931348623157e+308`,
		},
		{
			desc: "float64 min value",
			write: func(e *json.Encoder) {
				e.WriteFloat(-math.MaxFloat64, 64)
			},
			wantOut: `-1.7976931348623157e+308`,
		},
		{
			desc: "float64 NaN",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.NaN(), 64)
			},
			wantOut: `"NaN"`,
		},
		{
			desc: "float64 Infinity",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.Inf(+1), 64)
			},
			wantOut: `"Infinity"`,
		},
		{
			desc: "float64 -Infinity",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.Inf(-1), 64)
			},
			wantOut: `"-Infinity"`,
		},
		{
			desc: "float64 negative zero",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.Copysign(0, -1), 64)
			},
			wantOut: `-0`,
		},
		{
			desc: "float32",
			write: func(e *json.Encoder) {
				e.WriteFloat(1.02, 32)
			},
			wantOut: `1.02`,
		},
		{
			desc: "float32 max value",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.MaxFloat32, 32)
			},
			wantOut: `3.4028235e+38`,
		},
		{
			desc: "float32 min value",
			write: func(e *json.Encoder) {
				e.WriteFloat(-math.MaxFloat32, 32)
			},
			wantOut: `-3.4028235e+38`,
		},
		{
			desc: "float32 negative zero",
			write: func(e *json.Encoder) {
				e.WriteFloat(math.Copysign(0, -1), 32)
			},
			wantOut: `-0`,
		},
		{
			desc: "int",
			write: func(e *json.Encoder) {
				e.WriteInt(-math.MaxInt64)
			},
			wantOut: `-9223372036854775807`,
		},
		{
			desc: "uint",
			write: func(e *json.Encoder) {
				e.WriteUint(math.MaxUint64)
			},
			wantOut: `18446744073709551615`,
		},
		{
			desc: "empty object",
			write: func(e *json.Encoder) {
				e.StartObject()
				e.EndObject()
			},
			wantOut: `{}`,
		},
		{
			desc: "empty array",
			write: func(e *json.Encoder) {
				e.StartArray()
				e.EndArray()
			},
			wantOut: `[]`,
		},
		{
			desc: "object with one member",
			write: func(e *json.Encoder) {
				e.StartObject()
				e.WriteName("hello")
				e.WriteString("world")
				e.EndObject()
			},
			wantOut: `{"hello":"world"}`,
			wantOutIndent: `{
	"hello": "world"
}`,
		},
		{
			desc: "array with one member",
			write: func(e *json.Encoder) {
				e.StartArray()
				e.WriteNull()
				e.EndArray()
			},
			wantOut: `[null]`,
			wantOutIndent: `[
	null
]`,
		},
		{
			desc: "simple object",
			write: func(e *json.Encoder) {
				e.StartObject()
				{
					e.WriteName("null")
					e.WriteNull()
				}
				{
					e.WriteName("bool")
					e.WriteBool(true)
				}
				{
					e.WriteName("string")
					e.WriteString("hello")
				}
				{
					e.WriteName("float")
					e.WriteFloat(6.28318, 64)
				}
				{
					e.WriteName("int")
					e.WriteInt(42)
				}
				{
					e.WriteName("uint")
					e.WriteUint(47)
				}
				e.EndObject()
			},
			wantOut: `{"null":null,"bool":true,"string":"hello","float":6.28318,"int":42,"uint":47}`,
			wantOutIndent: `{
	"null": null,
	"bool": true,
	"string": "hello",
	"float": 6.28318,
	"int": 42,
	"uint": 47
}`,
		},
		{
			desc: "simple array",
			write: func(e *json.Encoder) {
				e.StartArray()
				{
					e.WriteString("hello")
					e.WriteFloat(6.28318, 32)
					e.WriteInt(42)
					e.WriteUint(47)
					e.WriteBool(true)
					e.WriteNull()
				}
				e.EndArray()
			},
			wantOut: `["hello",6.28318,42,47,true,null]`,
			wantOutIndent: `[
	"hello",
	6.28318,
	42,
	47,
	true,
	null
]`,
		},
		{
			desc: "fancy object",
			write: func(e *json.Encoder) {
				e.StartObject()
				{
					e.WriteName("object0")
					e.StartObject()
					e.EndObject()
				}
				{
					e.WriteName("array0")
					e.StartArray()
					e.EndArray()
				}
				{
					e.WriteName("object1")
					e.StartObject()
					{
						e.WriteName("null")
						e.WriteNull()
					}
					{
						e.WriteName("object1-1")
						e.StartObject()
						{
							e.WriteName("bool")
							e.WriteBool(false)
						}
						{
							e.WriteName("float")
							e.WriteFloat(3.14159, 32)
						}
						e.EndObject()
					}
					e.EndObject()
				}
				{
					e.WriteName("array1")
					e.StartArray()
					{
						e.WriteNull()
						e.StartObject()
						e.EndObject()
						e.StartObject()
						{
							e.WriteName("hello")
							e.WriteString("world")
						}
						{
							e.WriteName("hola")
							e.WriteString("mundo")
						}
						e.EndObject()
						e.StartArray()
						{
							e.WriteUint(1)
							e.WriteUint(0)
							e.WriteUint(1)
						}
						e.EndArray()
					}
					e.EndArray()
				}
				e.EndObject()
			},
			wantOutIndent: `{
	"object0": {},
	"array0": [],
	"object1": {
		"null": null,
		"object1-1": {
			"bool": false,
			"float": 3.14159
		}
	},
	"array1": [
		null,
		{},
		{
			"hello": "world",
			"hola": "mundo"
		},
		[
			1,
			0,
			1
		]
	]
}`,
		}}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			if tc.wantOut != "" {
				enc, err := json.NewEncoder(nil, "")
				if err != nil {
					t.Fatalf("NewEncoder() returned error: %v", err)
				}
				tc.write(enc)
				got := string(enc.Bytes())
				if got != tc.wantOut {
					t.Errorf("%s:\n<got>:\n%v\n<want>\n%v\n", tc.desc, got, tc.wantOut)
				}
			}
			if tc.wantOutIndent != "" {
				enc, err := json.NewEncoder(nil, "\t")
				if err != nil {
					t.Fatalf("NewEncoder() returned error: %v", err)
				}
				tc.write(enc)
				got, want := string(enc.Bytes()), tc.wantOutIndent
				if got != want {
					t.Errorf("%s(indent):\n<got>:\n%v\n<want>\n%v\n<diff -want +got>\n%v\n",
						tc.desc, got, want, cmp.Diff(want, got, splitLines))
				}
			}
		})
	}
}

func TestWriteStringError(t *testing.T) {
	tests := []string{"abc\xff"}

	for _, in := range tests {
		t.Run(in, func(t *testing.T) {
			enc, err := json.NewEncoder(nil, "")
			if err != nil {
				t.Fatalf("NewEncoder() returned error: %v", err)
			}
			if err := enc.WriteString(in); err == nil {
				t.Errorf("WriteString(%v): got nil error, want error", in)
			}
		})
	}
}
