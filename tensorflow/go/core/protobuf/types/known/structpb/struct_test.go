// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package structpb_test

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protocmp"

	spb "google.golang.org/protobuf/types/known/structpb"
)

var equateJSON = cmpopts.AcyclicTransformer("UnmarshalJSON", func(in []byte) (out interface{}) {
	if err := json.Unmarshal(in, &out); err != nil {
		return in
	}
	return out
})

func TestToStruct(t *testing.T) {
	tests := []struct {
		in      map[string]interface{}
		wantPB  *spb.Struct
		wantErr error
	}{{
		in:     nil,
		wantPB: new(spb.Struct),
	}, {
		in:     make(map[string]interface{}),
		wantPB: new(spb.Struct),
	}, {
		in: map[string]interface{}{
			"nil":     nil,
			"bool":    bool(false),
			"int":     int(-123),
			"int32":   int32(math.MinInt32),
			"int64":   int64(math.MinInt64),
			"uint":    uint(123),
			"uint32":  uint32(math.MaxInt32),
			"uint64":  uint64(math.MaxInt64),
			"float32": float32(123.456),
			"float64": float64(123.456),
			"string":  string("hello, world!"),
			"bytes":   []byte("\xde\xad\xbe\xef"),
			"map":     map[string]interface{}{"k1": "v1", "k2": "v2"},
			"slice":   []interface{}{"one", "two", "three"},
		},
		wantPB: &spb.Struct{Fields: map[string]*spb.Value{
			"nil":     spb.NewNullValue(),
			"bool":    spb.NewBoolValue(false),
			"int":     spb.NewNumberValue(float64(-123)),
			"int32":   spb.NewNumberValue(float64(math.MinInt32)),
			"int64":   spb.NewNumberValue(float64(math.MinInt64)),
			"uint":    spb.NewNumberValue(float64(123)),
			"uint32":  spb.NewNumberValue(float64(math.MaxInt32)),
			"uint64":  spb.NewNumberValue(float64(math.MaxInt64)),
			"float32": spb.NewNumberValue(float64(float32(123.456))),
			"float64": spb.NewNumberValue(float64(float64(123.456))),
			"string":  spb.NewStringValue("hello, world!"),
			"bytes":   spb.NewStringValue("3q2+7w=="),
			"map": spb.NewStructValue(&spb.Struct{Fields: map[string]*spb.Value{
				"k1": spb.NewStringValue("v1"),
				"k2": spb.NewStringValue("v2"),
			}}),
			"slice": spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
				spb.NewStringValue("one"),
				spb.NewStringValue("two"),
				spb.NewStringValue("three"),
			}}),
		}},
	}, {
		in:      map[string]interface{}{"\xde\xad\xbe\xef": "<invalid UTF-8>"},
		wantErr: cmpopts.AnyError,
	}, {
		in:      map[string]interface{}{"<invalid UTF-8>": "\xde\xad\xbe\xef"},
		wantErr: cmpopts.AnyError,
	}, {
		in:      map[string]interface{}{"key": protoreflect.Name("named string")},
		wantErr: cmpopts.AnyError,
	}}

	for _, tt := range tests {
		gotPB, gotErr := spb.NewStruct(tt.in)
		if diff := cmp.Diff(tt.wantPB, gotPB, protocmp.Transform()); diff != "" {
			t.Errorf("NewStruct(%v) output mismatch (-want +got):\n%s", tt.in, diff)
		}
		if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
			t.Errorf("NewStruct(%v) error mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestFromStruct(t *testing.T) {
	tests := []struct {
		in   *spb.Struct
		want map[string]interface{}
	}{{
		in:   nil,
		want: make(map[string]interface{}),
	}, {
		in:   new(spb.Struct),
		want: make(map[string]interface{}),
	}, {
		in:   &spb.Struct{Fields: make(map[string]*spb.Value)},
		want: make(map[string]interface{}),
	}, {
		in: &spb.Struct{Fields: map[string]*spb.Value{
			"nil":     spb.NewNullValue(),
			"bool":    spb.NewBoolValue(false),
			"int":     spb.NewNumberValue(float64(-123)),
			"int32":   spb.NewNumberValue(float64(math.MinInt32)),
			"int64":   spb.NewNumberValue(float64(math.MinInt64)),
			"uint":    spb.NewNumberValue(float64(123)),
			"uint32":  spb.NewNumberValue(float64(math.MaxInt32)),
			"uint64":  spb.NewNumberValue(float64(math.MaxInt64)),
			"float32": spb.NewNumberValue(float64(float32(123.456))),
			"float64": spb.NewNumberValue(float64(float64(123.456))),
			"string":  spb.NewStringValue("hello, world!"),
			"bytes":   spb.NewStringValue("3q2+7w=="),
			"map": spb.NewStructValue(&spb.Struct{Fields: map[string]*spb.Value{
				"k1": spb.NewStringValue("v1"),
				"k2": spb.NewStringValue("v2"),
			}}),
			"slice": spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
				spb.NewStringValue("one"),
				spb.NewStringValue("two"),
				spb.NewStringValue("three"),
			}}),
		}},
		want: map[string]interface{}{
			"nil":     nil,
			"bool":    bool(false),
			"int":     float64(-123),
			"int32":   float64(math.MinInt32),
			"int64":   float64(math.MinInt64),
			"uint":    float64(123),
			"uint32":  float64(math.MaxInt32),
			"uint64":  float64(math.MaxInt64),
			"float32": float64(float32(123.456)),
			"float64": float64(float64(123.456)),
			"string":  string("hello, world!"),
			"bytes":   string("3q2+7w=="),
			"map":     map[string]interface{}{"k1": "v1", "k2": "v2"},
			"slice":   []interface{}{"one", "two", "three"},
		},
	}}

	for _, tt := range tests {
		got := tt.in.AsMap()
		if diff := cmp.Diff(tt.want, got); diff != "" {
			t.Errorf("AsMap(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Errorf("Marshal error: %v", err)
		}
		wantJSON, err := tt.in.MarshalJSON()
		if err != nil {
			t.Errorf("Marshal error: %v", err)
		}
		if diff := cmp.Diff(wantJSON, gotJSON, equateJSON); diff != "" {
			t.Errorf("MarshalJSON(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestToListValue(t *testing.T) {
	tests := []struct {
		in      []interface{}
		wantPB  *spb.ListValue
		wantErr error
	}{{
		in:     nil,
		wantPB: new(spb.ListValue),
	}, {
		in:     make([]interface{}, 0),
		wantPB: new(spb.ListValue),
	}, {
		in: []interface{}{
			nil,
			bool(false),
			int(-123),
			int32(math.MinInt32),
			int64(math.MinInt64),
			uint(123),
			uint32(math.MaxInt32),
			uint64(math.MaxInt64),
			float32(123.456),
			float64(123.456),
			string("hello, world!"),
			[]byte("\xde\xad\xbe\xef"),
			map[string]interface{}{"k1": "v1", "k2": "v2"},
			[]interface{}{"one", "two", "three"},
		},
		wantPB: &spb.ListValue{Values: []*spb.Value{
			spb.NewNullValue(),
			spb.NewBoolValue(false),
			spb.NewNumberValue(float64(-123)),
			spb.NewNumberValue(float64(math.MinInt32)),
			spb.NewNumberValue(float64(math.MinInt64)),
			spb.NewNumberValue(float64(123)),
			spb.NewNumberValue(float64(math.MaxInt32)),
			spb.NewNumberValue(float64(math.MaxInt64)),
			spb.NewNumberValue(float64(float32(123.456))),
			spb.NewNumberValue(float64(float64(123.456))),
			spb.NewStringValue("hello, world!"),
			spb.NewStringValue("3q2+7w=="),
			spb.NewStructValue(&spb.Struct{Fields: map[string]*spb.Value{
				"k1": spb.NewStringValue("v1"),
				"k2": spb.NewStringValue("v2"),
			}}),
			spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
				spb.NewStringValue("one"),
				spb.NewStringValue("two"),
				spb.NewStringValue("three"),
			}}),
		}},
	}, {
		in:      []interface{}{"\xde\xad\xbe\xef"},
		wantErr: cmpopts.AnyError,
	}, {
		in:      []interface{}{protoreflect.Name("named string")},
		wantErr: cmpopts.AnyError,
	}}

	for _, tt := range tests {
		gotPB, gotErr := spb.NewList(tt.in)
		if diff := cmp.Diff(tt.wantPB, gotPB, protocmp.Transform()); diff != "" {
			t.Errorf("NewListValue(%v) output mismatch (-want +got):\n%s", tt.in, diff)
		}
		if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
			t.Errorf("NewListValue(%v) error mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestFromListValue(t *testing.T) {
	tests := []struct {
		in   *spb.ListValue
		want []interface{}
	}{{
		in:   nil,
		want: make([]interface{}, 0),
	}, {
		in:   new(spb.ListValue),
		want: make([]interface{}, 0),
	}, {
		in:   &spb.ListValue{Values: make([]*spb.Value, 0)},
		want: make([]interface{}, 0),
	}, {
		in: &spb.ListValue{Values: []*spb.Value{
			spb.NewNullValue(),
			spb.NewBoolValue(false),
			spb.NewNumberValue(float64(-123)),
			spb.NewNumberValue(float64(math.MinInt32)),
			spb.NewNumberValue(float64(math.MinInt64)),
			spb.NewNumberValue(float64(123)),
			spb.NewNumberValue(float64(math.MaxInt32)),
			spb.NewNumberValue(float64(math.MaxInt64)),
			spb.NewNumberValue(float64(float32(123.456))),
			spb.NewNumberValue(float64(float64(123.456))),
			spb.NewStringValue("hello, world!"),
			spb.NewStringValue("3q2+7w=="),
			spb.NewStructValue(&spb.Struct{Fields: map[string]*spb.Value{
				"k1": spb.NewStringValue("v1"),
				"k2": spb.NewStringValue("v2"),
			}}),
			spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
				spb.NewStringValue("one"),
				spb.NewStringValue("two"),
				spb.NewStringValue("three"),
			}}),
		}},
		want: []interface{}{
			nil,
			bool(false),
			float64(-123),
			float64(math.MinInt32),
			float64(math.MinInt64),
			float64(123),
			float64(math.MaxInt32),
			float64(math.MaxInt64),
			float64(float32(123.456)),
			float64(float64(123.456)),
			string("hello, world!"),
			string("3q2+7w=="),
			map[string]interface{}{"k1": "v1", "k2": "v2"},
			[]interface{}{"one", "two", "three"},
		},
	}}

	for _, tt := range tests {
		got := tt.in.AsSlice()
		if diff := cmp.Diff(tt.want, got); diff != "" {
			t.Errorf("AsSlice(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Errorf("Marshal error: %v", err)
		}
		wantJSON, err := tt.in.MarshalJSON()
		if err != nil {
			t.Errorf("Marshal error: %v", err)
		}
		if diff := cmp.Diff(wantJSON, gotJSON, equateJSON); diff != "" {
			t.Errorf("MarshalJSON(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestToValue(t *testing.T) {
	tests := []struct {
		in      interface{}
		wantPB  *spb.Value
		wantErr error
	}{{
		in:     nil,
		wantPB: spb.NewNullValue(),
	}, {
		in:     bool(false),
		wantPB: spb.NewBoolValue(false),
	}, {
		in:     int(-123),
		wantPB: spb.NewNumberValue(float64(-123)),
	}, {
		in:     int32(math.MinInt32),
		wantPB: spb.NewNumberValue(float64(math.MinInt32)),
	}, {
		in:     int64(math.MinInt64),
		wantPB: spb.NewNumberValue(float64(math.MinInt64)),
	}, {
		in:     uint(123),
		wantPB: spb.NewNumberValue(float64(123)),
	}, {
		in:     uint32(math.MaxInt32),
		wantPB: spb.NewNumberValue(float64(math.MaxInt32)),
	}, {
		in:     uint64(math.MaxInt64),
		wantPB: spb.NewNumberValue(float64(math.MaxInt64)),
	}, {
		in:     float32(123.456),
		wantPB: spb.NewNumberValue(float64(float32(123.456))),
	}, {
		in:     float64(123.456),
		wantPB: spb.NewNumberValue(float64(float64(123.456))),
	}, {
		in:     string("hello, world!"),
		wantPB: spb.NewStringValue("hello, world!"),
	}, {
		in:     []byte("\xde\xad\xbe\xef"),
		wantPB: spb.NewStringValue("3q2+7w=="),
	}, {
		in:     map[string]interface{}(nil),
		wantPB: spb.NewStructValue(nil),
	}, {
		in:     make(map[string]interface{}),
		wantPB: spb.NewStructValue(nil),
	}, {
		in: map[string]interface{}{"k1": "v1", "k2": "v2"},
		wantPB: spb.NewStructValue(&spb.Struct{Fields: map[string]*spb.Value{
			"k1": spb.NewStringValue("v1"),
			"k2": spb.NewStringValue("v2"),
		}}),
	}, {
		in:     []interface{}(nil),
		wantPB: spb.NewListValue(nil),
	}, {
		in:     make([]interface{}, 0),
		wantPB: spb.NewListValue(nil),
	}, {
		in: []interface{}{"one", "two", "three"},
		wantPB: spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
			spb.NewStringValue("one"),
			spb.NewStringValue("two"),
			spb.NewStringValue("three"),
		}}),
	}, {
		in:      "\xde\xad\xbe\xef",
		wantErr: cmpopts.AnyError,
	}, {
		in:      protoreflect.Name("named string"),
		wantErr: cmpopts.AnyError,
	}}

	for _, tt := range tests {
		gotPB, gotErr := spb.NewValue(tt.in)
		if diff := cmp.Diff(tt.wantPB, gotPB, protocmp.Transform()); diff != "" {
			t.Errorf("NewValue(%v) output mismatch (-want +got):\n%s", tt.in, diff)
		}
		if diff := cmp.Diff(tt.wantErr, gotErr, cmpopts.EquateErrors()); diff != "" {
			t.Errorf("NewValue(%v) error mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}

func TestFromValue(t *testing.T) {
	tests := []struct {
		in   *spb.Value
		want interface{}
	}{{
		in:   nil,
		want: nil,
	}, {
		in:   new(spb.Value),
		want: nil,
	}, {
		in:   &spb.Value{Kind: (*spb.Value_NullValue)(nil)},
		want: nil,
	}, {
		in:   spb.NewNullValue(),
		want: nil,
	}, {
		in:   &spb.Value{Kind: &spb.Value_NullValue{NullValue: math.MinInt32}},
		want: nil,
	}, {
		in:   &spb.Value{Kind: (*spb.Value_BoolValue)(nil)},
		want: nil,
	}, {
		in:   spb.NewBoolValue(false),
		want: bool(false),
	}, {
		in:   &spb.Value{Kind: (*spb.Value_NumberValue)(nil)},
		want: nil,
	}, {
		in:   spb.NewNumberValue(float64(math.MinInt32)),
		want: float64(math.MinInt32),
	}, {
		in:   spb.NewNumberValue(float64(math.MinInt64)),
		want: float64(math.MinInt64),
	}, {
		in:   spb.NewNumberValue(float64(123)),
		want: float64(123),
	}, {
		in:   spb.NewNumberValue(float64(math.MaxInt32)),
		want: float64(math.MaxInt32),
	}, {
		in:   spb.NewNumberValue(float64(math.MaxInt64)),
		want: float64(math.MaxInt64),
	}, {
		in:   spb.NewNumberValue(float64(float32(123.456))),
		want: float64(float32(123.456)),
	}, {
		in:   spb.NewNumberValue(float64(float64(123.456))),
		want: float64(float64(123.456)),
	}, {
		in:   spb.NewNumberValue(math.NaN()),
		want: string("NaN"),
	}, {
		in:   spb.NewNumberValue(math.Inf(-1)),
		want: string("-Infinity"),
	}, {
		in:   spb.NewNumberValue(math.Inf(+1)),
		want: string("Infinity"),
	}, {
		in:   &spb.Value{Kind: (*spb.Value_StringValue)(nil)},
		want: nil,
	}, {
		in:   spb.NewStringValue("hello, world!"),
		want: string("hello, world!"),
	}, {
		in:   spb.NewStringValue("3q2+7w=="),
		want: string("3q2+7w=="),
	}, {
		in:   &spb.Value{Kind: (*spb.Value_StructValue)(nil)},
		want: nil,
	}, {
		in:   &spb.Value{Kind: &spb.Value_StructValue{}},
		want: make(map[string]interface{}),
	}, {
		in: spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
			spb.NewStringValue("one"),
			spb.NewStringValue("two"),
			spb.NewStringValue("three"),
		}}),
		want: []interface{}{"one", "two", "three"},
	}, {
		in:   &spb.Value{Kind: (*spb.Value_ListValue)(nil)},
		want: nil,
	}, {
		in:   &spb.Value{Kind: &spb.Value_ListValue{}},
		want: make([]interface{}, 0),
	}, {
		in: spb.NewListValue(&spb.ListValue{Values: []*spb.Value{
			spb.NewStringValue("one"),
			spb.NewStringValue("two"),
			spb.NewStringValue("three"),
		}}),
		want: []interface{}{"one", "two", "three"},
	}}

	for _, tt := range tests {
		got := tt.in.AsInterface()
		if diff := cmp.Diff(tt.want, got); diff != "" {
			t.Errorf("AsInterface(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
		gotJSON, gotErr := json.Marshal(got)
		if gotErr != nil {
			t.Errorf("Marshal error: %v", gotErr)
		}
		wantJSON, wantErr := tt.in.MarshalJSON()
		if diff := cmp.Diff(wantJSON, gotJSON, equateJSON); diff != "" && wantErr == nil {
			t.Errorf("MarshalJSON(%v) mismatch (-want +got):\n%s", tt.in, diff)
		}
	}
}
