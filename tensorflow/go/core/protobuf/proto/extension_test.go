// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoimpl"
	"google.golang.org/protobuf/testing/protocmp"

	legacy1pb "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20160225_2fc053c5"
	testpb "google.golang.org/protobuf/internal/testprotos/test"
	test3pb "google.golang.org/protobuf/internal/testprotos/test3"
	testeditionspb "google.golang.org/protobuf/internal/testprotos/testeditions"
	descpb "google.golang.org/protobuf/types/descriptorpb"
)

func TestExtensionFuncs(t *testing.T) {
	for _, test := range []struct {
		message     proto.Message
		ext         protoreflect.ExtensionType
		wantDefault interface{}
		value       interface{}
	}{
		{
			message:     &testpb.TestAllExtensions{},
			ext:         testpb.E_OptionalInt32,
			wantDefault: int32(0),
			value:       int32(1),
		},
		{
			message:     &testpb.TestAllExtensions{},
			ext:         testpb.E_RepeatedString,
			wantDefault: ([]string)(nil),
			value:       []string{"a", "b", "c"},
		},
		{
			message:     &testeditionspb.TestAllExtensions{},
			ext:         testeditionspb.E_OptionalInt32,
			wantDefault: int32(0),
			value:       int32(1),
		},
		{
			message:     &testeditionspb.TestAllExtensions{},
			ext:         testeditionspb.E_RepeatedString,
			wantDefault: ([]string)(nil),
			value:       []string{"a", "b", "c"},
		},
		{
			message:     protoimpl.X.MessageOf(&legacy1pb.Message{}).Interface(),
			ext:         legacy1pb.E_Message_ExtensionOptionalBool,
			wantDefault: false,
			value:       true,
		},
	} {
		desc := fmt.Sprintf("Extension %v, value %v", test.ext.TypeDescriptor().FullName(), test.value)
		if proto.HasExtension(test.message, test.ext) {
			t.Errorf("%v:\nbefore setting extension HasExtension(...) = true, want false", desc)
		}
		got := proto.GetExtension(test.message, test.ext)
		if d := cmp.Diff(test.wantDefault, got); d != "" {
			t.Errorf("%v:\nbefore setting extension GetExtension(...) returns unexpected value (-want,+got):\n%v", desc, d)
		}
		proto.SetExtension(test.message, test.ext, test.value)
		if !proto.HasExtension(test.message, test.ext) {
			t.Errorf("%v:\nafter setting extension HasExtension(...) = false, want true", desc)
		}
		got = proto.GetExtension(test.message, test.ext)
		if d := cmp.Diff(test.value, got); d != "" {
			t.Errorf("%v:\nafter setting extension GetExtension(...) returns unexpected value (-want,+got):\n%v", desc, d)
		}
		proto.ClearExtension(test.message, test.ext)
		if proto.HasExtension(test.message, test.ext) {
			t.Errorf("%v:\nafter clearing extension HasExtension(...) = true, want false", desc)
		}
	}
}

func TestIsValid(t *testing.T) {
	tests := []struct {
		xt   protoreflect.ExtensionType
		vi   interface{}
		want bool
	}{
		{testpb.E_OptionalBool, nil, false},
		{testpb.E_OptionalBool, bool(true), true},
		{testpb.E_OptionalBool, new(bool), false},
		{testpb.E_OptionalInt32, nil, false},
		{testpb.E_OptionalInt32, int32(0), true},
		{testpb.E_OptionalInt32, new(int32), false},
		{testpb.E_OptionalInt64, nil, false},
		{testpb.E_OptionalInt64, int64(0), true},
		{testpb.E_OptionalInt64, new(int64), false},
		{testpb.E_OptionalUint32, nil, false},
		{testpb.E_OptionalUint32, uint32(0), true},
		{testpb.E_OptionalUint32, new(uint32), false},
		{testpb.E_OptionalUint64, nil, false},
		{testpb.E_OptionalUint64, uint64(0), true},
		{testpb.E_OptionalUint64, new(uint64), false},
		{testpb.E_OptionalFloat, nil, false},
		{testpb.E_OptionalFloat, float32(0), true},
		{testpb.E_OptionalFloat, new(float32), false},
		{testpb.E_OptionalDouble, nil, false},
		{testpb.E_OptionalDouble, float64(0), true},
		{testpb.E_OptionalDouble, new(float32), false},
		{testpb.E_OptionalString, nil, false},
		{testpb.E_OptionalString, string(""), true},
		{testpb.E_OptionalString, new(string), false},
		{testpb.E_OptionalNestedEnum, nil, false},
		{testpb.E_OptionalNestedEnum, testpb.TestAllTypes_BAZ, true},
		{testpb.E_OptionalNestedEnum, testpb.TestAllTypes_BAZ.Enum(), false},
		{testpb.E_OptionalNestedMessage, nil, false},
		{testpb.E_OptionalNestedMessage, (*testpb.TestAllExtensions_NestedMessage)(nil), true},
		{testpb.E_OptionalNestedMessage, new(testpb.TestAllExtensions_NestedMessage), true},
		{testpb.E_OptionalNestedMessage, new(testpb.TestAllExtensions), false},
		{testpb.E_RepeatedBool, nil, false},
		{testpb.E_RepeatedBool, []bool(nil), true},
		{testpb.E_RepeatedBool, []bool{}, true},
		{testpb.E_RepeatedBool, []bool{false}, true},
		{testpb.E_RepeatedBool, []*bool{}, false},
		{testpb.E_RepeatedInt32, nil, false},
		{testpb.E_RepeatedInt32, []int32(nil), true},
		{testpb.E_RepeatedInt32, []int32{}, true},
		{testpb.E_RepeatedInt32, []int32{0}, true},
		{testpb.E_RepeatedInt32, []*int32{}, false},
		{testpb.E_RepeatedInt64, nil, false},
		{testpb.E_RepeatedInt64, []int64(nil), true},
		{testpb.E_RepeatedInt64, []int64{}, true},
		{testpb.E_RepeatedInt64, []int64{0}, true},
		{testpb.E_RepeatedInt64, []*int64{}, false},
		{testpb.E_RepeatedUint32, nil, false},
		{testpb.E_RepeatedUint32, []uint32(nil), true},
		{testpb.E_RepeatedUint32, []uint32{}, true},
		{testpb.E_RepeatedUint32, []uint32{0}, true},
		{testpb.E_RepeatedUint32, []*uint32{}, false},
		{testpb.E_RepeatedUint64, nil, false},
		{testpb.E_RepeatedUint64, []uint64(nil), true},
		{testpb.E_RepeatedUint64, []uint64{}, true},
		{testpb.E_RepeatedUint64, []uint64{0}, true},
		{testpb.E_RepeatedUint64, []*uint64{}, false},
		{testpb.E_RepeatedFloat, nil, false},
		{testpb.E_RepeatedFloat, []float32(nil), true},
		{testpb.E_RepeatedFloat, []float32{}, true},
		{testpb.E_RepeatedFloat, []float32{0}, true},
		{testpb.E_RepeatedFloat, []*float32{}, false},
		{testpb.E_RepeatedDouble, nil, false},
		{testpb.E_RepeatedDouble, []float64(nil), true},
		{testpb.E_RepeatedDouble, []float64{}, true},
		{testpb.E_RepeatedDouble, []float64{0}, true},
		{testpb.E_RepeatedDouble, []*float64{}, false},
		{testpb.E_RepeatedString, nil, false},
		{testpb.E_RepeatedString, []string(nil), true},
		{testpb.E_RepeatedString, []string{}, true},
		{testpb.E_RepeatedString, []string{""}, true},
		{testpb.E_RepeatedString, []*string{}, false},
		{testpb.E_RepeatedNestedEnum, nil, false},
		{testpb.E_RepeatedNestedEnum, []testpb.TestAllTypes_NestedEnum(nil), true},
		{testpb.E_RepeatedNestedEnum, []testpb.TestAllTypes_NestedEnum{}, true},
		{testpb.E_RepeatedNestedEnum, []testpb.TestAllTypes_NestedEnum{0}, true},
		{testpb.E_RepeatedNestedEnum, []*testpb.TestAllTypes_NestedEnum{}, false},
		{testpb.E_RepeatedNestedMessage, nil, false},
		{testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage(nil), true},
		{testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage{}, true},
		{testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage{{}}, true},
		{testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions{}, false},
	}

	for _, tt := range tests {
		// Check the results of IsValidInterface.
		got := tt.xt.IsValidInterface(tt.vi)
		if got != tt.want {
			t.Errorf("%v.IsValidInterface() = %v, want %v", tt.xt.TypeDescriptor().FullName(), got, tt.want)
		}
		if !got {
			continue
		}

		// Set the extension value and verify the results of Has.
		wantHas := true
		pv := tt.xt.ValueOf(tt.vi)
		switch v := pv.Interface().(type) {
		case protoreflect.List:
			wantHas = v.Len() > 0
		case protoreflect.Message:
			wantHas = v.IsValid()
		}
		m := &testpb.TestAllExtensions{}
		proto.SetExtension(m, tt.xt, tt.vi)
		gotHas := proto.HasExtension(m, tt.xt)
		if gotHas != wantHas {
			t.Errorf("HasExtension(%q) = %v, want %v", tt.xt.TypeDescriptor().FullName(), gotHas, wantHas)
		}

		// Check consistency of IsValidInterface and IsValidValue.
		got = tt.xt.IsValidValue(pv)
		if got != tt.want {
			t.Errorf("%v.IsValidValue() = %v, want %v", tt.xt.TypeDescriptor().FullName(), got, tt.want)
		}
		if !got {
			continue
		}

		// Use of reflect.DeepEqual is intentional.
		// We really do want to ensure that the memory layout is identical.
		vi := tt.xt.InterfaceOf(pv)
		if !reflect.DeepEqual(vi, tt.vi) {
			t.Errorf("InterfaceOf(ValueOf(...)) round-trip mismatch: got %v, want %v", vi, tt.vi)
		}
	}
}

func TestExtensionRanger(t *testing.T) {
	tests := []struct {
		msg  proto.Message
		want map[protoreflect.ExtensionType]interface{}
	}{{
		msg: &testpb.TestAllExtensions{},
		want: map[protoreflect.ExtensionType]interface{}{
			testpb.E_OptionalInt32:         int32(5),
			testpb.E_OptionalString:        string("hello"),
			testpb.E_OptionalNestedMessage: &testpb.TestAllExtensions_NestedMessage{},
			testpb.E_OptionalNestedEnum:    testpb.TestAllTypes_BAZ,
			testpb.E_RepeatedFloat:         []float32{+32.32, -32.32},
			testpb.E_RepeatedNestedMessage: []*testpb.TestAllExtensions_NestedMessage{{}},
			testpb.E_RepeatedNestedEnum:    []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAZ},
		},
	}, {
		msg: &testeditionspb.TestAllExtensions{},
		want: map[protoreflect.ExtensionType]interface{}{
			testeditionspb.E_OptionalInt32:         int32(5),
			testeditionspb.E_OptionalString:        string("hello"),
			testeditionspb.E_OptionalNestedMessage: &testeditionspb.TestAllExtensions_NestedMessage{},
			testeditionspb.E_OptionalNestedEnum:    testeditionspb.TestAllTypes_BAZ,
			testeditionspb.E_RepeatedFloat:         []float32{+32.32, -32.32},
			testeditionspb.E_RepeatedNestedMessage: []*testeditionspb.TestAllExtensions_NestedMessage{{}},
			testeditionspb.E_RepeatedNestedEnum:    []testeditionspb.TestAllTypes_NestedEnum{testeditionspb.TestAllTypes_BAZ},
		},
	}, {
		msg: &descpb.MessageOptions{},
		want: map[protoreflect.ExtensionType]interface{}{
			test3pb.E_OptionalInt32:          int32(5),
			test3pb.E_OptionalString:         string("hello"),
			test3pb.E_OptionalForeignMessage: &test3pb.ForeignMessage{},
			test3pb.E_OptionalForeignEnum:    test3pb.ForeignEnum_FOREIGN_BAR,

			test3pb.E_OptionalOptionalInt32:          int32(5),
			test3pb.E_OptionalOptionalString:         string("hello"),
			test3pb.E_OptionalOptionalForeignMessage: &test3pb.ForeignMessage{},
			test3pb.E_OptionalOptionalForeignEnum:    test3pb.ForeignEnum_FOREIGN_BAR,
		},
	}}

	for _, tt := range tests {
		for xt, v := range tt.want {
			proto.SetExtension(tt.msg, xt, v)
		}

		got := make(map[protoreflect.ExtensionType]interface{})
		proto.RangeExtensions(tt.msg, func(xt protoreflect.ExtensionType, v interface{}) bool {
			got[xt] = v
			return true
		})

		if diff := cmp.Diff(tt.want, got, protocmp.Transform()); diff != "" {
			t.Errorf("proto.RangeExtensions mismatch (-want +got):\n%s", diff)
		}
	}
}

func TestExtensionGetRace(t *testing.T) {
	// Concurrently fetch an extension value while marshaling the message containing it.
	// Create the message with proto.Unmarshal to give lazy extension decoding (if present)
	// a chance to occur.
	want := int32(42)
	m1 := &testpb.TestAllExtensions{}
	proto.SetExtension(m1, testpb.E_OptionalNestedMessage, &testpb.TestAllExtensions_NestedMessage{A: proto.Int32(want)})
	b, err := proto.Marshal(m1)
	if err != nil {
		t.Fatal(err)
	}
	m := &testpb.TestAllExtensions{}
	if err := proto.Unmarshal(b, m); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := proto.Marshal(m); err != nil {
				t.Error(err)
			}
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := proto.GetExtension(m, testpb.E_OptionalNestedMessage).(*testpb.TestAllExtensions_NestedMessage).GetA()
			if got != want {
				t.Errorf("GetExtension(optional_nested_message).a = %v, want %v", got, want)
			}
		}()
	}
	wg.Wait()
}
