// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package descfmt_test

import (
	"reflect"
	"testing"

	"google.golang.org/protobuf/internal/descfmt"
	"google.golang.org/protobuf/reflect/protoreflect"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

// TestDescriptorAccessors tests that descriptorAccessors is up-to-date.
func TestDescriptorAccessors(t *testing.T) {
	ignore := map[string]bool{
		"ParentFile":    true,
		"Parent":        true,
		"Index":         true,
		"Syntax":        true,
		"Name":          true,
		"FullName":      true,
		"IsPlaceholder": true,
		"Options":       true,
		"ProtoInternal": true,
		"ProtoType":     true,

		"TextName":           true, // derived from other fields
		"HasOptionalKeyword": true, // captured by HasPresence
		"IsSynthetic":        true, // captured by HasPresence

		"SourceLocations":       true, // specific to FileDescriptor
		"ExtensionRangeOptions": true, // specific to MessageDescriptor
		"DefaultEnumValue":      true, // specific to FieldDescriptor
		"MapKey":                true, // specific to FieldDescriptor
		"MapValue":              true, // specific to FieldDescriptor
	}

	fileDesc := testpb.File_internal_testprotos_test_test_proto
	msgDesc := (&testpb.TestAllTypes{}).ProtoReflect().Descriptor()
	fields := msgDesc.Fields()
	fieldDesc := fields.ByName("oneof_uint32")
	oneofDesc := fieldDesc.ContainingOneof()
	enumDesc := fields.ByName("optional_nested_enum").Enum()
	enumValueDesc := fields.ByName("default_nested_enum").DefaultEnumValue()
	services := fileDesc.Services()
	serviceDesc := services.Get(0)
	methodDesc := serviceDesc.Methods().Get(0)
	rmsgDesc := (&testpb.TestNestedExtension{}).ProtoReflect().Descriptor()
	rfieldDesc := rmsgDesc.Extensions().Get(0)
	descriptors := map[reflect.Type][]protoreflect.Descriptor{
		reflect.TypeOf((*protoreflect.FileDescriptor)(nil)).Elem():      {fileDesc},
		reflect.TypeOf((*protoreflect.MessageDescriptor)(nil)).Elem():   {msgDesc},
		reflect.TypeOf((*protoreflect.FieldDescriptor)(nil)).Elem():     {fieldDesc, rfieldDesc},
		reflect.TypeOf((*protoreflect.OneofDescriptor)(nil)).Elem():     {oneofDesc},
		reflect.TypeOf((*protoreflect.EnumDescriptor)(nil)).Elem():      {enumDesc},
		reflect.TypeOf((*protoreflect.EnumValueDescriptor)(nil)).Elem(): {enumValueDesc},
		reflect.TypeOf((*protoreflect.ServiceDescriptor)(nil)).Elem():   {serviceDesc},
		reflect.TypeOf((*protoreflect.MethodDescriptor)(nil)).Elem():    {methodDesc},
	}
	for rt, descs := range descriptors {
		var m []string
		for _, desc := range descs {

			descfmt.InternalFormatDescOptForTesting(desc, true, false, func(name string) {
				m = append(m, name)
			})
		}

		got := map[string]bool{}
		for _, s := range m {
			got[s] = true
		}
		want := map[string]bool{}
		for i := 0; i < rt.NumMethod(); i++ {
			want[rt.Method(i).Name] = true
		}

		// Check if descriptorAccessors contains a non-existent accessor.
		// If this test fails, remove the accessor from descriptorAccessors.
		for s := range got {
			if !want[s] && !ignore[s] {
				t.Errorf("%v.%v does not exist", rt, s)
			}
		}

		// Check if there are new protoreflect interface methods that are not
		// handled by the formatter. If this fails, either add the method to
		// ignore or add them to descriptorAccessors.
		for s := range want {
			if !got[s] && !ignore[s] {
				t.Errorf("%v.%v is not called by formatter", rt, s)
			}
		}
	}
}
