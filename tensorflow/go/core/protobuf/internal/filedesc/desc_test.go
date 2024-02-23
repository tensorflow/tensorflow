// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc_test

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

func init() {
	// Disable detrand to enable direct comparisons on outputs.
	detrand.Disable()
}

// TODO: Test protodesc.NewFile with imported files.

func TestFile(t *testing.T) {
	f1 := &descriptorpb.FileDescriptorProto{
		Syntax:  proto.String("proto2"),
		Name:    proto.String("path/to/file.proto"),
		Package: proto.String("test"),
		Options: &descriptorpb.FileOptions{Deprecated: proto.Bool(true)},
		MessageType: []*descriptorpb.DescriptorProto{{
			Name: proto.String("A"),
			Options: &descriptorpb.MessageOptions{
				Deprecated: proto.Bool(true),
			},
		}, {
			Name: proto.String("B"),
			Field: []*descriptorpb.FieldDescriptorProto{{
				Name:         proto.String("field_one"),
				Number:       proto.Int32(1),
				Label:        descriptorpb.FieldDescriptorProto_Label(protoreflect.Optional).Enum(),
				Type:         descriptorpb.FieldDescriptorProto_Type(protoreflect.StringKind).Enum(),
				DefaultValue: proto.String("hello, \"world!\"\n"),
				OneofIndex:   proto.Int32(0),
			}, {
				Name:         proto.String("field_two"),
				JsonName:     proto.String("Field2"),
				Number:       proto.Int32(2),
				Label:        descriptorpb.FieldDescriptorProto_Label(protoreflect.Optional).Enum(),
				Type:         descriptorpb.FieldDescriptorProto_Type(protoreflect.EnumKind).Enum(),
				DefaultValue: proto.String("BAR"),
				TypeName:     proto.String(".test.E1"),
				OneofIndex:   proto.Int32(1),
			}, {
				Name:       proto.String("field_three"),
				Number:     proto.Int32(3),
				Label:      descriptorpb.FieldDescriptorProto_Label(protoreflect.Optional).Enum(),
				Type:       descriptorpb.FieldDescriptorProto_Type(protoreflect.MessageKind).Enum(),
				TypeName:   proto.String(".test.C"),
				OneofIndex: proto.Int32(1),
			}, {
				Name:     proto.String("field_four"),
				JsonName: proto.String("Field4"),
				Number:   proto.Int32(4),
				Label:    descriptorpb.FieldDescriptorProto_Label(protoreflect.Repeated).Enum(),
				Type:     descriptorpb.FieldDescriptorProto_Type(protoreflect.MessageKind).Enum(),
				TypeName: proto.String(".test.B.FieldFourEntry"),
			}, {
				Name:    proto.String("field_five"),
				Number:  proto.Int32(5),
				Label:   descriptorpb.FieldDescriptorProto_Label(protoreflect.Repeated).Enum(),
				Type:    descriptorpb.FieldDescriptorProto_Type(protoreflect.Int32Kind).Enum(),
				Options: &descriptorpb.FieldOptions{Packed: proto.Bool(true)},
			}, {
				Name:   proto.String("field_six"),
				Number: proto.Int32(6),
				Label:  descriptorpb.FieldDescriptorProto_Label(protoreflect.Required).Enum(),
				Type:   descriptorpb.FieldDescriptorProto_Type(protoreflect.BytesKind).Enum(),
			}},
			OneofDecl: []*descriptorpb.OneofDescriptorProto{
				{
					Name: proto.String("O1"),
					Options: &descriptorpb.OneofOptions{
						UninterpretedOption: []*descriptorpb.UninterpretedOption{
							{StringValue: []byte("option")},
						},
					},
				},
				{Name: proto.String("O2")},
			},
			ReservedName: []string{"fizz", "buzz"},
			ReservedRange: []*descriptorpb.DescriptorProto_ReservedRange{
				{Start: proto.Int32(100), End: proto.Int32(200)},
				{Start: proto.Int32(300), End: proto.Int32(301)},
			},
			ExtensionRange: []*descriptorpb.DescriptorProto_ExtensionRange{
				{Start: proto.Int32(1000), End: proto.Int32(2000)},
				{Start: proto.Int32(3000), End: proto.Int32(3001), Options: new(descriptorpb.ExtensionRangeOptions)},
			},
			NestedType: []*descriptorpb.DescriptorProto{{
				Name: proto.String("FieldFourEntry"),
				Field: []*descriptorpb.FieldDescriptorProto{{
					Name:   proto.String("key"),
					Number: proto.Int32(1),
					Label:  descriptorpb.FieldDescriptorProto_Label(protoreflect.Optional).Enum(),
					Type:   descriptorpb.FieldDescriptorProto_Type(protoreflect.StringKind).Enum(),
				}, {
					Name:     proto.String("value"),
					Number:   proto.Int32(2),
					Label:    descriptorpb.FieldDescriptorProto_Label(protoreflect.Optional).Enum(),
					Type:     descriptorpb.FieldDescriptorProto_Type(protoreflect.MessageKind).Enum(),
					TypeName: proto.String(".test.B"),
				}},
				Options: &descriptorpb.MessageOptions{
					MapEntry: proto.Bool(true),
				},
			}},
		}, {
			Name: proto.String("C"),
			NestedType: []*descriptorpb.DescriptorProto{{
				Name: proto.String("A"),
				Field: []*descriptorpb.FieldDescriptorProto{{
					Name:         proto.String("F"),
					Number:       proto.Int32(1),
					Label:        descriptorpb.FieldDescriptorProto_Label(protoreflect.Required).Enum(),
					Type:         descriptorpb.FieldDescriptorProto_Type(protoreflect.BytesKind).Enum(),
					DefaultValue: proto.String(`dead\276\357`),
				}},
			}},
			EnumType: []*descriptorpb.EnumDescriptorProto{{
				Name: proto.String("E1"),
				Value: []*descriptorpb.EnumValueDescriptorProto{
					{Name: proto.String("FOO"), Number: proto.Int32(0)},
					{Name: proto.String("BAR"), Number: proto.Int32(1)},
				},
			}},
			Extension: []*descriptorpb.FieldDescriptorProto{{
				Name:     proto.String("X"),
				Number:   proto.Int32(1000),
				Label:    descriptorpb.FieldDescriptorProto_Label(protoreflect.Repeated).Enum(),
				Type:     descriptorpb.FieldDescriptorProto_Type(protoreflect.MessageKind).Enum(),
				TypeName: proto.String(".test.C"),
				Extendee: proto.String(".test.B"),
			}},
		}},
		EnumType: []*descriptorpb.EnumDescriptorProto{{
			Name:    proto.String("E1"),
			Options: &descriptorpb.EnumOptions{Deprecated: proto.Bool(true)},
			Value: []*descriptorpb.EnumValueDescriptorProto{
				{
					Name:    proto.String("FOO"),
					Number:  proto.Int32(0),
					Options: &descriptorpb.EnumValueOptions{Deprecated: proto.Bool(true)},
				},
				{Name: proto.String("BAR"), Number: proto.Int32(1)},
			},
			ReservedName: []string{"FIZZ", "BUZZ"},
			ReservedRange: []*descriptorpb.EnumDescriptorProto_EnumReservedRange{
				{Start: proto.Int32(10), End: proto.Int32(19)},
				{Start: proto.Int32(30), End: proto.Int32(30)},
			},
		}},
		Extension: []*descriptorpb.FieldDescriptorProto{{
			Name:     proto.String("X"),
			Number:   proto.Int32(1000),
			Label:    descriptorpb.FieldDescriptorProto_Label(protoreflect.Repeated).Enum(),
			Type:     descriptorpb.FieldDescriptorProto_Type(protoreflect.EnumKind).Enum(),
			Options:  &descriptorpb.FieldOptions{Packed: proto.Bool(true)},
			TypeName: proto.String(".test.E1"),
			Extendee: proto.String(".test.B"),
		}},
		Service: []*descriptorpb.ServiceDescriptorProto{{
			Name:    proto.String("S"),
			Options: &descriptorpb.ServiceOptions{Deprecated: proto.Bool(true)},
			Method: []*descriptorpb.MethodDescriptorProto{{
				Name:            proto.String("M"),
				InputType:       proto.String(".test.A"),
				OutputType:      proto.String(".test.C.A"),
				ClientStreaming: proto.Bool(true),
				ServerStreaming: proto.Bool(true),
				Options:         &descriptorpb.MethodOptions{Deprecated: proto.Bool(true)},
			}},
		}},
	}
	fd1, err := protodesc.NewFile(f1, nil)
	if err != nil {
		t.Fatalf("protodesc.NewFile() error: %v", err)
	}

	b, err := proto.Marshal(f1)
	if err != nil {
		t.Fatalf("proto.Marshal() error: %v", err)
	}
	fd2 := filedesc.Builder{RawDescriptor: b}.Build().File

	tests := []struct {
		name string
		desc protoreflect.FileDescriptor
	}{
		{"protodesc.NewFile", fd1},
		{"filedesc.Builder.Build", fd2},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			// Run sub-tests in parallel to induce potential races.
			for i := 0; i < 2; i++ {
				t.Run("Accessors", func(t *testing.T) { t.Parallel(); testFileAccessors(t, tt.desc) })
				t.Run("Format", func(t *testing.T) { t.Parallel(); testFileFormat(t, tt.desc) })
			}
		})
	}
}

func testFileAccessors(t *testing.T, fd protoreflect.FileDescriptor) {
	// Represent the descriptor as a map where each key is an accessor method
	// and the value is either the wanted tail value or another accessor map.
	type M = map[string]interface{}
	want := M{
		"Parent":        nil,
		"Index":         0,
		"Syntax":        protoreflect.Proto2,
		"Name":          protoreflect.Name("test"),
		"FullName":      protoreflect.FullName("test"),
		"Path":          "path/to/file.proto",
		"Package":       protoreflect.FullName("test"),
		"IsPlaceholder": false,
		"Options":       &descriptorpb.FileOptions{Deprecated: proto.Bool(true)},
		"Messages": M{
			"Len": 3,
			"Get:0": M{
				"Parent":        M{"FullName": protoreflect.FullName("test")},
				"Index":         0,
				"Syntax":        protoreflect.Proto2,
				"Name":          protoreflect.Name("A"),
				"FullName":      protoreflect.FullName("test.A"),
				"IsPlaceholder": false,
				"IsMapEntry":    false,
				"Options": &descriptorpb.MessageOptions{
					Deprecated: proto.Bool(true),
				},
				"Oneofs":          M{"Len": 0},
				"RequiredNumbers": M{"Len": 0},
				"ExtensionRanges": M{"Len": 0},
				"Messages":        M{"Len": 0},
				"Enums":           M{"Len": 0},
				"Extensions":      M{"Len": 0},
			},
			"ByName:B": M{
				"Name":  protoreflect.Name("B"),
				"Index": 1,
				"Fields": M{
					"Len":                  6,
					"ByJSONName:field_one": nil,
					"ByJSONName:fieldOne": M{
						"Name":              protoreflect.Name("field_one"),
						"Index":             0,
						"JSONName":          "fieldOne",
						"Default":           "hello, \"world!\"\n",
						"ContainingOneof":   M{"Name": protoreflect.Name("O1"), "IsPlaceholder": false},
						"ContainingMessage": M{"FullName": protoreflect.FullName("test.B")},
					},
					"ByJSONName:fieldTwo": nil,
					"ByJSONName:Field2": M{
						"Name":            protoreflect.Name("field_two"),
						"Index":           1,
						"HasJSONName":     true,
						"JSONName":        "Field2",
						"Default":         protoreflect.EnumNumber(1),
						"ContainingOneof": M{"Name": protoreflect.Name("O2"), "IsPlaceholder": false},
					},
					"ByName:fieldThree": nil,
					"ByName:field_three": M{
						"IsExtension":       false,
						"IsMap":             false,
						"MapKey":            nil,
						"MapValue":          nil,
						"Message":           M{"FullName": protoreflect.FullName("test.C"), "IsPlaceholder": false},
						"ContainingOneof":   M{"Name": protoreflect.Name("O2"), "IsPlaceholder": false},
						"ContainingMessage": M{"FullName": protoreflect.FullName("test.B")},
					},
					"ByNumber:12": nil,
					"ByNumber:4": M{
						"Cardinality": protoreflect.Repeated,
						"IsExtension": false,
						"IsList":      false,
						"IsMap":       true,
						"MapKey":      M{"Kind": protoreflect.StringKind},
						"MapValue":    M{"Kind": protoreflect.MessageKind, "Message": M{"FullName": protoreflect.FullName("test.B")}},
						"Default":     nil,
						"Message":     M{"FullName": protoreflect.FullName("test.B.FieldFourEntry"), "IsPlaceholder": false},
					},
					"ByNumber:5": M{
						"Cardinality": protoreflect.Repeated,
						"Kind":        protoreflect.Int32Kind,
						"IsPacked":    true,
						"IsList":      true,
						"IsMap":       false,
						"Default":     nil,
					},
					"ByNumber:6": M{
						"Cardinality":     protoreflect.Required,
						"Default":         []byte(nil),
						"ContainingOneof": nil,
					},
				},
				"Oneofs": M{
					"Len":       2,
					"ByName:O0": nil,
					"ByName:O1": M{
						"FullName": protoreflect.FullName("test.B.O1"),
						"Index":    0,
						"Options": &descriptorpb.OneofOptions{
							UninterpretedOption: []*descriptorpb.UninterpretedOption{
								{StringValue: []byte("option")},
							},
						},
						"Fields": M{
							"Len":   1,
							"Get:0": M{"FullName": protoreflect.FullName("test.B.field_one")},
						},
					},
					"Get:1": M{
						"FullName": protoreflect.FullName("test.B.O2"),
						"Index":    1,
						"Fields": M{
							"Len":              2,
							"ByName:field_two": M{"Name": protoreflect.Name("field_two")},
							"Get:1":            M{"Name": protoreflect.Name("field_three")},
						},
					},
				},
				"ReservedNames": M{
					"Len":         2,
					"Get:0":       protoreflect.Name("fizz"),
					"Has:buzz":    true,
					"Has:noexist": false,
				},
				"ReservedRanges": M{
					"Len":     2,
					"Get:0":   [2]protoreflect.FieldNumber{100, 200},
					"Has:99":  false,
					"Has:100": true,
					"Has:150": true,
					"Has:199": true,
					"Has:200": false,
					"Has:300": true,
					"Has:301": false,
				},
				"RequiredNumbers": M{
					"Len":   1,
					"Get:0": protoreflect.FieldNumber(6),
					"Has:1": false,
					"Has:6": true,
				},
				"ExtensionRanges": M{
					"Len":      2,
					"Get:0":    [2]protoreflect.FieldNumber{1000, 2000},
					"Has:999":  false,
					"Has:1000": true,
					"Has:1500": true,
					"Has:1999": true,
					"Has:2000": false,
					"Has:3000": true,
					"Has:3001": false,
				},
				"ExtensionRangeOptions:0": (*descriptorpb.ExtensionRangeOptions)(nil),
				"ExtensionRangeOptions:1": new(descriptorpb.ExtensionRangeOptions),
				"Messages": M{
					"Get:0": M{
						"Fields": M{
							"Len": 2,
							"ByNumber:1": M{
								"Parent":            M{"FullName": protoreflect.FullName("test.B.FieldFourEntry")},
								"Index":             0,
								"Name":              protoreflect.Name("key"),
								"FullName":          protoreflect.FullName("test.B.FieldFourEntry.key"),
								"Number":            protoreflect.FieldNumber(1),
								"Cardinality":       protoreflect.Optional,
								"Kind":              protoreflect.StringKind,
								"Options":           (*descriptorpb.FieldOptions)(nil),
								"HasJSONName":       false,
								"JSONName":          "key",
								"IsPacked":          false,
								"IsList":            false,
								"IsMap":             false,
								"IsExtension":       false,
								"IsWeak":            false,
								"Default":           "",
								"ContainingOneof":   nil,
								"ContainingMessage": M{"FullName": protoreflect.FullName("test.B.FieldFourEntry")},
								"Message":           nil,
								"Enum":              nil,
							},
							"ByNumber:2": M{
								"Parent":            M{"FullName": protoreflect.FullName("test.B.FieldFourEntry")},
								"Index":             1,
								"Name":              protoreflect.Name("value"),
								"FullName":          protoreflect.FullName("test.B.FieldFourEntry.value"),
								"Number":            protoreflect.FieldNumber(2),
								"Cardinality":       protoreflect.Optional,
								"Kind":              protoreflect.MessageKind,
								"JSONName":          "value",
								"IsPacked":          false,
								"IsList":            false,
								"IsMap":             false,
								"IsExtension":       false,
								"IsWeak":            false,
								"Default":           nil,
								"ContainingOneof":   nil,
								"ContainingMessage": M{"FullName": protoreflect.FullName("test.B.FieldFourEntry")},
								"Message":           M{"FullName": protoreflect.FullName("test.B"), "IsPlaceholder": false},
								"Enum":              nil,
							},
							"ByNumber:3": nil,
						},
					},
				},
			},
			"Get:2": M{
				"Name":  protoreflect.Name("C"),
				"Index": 2,
				"Messages": M{
					"Len":   1,
					"Get:0": M{"FullName": protoreflect.FullName("test.C.A")},
				},
				"Enums": M{
					"Len":   1,
					"Get:0": M{"FullName": protoreflect.FullName("test.C.E1")},
				},
				"Extensions": M{
					"Len":   1,
					"Get:0": M{"FullName": protoreflect.FullName("test.C.X")},
				},
			},
		},
		"Enums": M{
			"Len": 1,
			"Get:0": M{
				"Name":    protoreflect.Name("E1"),
				"Options": &descriptorpb.EnumOptions{Deprecated: proto.Bool(true)},
				"Values": M{
					"Len":        2,
					"ByName:Foo": nil,
					"ByName:FOO": M{
						"FullName": protoreflect.FullName("test.FOO"),
						"Options":  &descriptorpb.EnumValueOptions{Deprecated: proto.Bool(true)},
					},
					"ByNumber:2": nil,
					"ByNumber:1": M{"FullName": protoreflect.FullName("test.BAR")},
				},
				"ReservedNames": M{
					"Len":         2,
					"Get:0":       protoreflect.Name("FIZZ"),
					"Has:BUZZ":    true,
					"Has:NOEXIST": false,
				},
				"ReservedRanges": M{
					"Len":    2,
					"Get:0":  [2]protoreflect.EnumNumber{10, 19},
					"Has:9":  false,
					"Has:10": true,
					"Has:15": true,
					"Has:19": true,
					"Has:20": false,
					"Has:30": true,
					"Has:31": false,
				},
			},
		},
		"Extensions": M{
			"Len": 1,
			"ByName:X": M{
				"Name":              protoreflect.Name("X"),
				"Number":            protoreflect.FieldNumber(1000),
				"Cardinality":       protoreflect.Repeated,
				"Kind":              protoreflect.EnumKind,
				"IsExtension":       true,
				"IsPacked":          true,
				"IsList":            true,
				"IsMap":             false,
				"MapKey":            nil,
				"MapValue":          nil,
				"ContainingOneof":   nil,
				"ContainingMessage": M{"FullName": protoreflect.FullName("test.B"), "IsPlaceholder": false},
				"Enum":              M{"FullName": protoreflect.FullName("test.E1"), "IsPlaceholder": false},
				"Options":           &descriptorpb.FieldOptions{Packed: proto.Bool(true)},
			},
		},
		"Services": M{
			"Len":      1,
			"ByName:s": nil,
			"ByName:S": M{
				"Parent":   M{"FullName": protoreflect.FullName("test")},
				"Name":     protoreflect.Name("S"),
				"FullName": protoreflect.FullName("test.S"),
				"Options":  &descriptorpb.ServiceOptions{Deprecated: proto.Bool(true)},
				"Methods": M{
					"Len": 1,
					"Get:0": M{
						"Parent":            M{"FullName": protoreflect.FullName("test.S")},
						"Name":              protoreflect.Name("M"),
						"FullName":          protoreflect.FullName("test.S.M"),
						"Input":             M{"FullName": protoreflect.FullName("test.A"), "IsPlaceholder": false},
						"Output":            M{"FullName": protoreflect.FullName("test.C.A"), "IsPlaceholder": false},
						"IsStreamingClient": true,
						"IsStreamingServer": true,
						"Options":           &descriptorpb.MethodOptions{Deprecated: proto.Bool(true)},
					},
				},
			},
		},
	}
	checkAccessors(t, "", reflect.ValueOf(fd), want)
}
func checkAccessors(t *testing.T, p string, rv reflect.Value, want map[string]interface{}) {
	p0 := p
	defer func() {
		if ex := recover(); ex != nil {
			t.Errorf("panic at %v: %v", p, ex)
		}
	}()

	if rv.Interface() == nil {
		t.Errorf("%v is nil, want non-nil", p)
		return
	}
	for s, v := range want {
		// Call the accessor method.
		p = p0 + "." + s
		var rets []reflect.Value
		if i := strings.IndexByte(s, ':'); i >= 0 {
			// Accessor method takes in a single argument, which is encoded
			// after the accessor name, separated by a ':' delimiter.
			fnc := rv.MethodByName(s[:i])
			arg := reflect.New(fnc.Type().In(0)).Elem()
			s = s[i+len(":"):]
			switch arg.Kind() {
			case reflect.String:
				arg.SetString(s)
			case reflect.Int32, reflect.Int:
				n, _ := strconv.ParseInt(s, 0, 64)
				arg.SetInt(n)
			}
			rets = fnc.Call([]reflect.Value{arg})
		} else {
			rets = rv.MethodByName(s).Call(nil)
		}

		// Check that (val, ok) pattern is internally consistent.
		if len(rets) == 2 {
			if rets[0].IsNil() && rets[1].Bool() {
				t.Errorf("%v = (nil, true), want (nil, false)", p)
			}
			if !rets[0].IsNil() && !rets[1].Bool() {
				t.Errorf("%v = (non-nil, false), want (non-nil, true)", p)
			}
		}

		// Check that the accessor output matches.
		if want, ok := v.(map[string]interface{}); ok {
			checkAccessors(t, p, rets[0], want)
			continue
		}

		got := rets[0].Interface()
		if pv, ok := got.(protoreflect.Value); ok {
			got = pv.Interface()
		}

		// Compare with proto.Equal if possible.
		gotMsg, gotMsgOK := got.(proto.Message)
		wantMsg, wantMsgOK := v.(proto.Message)
		if gotMsgOK && wantMsgOK {
			gotNil := reflect.ValueOf(gotMsg).IsNil()
			wantNil := reflect.ValueOf(wantMsg).IsNil()
			switch {
			case !gotNil && wantNil:
				t.Errorf("%v = non-nil, want nil", p)
			case gotNil && !wantNil:
				t.Errorf("%v = nil, want non-nil", p)
			case !proto.Equal(gotMsg, wantMsg):
				t.Errorf("%v = %v, want %v", p, gotMsg, wantMsg)
			}
			continue
		}

		if want := v; !reflect.DeepEqual(got, want) {
			t.Errorf("%v = %T(%v), want %T(%v)", p, got, got, want, want)
		}
	}
}

func testFileFormat(t *testing.T, fd protoreflect.FileDescriptor) {
	const wantFileDescriptor = `FileDescriptor{
	Syntax:  proto2
	Path:    "path/to/file.proto"
	Package: test
	Messages: [{
		Name: A
	}, {
		Name: B
		Fields: [{
			Name:        field_one
			Number:      1
			Cardinality: optional
			Kind:        string
			JSONName:    "fieldOne"
			HasPresence: true
			HasDefault:  true
			Default:     "hello, \"world!\"\n"
			Oneof:       O1
		}, {
			Name:        field_two
			Number:      2
			Cardinality: optional
			Kind:        enum
			HasJSONName: true
			JSONName:    "Field2"
			HasPresence: true
			HasDefault:  true
			Default:     1
			Oneof:       O2
			Enum:        test.E1
		}, {
			Name:        field_three
			Number:      3
			Cardinality: optional
			Kind:        message
			JSONName:    "fieldThree"
			HasPresence: true
			Oneof:       O2
			Message:     test.C
		}, {
			Name:        field_four
			Number:      4
			Cardinality: repeated
			Kind:        message
			HasJSONName: true
			JSONName:    "Field4"
			IsMap:       true
			MapKey:      string
			MapValue:    test.B
		}, {
			Name:        field_five
			Number:      5
			Cardinality: repeated
			Kind:        int32
			JSONName:    "fieldFive"
			IsPacked:    true
			IsList:      true
		}, {
			Name:        field_six
			Number:      6
			Cardinality: required
			Kind:        bytes
			JSONName:    "fieldSix"
			HasPresence: true
		}]
		Oneofs: [{
			Name:   O1
			Fields: [field_one]
		}, {
			Name:   O2
			Fields: [field_two, field_three]
		}]
		ReservedNames:   [fizz, buzz]
		ReservedRanges:  [100:200, 300]
		RequiredNumbers: [6]
		ExtensionRanges: [1000:2000, 3000]
		Messages: [{
			Name:       FieldFourEntry
			IsMapEntry: true
			Fields: [{
				Name:        key
				Number:      1
				Cardinality: optional
				Kind:        string
				JSONName:    "key"
				HasPresence: true
			}, {
				Name:        value
				Number:      2
				Cardinality: optional
				Kind:        message
				JSONName:    "value"
				HasPresence: true
				Message:     test.B
			}]
		}]
	}, {
		Name: C
		Messages: [{
			Name: A
			Fields: [{
				Name:        F
				Number:      1
				Cardinality: required
				Kind:        bytes
				JSONName:    "F"
				HasPresence: true
				HasDefault:  true
				Default:     "dead\xbe\xef"
			}]
			RequiredNumbers: [1]
		}]
		Enums: [{
			Name: E1
			Values: [
				{Name: FOO}
				{Name: BAR, Number: 1}
			]
		}]
		Extensions: [{
			Name:        X
			Number:      1000
			Cardinality: repeated
			Kind:        message
			JSONName:    "[test.C.X]"
			IsExtension: true
			IsList:      true
			Extendee:    test.B
			Message:     test.C
		}]
	}]
	Enums: [{
		Name: E1
		Values: [
			{Name: FOO}
			{Name: BAR, Number: 1}
		]
		ReservedNames:  [FIZZ, BUZZ]
		ReservedRanges: [10:20, 30]
	}]
	Extensions: [{
		Name:        X
		Number:      1000
		Cardinality: repeated
		Kind:        enum
		JSONName:    "[test.X]"
		IsExtension: true
		IsPacked:    true
		IsList:      true
		Extendee:    test.B
		Enum:        test.E1
	}]
	Services: [{
		Name: S
		Methods: [{
			Name:              M
			Input:             test.A
			Output:            test.C.A
			IsStreamingClient: true
			IsStreamingServer: true
		}]
	}]
}`

	const wantEnums = `Enums{{
	Name: E1
	Values: [
		{Name: FOO}
		{Name: BAR, Number: 1}
	]
	ReservedNames:  [FIZZ, BUZZ]
	ReservedRanges: [10:20, 30]
}}`

	const wantExtensions = `Extensions{{
	Name:        X
	Number:      1000
	Cardinality: repeated
	Kind:        enum
	JSONName:    "[test.X]"
	IsExtension: true
	IsPacked:    true
	IsList:      true
	Extendee:    test.B
	Enum:        test.E1
}}`

	const wantImports = `FileImports{}`

	const wantReservedNames = "Names{fizz, buzz}"

	const wantReservedRanges = "FieldRanges{100:200, 300}"

	const wantServices = `Services{{
	Name: S
	Methods: [{
		Name:              M
		Input:             test.A
		Output:            test.C.A
		IsStreamingClient: true
		IsStreamingServer: true
	}]
}}`

	tests := []struct {
		path string
		fmt  string
		want string
		val  interface{}
	}{
		{"fd", "%v", compactMultiFormat(wantFileDescriptor), fd},
		{"fd", "%+v", wantFileDescriptor, fd},
		{"fd.Enums()", "%v", compactMultiFormat(wantEnums), fd.Enums()},
		{"fd.Enums()", "%+v", wantEnums, fd.Enums()},
		{"fd.Extensions()", "%v", compactMultiFormat(wantExtensions), fd.Extensions()},
		{"fd.Extensions()", "%+v", wantExtensions, fd.Extensions()},
		{"fd.Imports()", "%v", compactMultiFormat(wantImports), fd.Imports()},
		{"fd.Imports()", "%+v", wantImports, fd.Imports()},
		{"fd.Messages(B).ReservedNames()", "%v", compactMultiFormat(wantReservedNames), fd.Messages().ByName("B").ReservedNames()},
		{"fd.Messages(B).ReservedNames()", "%+v", wantReservedNames, fd.Messages().ByName("B").ReservedNames()},
		{"fd.Messages(B).ReservedRanges()", "%v", compactMultiFormat(wantReservedRanges), fd.Messages().ByName("B").ReservedRanges()},
		{"fd.Messages(B).ReservedRanges()", "%+v", wantReservedRanges, fd.Messages().ByName("B").ReservedRanges()},
		{"fd.Services()", "%v", compactMultiFormat(wantServices), fd.Services()},
		{"fd.Services()", "%+v", wantServices, fd.Services()},
	}
	for _, tt := range tests {
		got := fmt.Sprintf(tt.fmt, tt.val)
		if diff := cmp.Diff(got, tt.want); diff != "" {
			t.Errorf("fmt.Sprintf(%q, %s) mismatch (-got +want):\n%s", tt.fmt, tt.path, diff)
		}
	}
}

// compactMultiFormat returns the single line form of a multi line output.
func compactMultiFormat(s string) string {
	var b []byte
	for _, s := range strings.Split(s, "\n") {
		s = strings.TrimSpace(s)
		s = regexp.MustCompile(": +").ReplaceAllString(s, ": ")
		prevWord := len(b) > 0 && b[len(b)-1] != '[' && b[len(b)-1] != '{'
		nextWord := len(s) > 0 && s[0] != ']' && s[0] != '}'
		if prevWord && nextWord {
			b = append(b, ", "...)
		}
		b = append(b, s...)
	}
	return string(b)
}
