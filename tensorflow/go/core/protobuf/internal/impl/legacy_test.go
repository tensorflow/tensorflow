// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl_test

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/protobuf/encoding/prototext"
	pimpl "google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"

	proto2_20180125 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20180125_92554152"
	"google.golang.org/protobuf/types/descriptorpb"
)

type LegacyTestMessage struct {
	XXX_unrecognized       []byte
	XXX_InternalExtensions map[int32]pimpl.ExtensionField
}

func (*LegacyTestMessage) Reset()         {}
func (*LegacyTestMessage) String() string { return "" }
func (*LegacyTestMessage) ProtoMessage()  {}
func (*LegacyTestMessage) ExtensionRangeArray() []protoiface.ExtensionRangeV1 {
	return []protoiface.ExtensionRangeV1{{Start: 10, End: 20}, {Start: 40, End: 80}, {Start: 10000, End: 20000}}
}
func (*LegacyTestMessage) Descriptor() ([]byte, []int) { return legacyFD, []int{0} }

var legacyFD = func() []byte {
	b, _ := proto.Marshal(protodesc.ToFileDescriptorProto(mustMakeFileDesc(`
		name:   "legacy.proto"
		syntax: "proto2"
		message_type: [{
			name:            "LegacyTestMessage"
			extension_range: [{start:10 end:20}, {start:40 end:80}, {start:10000 end:20000}]
		}]
	`, nil)))
	return pimpl.Export{}.CompressGZIP(b)
}()

func init() {
	mt := pimpl.Export{}.MessageTypeOf((*LegacyTestMessage)(nil))
	protoregistry.GlobalFiles.RegisterFile(mt.Descriptor().ParentFile())
	protoregistry.GlobalTypes.RegisterMessage(mt)
}

func mustMakeExtensionType(fileDesc, extDesc string, t reflect.Type, r protodesc.Resolver) protoreflect.ExtensionType {
	s := fmt.Sprintf(`name:"test.proto" syntax:"proto2" %s extension:[{%s}]`, fileDesc, extDesc)
	xd := mustMakeFileDesc(s, r).Extensions().Get(0)
	xi := &pimpl.ExtensionInfo{}
	pimpl.InitExtensionInfo(xi, xd, t)
	return xi
}

func mustMakeFileDesc(s string, r protodesc.Resolver) protoreflect.FileDescriptor {
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(pb, r)
	if err != nil {
		panic(err)
	}
	return fd
}

var (
	testParentDesc    = pimpl.Export{}.MessageDescriptorOf((*LegacyTestMessage)(nil))
	testEnumV1Desc    = pimpl.Export{}.EnumDescriptorOf(proto2_20180125.Message_ChildEnum(0))
	testMessageV1Desc = pimpl.Export{}.MessageDescriptorOf((*proto2_20180125.Message_ChildMessage)(nil))
	testMessageV2Desc = enumMessagesType.Desc

	depReg = newFileRegistry(
		testParentDesc.ParentFile(),
		testEnumV1Desc.ParentFile(),
		testMessageV1Desc.ParentFile(),
		enumProto2Desc.ParentFile(),
		testMessageV2Desc.ParentFile(),
	)
	extensionTypes = []protoreflect.ExtensionType{
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_bool" number:10000 label:LABEL_OPTIONAL type:TYPE_BOOL default_value:"true" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(false), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_int32" number:10001 label:LABEL_OPTIONAL type:TYPE_INT32 default_value:"-12345" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(int32(0)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_uint32" number:10002 label:LABEL_OPTIONAL type:TYPE_UINT32 default_value:"3200" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(uint32(0)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_float" number:10003 label:LABEL_OPTIONAL type:TYPE_FLOAT default_value:"3.14159" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(float32(0)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_string" number:10004 label:LABEL_OPTIONAL type:TYPE_STRING default_value:"hello, \"world!\"\n" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(""), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"optional_bytes" number:10005 label:LABEL_OPTIONAL type:TYPE_BYTES default_value:"dead\\336\\255\\276\\357beef" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(([]byte)(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "proto2_20180125_92554152/test.proto"]`,
			`name:"optional_enum_v1" number:10006 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:".google.golang.org.proto2_20180125.Message.ChildEnum" default_value:"ALPHA" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(proto2_20180125.Message_ChildEnum(0)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "proto2_20180125_92554152/test.proto"]`,
			`name:"optional_message_v1" number:10007 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".google.golang.org.proto2_20180125.Message.ChildMessage" extendee:".LegacyTestMessage"`,
			reflect.TypeOf((*proto2_20180125.Message_ChildMessage)(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "enum2.proto"]`,
			`name:"optional_enum_v2" number:10008 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:".EnumProto2" default_value:"DEAD" extendee:".LegacyTestMessage"`,
			reflect.TypeOf(EnumProto2(0)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "enum-messages.proto"]`,
			`name:"optional_message_v2" number:10009 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".EnumMessages" extendee:".LegacyTestMessage"`,
			reflect.TypeOf((*EnumMessages)(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_bool" number:10010 label:LABEL_REPEATED type:TYPE_BOOL extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]bool(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_int32" number:10011 label:LABEL_REPEATED type:TYPE_INT32 extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]int32(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_uint32" number:10012 label:LABEL_REPEATED type:TYPE_UINT32 extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]uint32(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_float" number:10013 label:LABEL_REPEATED type:TYPE_FLOAT extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]float32(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_string" number:10014 label:LABEL_REPEATED type:TYPE_STRING extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]string(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:"legacy.proto"`,
			`name:"repeated_bytes" number:10015 label:LABEL_REPEATED type:TYPE_BYTES extendee:".LegacyTestMessage"`,
			reflect.TypeOf([][]byte(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "proto2_20180125_92554152/test.proto"]`,
			`name:"repeated_enum_v1" number:10016 label:LABEL_REPEATED type:TYPE_ENUM type_name:".google.golang.org.proto2_20180125.Message.ChildEnum" extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]proto2_20180125.Message_ChildEnum(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "proto2_20180125_92554152/test.proto"]`,
			`name:"repeated_message_v1" number:10017 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".google.golang.org.proto2_20180125.Message.ChildMessage" extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]*proto2_20180125.Message_ChildMessage(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "enum2.proto"]`,
			`name:"repeated_enum_v2" number:10018 label:LABEL_REPEATED type:TYPE_ENUM type_name:".EnumProto2" extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]EnumProto2(nil)), depReg,
		),
		mustMakeExtensionType(
			`package:"fizz.buzz" dependency:["legacy.proto", "enum-messages.proto"]`,
			`name:"repeated_message_v2" number:10019 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".EnumMessages" extendee:".LegacyTestMessage"`,
			reflect.TypeOf([]*EnumMessages(nil)), depReg,
		),
	}

	extensionDescs = []*pimpl.ExtensionInfo{{
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*bool)(nil),
		Field:         10000,
		Name:          "fizz.buzz.optional_bool",
		Tag:           "varint,10000,opt,name=optional_bool,def=1",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*int32)(nil),
		Field:         10001,
		Name:          "fizz.buzz.optional_int32",
		Tag:           "varint,10001,opt,name=optional_int32,def=-12345",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*uint32)(nil),
		Field:         10002,
		Name:          "fizz.buzz.optional_uint32",
		Tag:           "varint,10002,opt,name=optional_uint32,def=3200",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*float32)(nil),
		Field:         10003,
		Name:          "fizz.buzz.optional_float",
		Tag:           "fixed32,10003,opt,name=optional_float,def=3.14159",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*string)(nil),
		Field:         10004,
		Name:          "fizz.buzz.optional_string",
		Tag:           "bytes,10004,opt,name=optional_string,def=hello, \"world!\"\n",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]byte)(nil),
		Field:         10005,
		Name:          "fizz.buzz.optional_bytes",
		Tag:           "bytes,10005,opt,name=optional_bytes,def=dead\\336\\255\\276\\357beef",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*proto2_20180125.Message_ChildEnum)(nil),
		Field:         10006,
		Name:          "fizz.buzz.optional_enum_v1",
		Tag:           "varint,10006,opt,name=optional_enum_v1,enum=google.golang.org.proto2_20180125.Message_ChildEnum,def=0",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*proto2_20180125.Message_ChildMessage)(nil),
		Field:         10007,
		Name:          "fizz.buzz.optional_message_v1",
		Tag:           "bytes,10007,opt,name=optional_message_v1",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*EnumProto2)(nil),
		Field:         10008,
		Name:          "fizz.buzz.optional_enum_v2",
		Tag:           "varint,10008,opt,name=optional_enum_v2,enum=EnumProto2,def=57005",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: (*EnumMessages)(nil),
		Field:         10009,
		Name:          "fizz.buzz.optional_message_v2",
		Tag:           "bytes,10009,opt,name=optional_message_v2",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]bool)(nil),
		Field:         10010,
		Name:          "fizz.buzz.repeated_bool",
		Tag:           "varint,10010,rep,name=repeated_bool",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]int32)(nil),
		Field:         10011,
		Name:          "fizz.buzz.repeated_int32",
		Tag:           "varint,10011,rep,name=repeated_int32",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]uint32)(nil),
		Field:         10012,
		Name:          "fizz.buzz.repeated_uint32",
		Tag:           "varint,10012,rep,name=repeated_uint32",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]float32)(nil),
		Field:         10013,
		Name:          "fizz.buzz.repeated_float",
		Tag:           "fixed32,10013,rep,name=repeated_float",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]string)(nil),
		Field:         10014,
		Name:          "fizz.buzz.repeated_string",
		Tag:           "bytes,10014,rep,name=repeated_string",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([][]byte)(nil),
		Field:         10015,
		Name:          "fizz.buzz.repeated_bytes",
		Tag:           "bytes,10015,rep,name=repeated_bytes",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]proto2_20180125.Message_ChildEnum)(nil),
		Field:         10016,
		Name:          "fizz.buzz.repeated_enum_v1",
		Tag:           "varint,10016,rep,name=repeated_enum_v1,enum=google.golang.org.proto2_20180125.Message_ChildEnum",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]*proto2_20180125.Message_ChildMessage)(nil),
		Field:         10017,
		Name:          "fizz.buzz.repeated_message_v1",
		Tag:           "bytes,10017,rep,name=repeated_message_v1",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]EnumProto2)(nil),
		Field:         10018,
		Name:          "fizz.buzz.repeated_enum_v2",
		Tag:           "varint,10018,rep,name=repeated_enum_v2,enum=EnumProto2",
		Filename:      "test.proto",
	}, {
		ExtendedType:  (*LegacyTestMessage)(nil),
		ExtensionType: ([]*EnumMessages)(nil),
		Field:         10019,
		Name:          "fizz.buzz.repeated_message_v2",
		Tag:           "bytes,10019,rep,name=repeated_message_v2",
		Filename:      "test.proto",
	}}
)

func TestLegacyExtensions(t *testing.T) {
	opts := cmp.Options{cmp.Comparer(func(x, y *proto2_20180125.Message_ChildMessage) bool {
		return x == y // pointer compare messages for object identity
	})}

	m := pimpl.Export{}.MessageOf(new(LegacyTestMessage))

	// Check that getting the zero value returns the default value for scalars,
	// nil for singular messages, and an empty list for repeated fields.
	defaultValues := map[int]interface{}{
		0: bool(true),
		1: int32(-12345),
		2: uint32(3200),
		3: float32(3.14159),
		4: string("hello, \"world!\"\n"),
		5: []byte("dead\xde\xad\xbe\xefbeef"),
		6: proto2_20180125.Message_ALPHA,
		7: nil,
		8: EnumProto2(0xdead),
		9: nil,
	}
	for i, xt := range extensionTypes {
		var got interface{}
		xd := xt.TypeDescriptor()
		if !(xd.IsList() || xd.IsMap() || xd.Message() != nil) {
			got = xt.InterfaceOf(m.Get(xd))
		}
		want := defaultValues[i]
		if diff := cmp.Diff(want, got, opts); diff != "" {
			t.Errorf("Message.Get(%d) mismatch (-want +got):\n%v", xd.Number(), diff)
		}
	}

	// All fields should be unpopulated.
	for _, xt := range extensionTypes {
		xd := xt.TypeDescriptor()
		if m.Has(xd) {
			t.Errorf("Message.Has(%d) = true, want false", xd.Number())
		}
	}

	// Set some values and append to values to the lists.
	m1a := &proto2_20180125.Message_ChildMessage{F1: proto.String("m1a")}
	m1b := &proto2_20180125.Message_ChildMessage{F1: proto.String("m2b")}
	m2a := &EnumMessages{EnumP2: EnumProto2(0x1b).Enum()}
	m2b := &EnumMessages{EnumP2: EnumProto2(0x2b).Enum()}
	setValues := map[int]interface{}{
		0:  bool(false),
		1:  int32(-54321),
		2:  uint32(6400),
		3:  float32(2.71828),
		4:  string("goodbye, \"world!\"\n"),
		5:  []byte("live\xde\xad\xbe\xefchicken"),
		6:  proto2_20180125.Message_CHARLIE,
		7:  m1a,
		8:  EnumProto2(0xbeef),
		9:  m2a,
		10: []bool{true},
		11: []int32{-1000},
		12: []uint32{1280},
		13: []float32{1.6180},
		14: []string{"zero"},
		15: [][]byte{[]byte("zero")},
		16: []proto2_20180125.Message_ChildEnum{proto2_20180125.Message_BRAVO},
		17: []*proto2_20180125.Message_ChildMessage{m1b},
		18: []EnumProto2{0xdead},
		19: []*EnumMessages{m2b},
	}
	for i, xt := range extensionTypes {
		m.Set(xt.TypeDescriptor(), xt.ValueOf(setValues[i]))
	}
	for i, xt := range extensionTypes[len(extensionTypes)/2:] {
		v := extensionTypes[i].ValueOf(setValues[i])
		m.Get(xt.TypeDescriptor()).List().Append(v)
	}

	// Get the values and check for equality.
	getValues := map[int]interface{}{
		0:  bool(false),
		1:  int32(-54321),
		2:  uint32(6400),
		3:  float32(2.71828),
		4:  string("goodbye, \"world!\"\n"),
		5:  []byte("live\xde\xad\xbe\xefchicken"),
		6:  proto2_20180125.Message_ChildEnum(proto2_20180125.Message_CHARLIE),
		7:  m1a,
		8:  EnumProto2(0xbeef),
		9:  m2a,
		10: []bool{true, false},
		11: []int32{-1000, -54321},
		12: []uint32{1280, 6400},
		13: []float32{1.6180, 2.71828},
		14: []string{"zero", "goodbye, \"world!\"\n"},
		15: [][]byte{[]byte("zero"), []byte("live\xde\xad\xbe\xefchicken")},
		16: []proto2_20180125.Message_ChildEnum{proto2_20180125.Message_BRAVO, proto2_20180125.Message_CHARLIE},
		17: []*proto2_20180125.Message_ChildMessage{m1b, m1a},
		18: []EnumProto2{0xdead, 0xbeef},
		19: []*EnumMessages{m2b, m2a},
	}
	for i, xt := range extensionTypes {
		xd := xt.TypeDescriptor()
		got := xt.InterfaceOf(m.Get(xd))
		want := getValues[i]
		if diff := cmp.Diff(want, got, opts); diff != "" {
			t.Errorf("Message.Get(%d) mismatch (-want +got):\n%v", xd.Number(), diff)
		}
	}

	// Clear all singular fields and truncate all repeated fields.
	for _, xt := range extensionTypes[:len(extensionTypes)/2] {
		m.Clear(xt.TypeDescriptor())
	}
	for _, xt := range extensionTypes[len(extensionTypes)/2:] {
		m.Get(xt.TypeDescriptor()).List().Truncate(0)
	}

	// Clear all repeated fields.
	for _, xt := range extensionTypes[len(extensionTypes)/2:] {
		m.Clear(xt.TypeDescriptor())
	}
}

func TestLegacyExtensionConvert(t *testing.T) {
	for i := range extensionTypes {
		i := i
		t.Run("", func(t *testing.T) {
			t.Parallel()

			wantType := extensionTypes[i]
			wantDesc := extensionDescs[i]
			gotType := (protoreflect.ExtensionType)(wantDesc)
			gotDesc := wantType.(*pimpl.ExtensionInfo)

			// Concurrently call accessors to trigger possible races.
			for _, xt := range []protoreflect.ExtensionType{wantType, wantDesc} {
				xt := xt
				go func() { xt.New() }()
				go func() { xt.Zero() }()
				go func() { xt.TypeDescriptor() }()
			}

			// TODO: We need a test package to compare descriptors.
			type list interface {
				Len() int
				pragma.DoNotImplement
			}
			opts := cmp.Options{
				cmp.Comparer(func(x, y reflect.Type) bool {
					return x == y
				}),
				cmp.Transformer("", func(x list) []interface{} {
					out := make([]interface{}, x.Len())
					v := reflect.ValueOf(x)
					for i := 0; i < x.Len(); i++ {
						m := v.MethodByName("Get")
						out[i] = m.Call([]reflect.Value{reflect.ValueOf(i)})[0].Interface()
					}
					return out
				}),
				cmp.Transformer("", func(x protoreflect.Descriptor) map[string]interface{} {
					out := make(map[string]interface{})
					v := reflect.ValueOf(x)
					for i := 0; i < v.NumMethod(); i++ {
						name := v.Type().Method(i).Name
						if m := v.Method(i); m.Type().NumIn() == 0 && m.Type().NumOut() == 1 {
							switch name {
							case "ParentFile", "Parent":
							// Ignore parents to avoid recursive cycle.
							case "Options":
								// Ignore descriptor options since protos are not cmperable.
							case "ContainingOneof", "ContainingMessage", "Enum", "Message":
								// Avoid descending into a dependency to avoid a cycle.
								// Just record the full name if available.
								//
								// TODO: Cycle support in cmp would be useful here.
								v := m.Call(nil)[0]
								if !v.IsNil() {
									out[name] = v.Interface().(protoreflect.Descriptor).FullName()
								}
							case "Type":
								// Ignore ExtensionTypeDescriptor.Type method to avoid cycle.
							default:
								out[name] = m.Call(nil)[0].Interface()
							}
						}
					}
					return out
				}),
				cmp.Transformer("", func(xt protoreflect.ExtensionType) map[string]interface{} {
					return map[string]interface{}{
						"Descriptor": xt.TypeDescriptor(),
					}
				}),
				cmp.Transformer("", func(v protoreflect.Value) interface{} {
					return v.Interface()
				}),
			}
			if diff := cmp.Diff(&wantType, &gotType, opts); diff != "" {
				t.Errorf("ExtensionType mismatch (-want, +got):\n%v", diff)
			}

			opts = cmp.Options{
				cmpopts.IgnoreFields(pimpl.ExtensionInfo{}, "ExtensionType"),
				cmpopts.IgnoreUnexported(pimpl.ExtensionInfo{}),
			}
			if diff := cmp.Diff(wantDesc, gotDesc, opts); diff != "" {
				t.Errorf("ExtensionDesc mismatch (-want, +got):\n%v", diff)
			}
		})
	}
}

type (
	MessageA struct {
		A1 *MessageA `protobuf:"bytes,1,req,name=a1"`
		A2 *MessageB `protobuf:"bytes,2,req,name=a2"`
		A3 Enum      `protobuf:"varint,3,opt,name=a3,enum=legacy.Enum"`
	}
	MessageB struct {
		B1 *MessageA `protobuf:"bytes,1,req,name=b1"`
		B2 *MessageB `protobuf:"bytes,2,req,name=b2"`
		B3 Enum      `protobuf:"varint,3,opt,name=b3,enum=legacy.Enum"`
	}
	Enum int32
)

func (*MessageA) Reset()                      { panic("not implemented") }
func (*MessageA) String() string              { panic("not implemented") }
func (*MessageA) ProtoMessage()               { panic("not implemented") }
func (*MessageA) Descriptor() ([]byte, []int) { return concurrentFD, []int{0} }

func (*MessageB) Reset()                      { panic("not implemented") }
func (*MessageB) String() string              { panic("not implemented") }
func (*MessageB) ProtoMessage()               { panic("not implemented") }
func (*MessageB) Descriptor() ([]byte, []int) { return concurrentFD, []int{1} }

func (Enum) EnumDescriptor() ([]byte, []int) { return concurrentFD, []int{0} }

var concurrentFD = func() []byte {
	b, _ := proto.Marshal(protodesc.ToFileDescriptorProto(mustMakeFileDesc(`
		name:    "concurrent.proto"
		syntax:  "proto2"
		package: "legacy"
		message_type: [{
			name: "MessageA"
			field: [
				{name:"a1" number:1 label:LABEL_REQUIRED type:TYPE_MESSAGE type_name:".legacy.MessageA"},
				{name:"a2" number:2 label:LABEL_REQUIRED type:TYPE_MESSAGE type_name:".legacy.MessageB"},
				{name:"a3" number:3 label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".legacy.Enum"}
			]
		}, {
			name: "MessageB"
			field: [
				{name:"a1" number:1 label:LABEL_REQUIRED type:TYPE_MESSAGE type_name:".legacy.MessageA"},
				{name:"a2" number:2 label:LABEL_REQUIRED type:TYPE_MESSAGE type_name:".legacy.MessageB"},
				{name:"a3" number:3 label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".legacy.Enum"}
			]
		}]
		enum_type: [{
			name:  "Enum"
			value: [{name:"FOO" number:500}]
		}]
	`, nil)))
	return pimpl.Export{}.CompressGZIP(b)
}()

// TestLegacyConcurrentInit tests that concurrent wrapping of multiple legacy types
// results in the exact same descriptor being created.
func TestLegacyConcurrentInit(t *testing.T) {
	const numParallel = 5
	var messageATypes [numParallel]protoreflect.MessageType
	var messageBTypes [numParallel]protoreflect.MessageType
	var enumDescs [numParallel]protoreflect.EnumDescriptor

	// Concurrently load message and enum types.
	var wg sync.WaitGroup
	for i := 0; i < numParallel; i++ {
		i := i
		wg.Add(3)
		go func() {
			defer wg.Done()
			messageATypes[i] = pimpl.Export{}.MessageTypeOf((*MessageA)(nil))
		}()
		go func() {
			defer wg.Done()
			messageBTypes[i] = pimpl.Export{}.MessageTypeOf((*MessageB)(nil))
		}()
		go func() {
			defer wg.Done()
			enumDescs[i] = pimpl.Export{}.EnumDescriptorOf(Enum(0))
		}()
	}
	wg.Wait()

	var (
		wantMTA = messageATypes[0]
		wantMDA = messageATypes[0].Descriptor().Fields().ByNumber(1).Message()
		wantMTB = messageBTypes[0]
		wantMDB = messageBTypes[0].Descriptor().Fields().ByNumber(2).Message()
		wantED  = messageATypes[0].Descriptor().Fields().ByNumber(3).Enum()
	)

	for _, gotMT := range messageATypes[1:] {
		if gotMT != wantMTA {
			t.Error("MessageType(MessageA) mismatch")
		}
		if gotMDA := gotMT.Descriptor().Fields().ByNumber(1).Message(); gotMDA != wantMDA {
			t.Error("MessageDescriptor(MessageA) mismatch")
		}
		if gotMDB := gotMT.Descriptor().Fields().ByNumber(2).Message(); gotMDB != wantMDB {
			t.Error("MessageDescriptor(MessageB) mismatch")
		}
		if gotED := gotMT.Descriptor().Fields().ByNumber(3).Enum(); gotED != wantED {
			t.Error("EnumDescriptor(Enum) mismatch")
		}
	}
	for _, gotMT := range messageBTypes[1:] {
		if gotMT != wantMTB {
			t.Error("MessageType(MessageB) mismatch")
		}
		if gotMDA := gotMT.Descriptor().Fields().ByNumber(1).Message(); gotMDA != wantMDA {
			t.Error("MessageDescriptor(MessageA) mismatch")
		}
		if gotMDB := gotMT.Descriptor().Fields().ByNumber(2).Message(); gotMDB != wantMDB {
			t.Error("MessageDescriptor(MessageB) mismatch")
		}
		if gotED := gotMT.Descriptor().Fields().ByNumber(3).Enum(); gotED != wantED {
			t.Error("EnumDescriptor(Enum) mismatch")
		}
	}
	for _, gotED := range enumDescs[1:] {
		if gotED != wantED {
			t.Error("EnumType(Enum) mismatch")
		}
	}
}

type LegacyTestMessageName1 struct{}

func (*LegacyTestMessageName1) Reset()         { panic("not implemented") }
func (*LegacyTestMessageName1) String() string { panic("not implemented") }
func (*LegacyTestMessageName1) ProtoMessage()  { panic("not implemented") }

type LegacyTestMessageName2 struct{}

func (*LegacyTestMessageName2) Reset()         { panic("not implemented") }
func (*LegacyTestMessageName2) String() string { panic("not implemented") }
func (*LegacyTestMessageName2) ProtoMessage()  { panic("not implemented") }
func (*LegacyTestMessageName2) XXX_MessageName() string {
	return "google.golang.org.LegacyTestMessageName2"
}

func TestLegacyMessageName(t *testing.T) {
	tests := []struct {
		in          protoiface.MessageV1
		suggestName protoreflect.FullName
		wantName    protoreflect.FullName
	}{
		{new(LegacyTestMessageName1), "google.golang.org.LegacyTestMessageName1", "google.golang.org.LegacyTestMessageName1"},
		{new(LegacyTestMessageName2), "", "google.golang.org.LegacyTestMessageName2"},
	}

	for _, tt := range tests {
		mt := pimpl.Export{}.LegacyMessageTypeOf(tt.in, tt.suggestName)
		if got := mt.Descriptor().FullName(); got != tt.wantName {
			t.Errorf("type: %T, name mismatch: got %v, want %v", tt.in, got, tt.wantName)
		}
	}
}
