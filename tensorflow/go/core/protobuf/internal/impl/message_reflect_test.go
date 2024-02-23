// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl_test

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/protobuf/encoding/prototext"
	pimpl "google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/testing/protopack"

	proto2_20180125 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20180125_92554152"
	testpb "google.golang.org/protobuf/internal/testprotos/test"
	"google.golang.org/protobuf/types/descriptorpb"
)

// List of test operations to perform on messages, lists, or maps.
type (
	messageOp  interface{ isMessageOp() }
	messageOps []messageOp

	listOp  interface{ isListOp() }
	listOps []listOp

	mapOp  interface{ isMapOp() }
	mapOps []mapOp
)

// Test operations performed on a message.
type (
	// check that the message contents match
	equalMessage struct{ protoreflect.Message }
	// check presence for specific fields in the message
	hasFields map[protoreflect.FieldNumber]bool
	// check that specific message fields match
	getFields map[protoreflect.FieldNumber]protoreflect.Value
	// set specific message fields
	setFields map[protoreflect.FieldNumber]protoreflect.Value
	// clear specific fields in the message
	clearFields []protoreflect.FieldNumber
	// check for the presence of specific oneof member fields.
	whichOneofs map[protoreflect.Name]protoreflect.FieldNumber
	// apply messageOps on each specified message field
	messageFields        map[protoreflect.FieldNumber]messageOps
	messageFieldsMutable map[protoreflect.FieldNumber]messageOps
	// apply listOps on each specified list field
	listFields        map[protoreflect.FieldNumber]listOps
	listFieldsMutable map[protoreflect.FieldNumber]listOps
	// apply mapOps on each specified map fields
	mapFields        map[protoreflect.FieldNumber]mapOps
	mapFieldsMutable map[protoreflect.FieldNumber]mapOps
	// range through all fields and check that they match
	rangeFields map[protoreflect.FieldNumber]protoreflect.Value
)

func (equalMessage) isMessageOp()         {}
func (hasFields) isMessageOp()            {}
func (getFields) isMessageOp()            {}
func (setFields) isMessageOp()            {}
func (clearFields) isMessageOp()          {}
func (whichOneofs) isMessageOp()          {}
func (messageFields) isMessageOp()        {}
func (messageFieldsMutable) isMessageOp() {}
func (listFields) isMessageOp()           {}
func (listFieldsMutable) isMessageOp()    {}
func (mapFields) isMessageOp()            {}
func (mapFieldsMutable) isMessageOp()     {}
func (rangeFields) isMessageOp()          {}

// Test operations performed on a list.
type (
	// check that the list contents match
	equalList struct{ protoreflect.List }
	// check that list length matches
	lenList int
	// check that specific list entries match
	getList map[int]protoreflect.Value
	// set specific list entries
	setList map[int]protoreflect.Value
	// append entries to the list
	appendList []protoreflect.Value
	// apply messageOps on a newly appended message
	appendMessageList messageOps
	// truncate the list to the specified length
	truncList int
)

func (equalList) isListOp()         {}
func (lenList) isListOp()           {}
func (getList) isListOp()           {}
func (setList) isListOp()           {}
func (appendList) isListOp()        {}
func (appendMessageList) isListOp() {}
func (truncList) isListOp()         {}

// Test operations performed on a map.
type (
	// check that the map contents match
	equalMap struct{ protoreflect.Map }
	// check that map length matches
	lenMap int
	// check presence for specific entries in the map
	hasMap map[interface{}]bool
	// check that specific map entries match
	getMap map[interface{}]protoreflect.Value
	// set specific map entries
	setMap map[interface{}]protoreflect.Value
	// clear specific entries in the map
	clearMap []interface{}
	// apply messageOps on each specified message entry
	messageMap map[interface{}]messageOps
	// range through all entries and check that they match
	rangeMap map[interface{}]protoreflect.Value
)

func (equalMap) isMapOp()   {}
func (lenMap) isMapOp()     {}
func (hasMap) isMapOp()     {}
func (getMap) isMapOp()     {}
func (setMap) isMapOp()     {}
func (clearMap) isMapOp()   {}
func (messageMap) isMapOp() {}
func (rangeMap) isMapOp()   {}

type ScalarProto2 struct {
	Bool    *bool    `protobuf:"1"`
	Int32   *int32   `protobuf:"2"`
	Int64   *int64   `protobuf:"3"`
	Uint32  *uint32  `protobuf:"4"`
	Uint64  *uint64  `protobuf:"5"`
	Float32 *float32 `protobuf:"6"`
	Float64 *float64 `protobuf:"7"`
	String  *string  `protobuf:"8"`
	StringA []byte   `protobuf:"9"`
	Bytes   []byte   `protobuf:"10"`
	BytesA  *string  `protobuf:"11"`

	MyBool    *MyBool    `protobuf:"12"`
	MyInt32   *MyInt32   `protobuf:"13"`
	MyInt64   *MyInt64   `protobuf:"14"`
	MyUint32  *MyUint32  `protobuf:"15"`
	MyUint64  *MyUint64  `protobuf:"16"`
	MyFloat32 *MyFloat32 `protobuf:"17"`
	MyFloat64 *MyFloat64 `protobuf:"18"`
	MyString  *MyString  `protobuf:"19"`
	MyStringA MyBytes    `protobuf:"20"`
	MyBytes   MyBytes    `protobuf:"21"`
	MyBytesA  *MyString  `protobuf:"22"`
}

func mustMakeEnumDesc(path string, syntax protoreflect.Syntax, enumDesc string) protoreflect.EnumDescriptor {
	s := fmt.Sprintf(`name:%q syntax:%q enum_type:[{%s}]`, path, syntax, enumDesc)
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(pb, nil)
	if err != nil {
		panic(err)
	}
	return fd.Enums().Get(0)
}

func mustMakeMessageDesc(path string, syntax protoreflect.Syntax, fileDesc, msgDesc string, r protodesc.Resolver) protoreflect.MessageDescriptor {
	s := fmt.Sprintf(`name:%q syntax:%q %s message_type:[{%s}]`, path, syntax, fileDesc, msgDesc)
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(pb, r)
	if err != nil {
		panic(err)
	}
	return fd.Messages().Get(0)
}

var V = protoreflect.ValueOf
var VE = func(n protoreflect.EnumNumber) protoreflect.Value { return V(n) }

type (
	MyBool    bool
	MyInt32   int32
	MyInt64   int64
	MyUint32  uint32
	MyUint64  uint64
	MyFloat32 float32
	MyFloat64 float64
	MyString  string
	MyBytes   []byte

	ListStrings []MyString
	ListBytes   []MyBytes

	MapStrings map[MyString]MyString
	MapBytes   map[MyString]MyBytes
)

var scalarProto2Type = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(ScalarProto2)), Desc: mustMakeMessageDesc("scalar2.proto", protoreflect.Proto2, "", `
		name: "ScalarProto2"
		field: [
			{name:"f1"  number:1  label:LABEL_OPTIONAL type:TYPE_BOOL   default_value:"true"},
			{name:"f2"  number:2  label:LABEL_OPTIONAL type:TYPE_INT32  default_value:"2"},
			{name:"f3"  number:3  label:LABEL_OPTIONAL type:TYPE_INT64  default_value:"3"},
			{name:"f4"  number:4  label:LABEL_OPTIONAL type:TYPE_UINT32 default_value:"4"},
			{name:"f5"  number:5  label:LABEL_OPTIONAL type:TYPE_UINT64 default_value:"5"},
			{name:"f6"  number:6  label:LABEL_OPTIONAL type:TYPE_FLOAT  default_value:"6"},
			{name:"f7"  number:7  label:LABEL_OPTIONAL type:TYPE_DOUBLE default_value:"7"},
			{name:"f8"  number:8  label:LABEL_OPTIONAL type:TYPE_STRING default_value:"8"},
			{name:"f9"  number:9  label:LABEL_OPTIONAL type:TYPE_STRING default_value:"9"},
			{name:"f10" number:10 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"10"},
			{name:"f11" number:11 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"11"},

			{name:"f12" number:12 label:LABEL_OPTIONAL type:TYPE_BOOL   default_value:"true"},
			{name:"f13" number:13 label:LABEL_OPTIONAL type:TYPE_INT32  default_value:"13"},
			{name:"f14" number:14 label:LABEL_OPTIONAL type:TYPE_INT64  default_value:"14"},
			{name:"f15" number:15 label:LABEL_OPTIONAL type:TYPE_UINT32 default_value:"15"},
			{name:"f16" number:16 label:LABEL_OPTIONAL type:TYPE_UINT64 default_value:"16"},
			{name:"f17" number:17 label:LABEL_OPTIONAL type:TYPE_FLOAT  default_value:"17"},
			{name:"f18" number:18 label:LABEL_OPTIONAL type:TYPE_DOUBLE default_value:"18"},
			{name:"f19" number:19 label:LABEL_OPTIONAL type:TYPE_STRING default_value:"19"},
			{name:"f20" number:20 label:LABEL_OPTIONAL type:TYPE_STRING default_value:"20"},
			{name:"f21" number:21 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"21"},
			{name:"f22" number:22 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"22"}
		]
	`, nil),
}

func (m *ScalarProto2) ProtoReflect() protoreflect.Message { return scalarProto2Type.MessageOf(m) }

func TestScalarProto2(t *testing.T) {
	testMessage(t, nil, new(ScalarProto2).ProtoReflect(), messageOps{
		hasFields{
			1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false,
			12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false,
		},
		getFields{
			1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V([]byte("10")), 11: V([]byte("11")),
			12: V(bool(true)), 13: V(int32(13)), 14: V(int64(14)), 15: V(uint32(15)), 16: V(uint64(16)), 17: V(float32(17)), 18: V(float64(18)), 19: V(string("19")), 20: V(string("20")), 21: V([]byte("21")), 22: V([]byte("22")),
		},
		setFields{
			1: V(bool(false)), 2: V(int32(0)), 3: V(int64(0)), 4: V(uint32(0)), 5: V(uint64(0)), 6: V(float32(0)), 7: V(float64(0)), 8: V(string("")), 9: V(string("")), 10: V([]byte(nil)), 11: V([]byte(nil)),
			12: V(bool(false)), 13: V(int32(0)), 14: V(int64(0)), 15: V(uint32(0)), 16: V(uint64(0)), 17: V(float32(0)), 18: V(float64(0)), 19: V(string("")), 20: V(string("")), 21: V([]byte(nil)), 22: V([]byte(nil)),
		},
		hasFields{
			1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true,
			12: true, 13: true, 14: true, 15: true, 16: true, 17: true, 18: true, 19: true, 20: true, 21: true, 22: true,
		},
		equalMessage{(&ScalarProto2{
			new(bool), new(int32), new(int64), new(uint32), new(uint64), new(float32), new(float64), new(string), []byte{}, []byte{}, new(string),
			new(MyBool), new(MyInt32), new(MyInt64), new(MyUint32), new(MyUint64), new(MyFloat32), new(MyFloat64), new(MyString), MyBytes{}, MyBytes{}, new(MyString),
		}).ProtoReflect()},
		clearFields{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
		equalMessage{new(ScalarProto2).ProtoReflect()},

		// Setting a bytes field nil empty bytes should preserve presence.
		setFields{10: V([]byte(nil)), 11: V([]byte(nil)), 21: V([]byte(nil)), 22: V([]byte(nil))},
		getFields{10: V([]byte{}), 11: V([]byte(nil)), 21: V([]byte{}), 22: V([]byte(nil))},
		hasFields{10: true, 11: true, 21: true, 22: true},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*ScalarProto2)(nil).ProtoReflect(), messageOps{
		hasFields{
			1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false,
			12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false,
		},
		getFields{
			1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V([]byte("10")), 11: V([]byte("11")),
			12: V(bool(true)), 13: V(int32(13)), 14: V(int64(14)), 15: V(uint32(15)), 16: V(uint64(16)), 17: V(float32(17)), 18: V(float64(18)), 19: V(string("19")), 20: V(string("20")), 21: V([]byte("21")), 22: V([]byte("22")),
		},
	})
}

type ScalarProto3 struct {
	Bool    bool    `protobuf:"1"`
	Int32   int32   `protobuf:"2"`
	Int64   int64   `protobuf:"3"`
	Uint32  uint32  `protobuf:"4"`
	Uint64  uint64  `protobuf:"5"`
	Float32 float32 `protobuf:"6"`
	Float64 float64 `protobuf:"7"`
	String  string  `protobuf:"8"`
	StringA []byte  `protobuf:"9"`
	Bytes   []byte  `protobuf:"10"`
	BytesA  string  `protobuf:"11"`

	MyBool    MyBool    `protobuf:"12"`
	MyInt32   MyInt32   `protobuf:"13"`
	MyInt64   MyInt64   `protobuf:"14"`
	MyUint32  MyUint32  `protobuf:"15"`
	MyUint64  MyUint64  `protobuf:"16"`
	MyFloat32 MyFloat32 `protobuf:"17"`
	MyFloat64 MyFloat64 `protobuf:"18"`
	MyString  MyString  `protobuf:"19"`
	MyStringA MyBytes   `protobuf:"20"`
	MyBytes   MyBytes   `protobuf:"21"`
	MyBytesA  MyString  `protobuf:"22"`
}

var scalarProto3Type = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(ScalarProto3)), Desc: mustMakeMessageDesc("scalar3.proto", protoreflect.Proto3, "", `
		name: "ScalarProto3"
		field: [
			{name:"f1"  number:1  label:LABEL_OPTIONAL type:TYPE_BOOL},
			{name:"f2"  number:2  label:LABEL_OPTIONAL type:TYPE_INT32},
			{name:"f3"  number:3  label:LABEL_OPTIONAL type:TYPE_INT64},
			{name:"f4"  number:4  label:LABEL_OPTIONAL type:TYPE_UINT32},
			{name:"f5"  number:5  label:LABEL_OPTIONAL type:TYPE_UINT64},
			{name:"f6"  number:6  label:LABEL_OPTIONAL type:TYPE_FLOAT},
			{name:"f7"  number:7  label:LABEL_OPTIONAL type:TYPE_DOUBLE},
			{name:"f8"  number:8  label:LABEL_OPTIONAL type:TYPE_STRING},
			{name:"f9"  number:9  label:LABEL_OPTIONAL type:TYPE_STRING},
			{name:"f10" number:10 label:LABEL_OPTIONAL type:TYPE_BYTES},
			{name:"f11" number:11 label:LABEL_OPTIONAL type:TYPE_BYTES},

			{name:"f12" number:12 label:LABEL_OPTIONAL type:TYPE_BOOL},
			{name:"f13" number:13 label:LABEL_OPTIONAL type:TYPE_INT32},
			{name:"f14" number:14 label:LABEL_OPTIONAL type:TYPE_INT64},
			{name:"f15" number:15 label:LABEL_OPTIONAL type:TYPE_UINT32},
			{name:"f16" number:16 label:LABEL_OPTIONAL type:TYPE_UINT64},
			{name:"f17" number:17 label:LABEL_OPTIONAL type:TYPE_FLOAT},
			{name:"f18" number:18 label:LABEL_OPTIONAL type:TYPE_DOUBLE},
			{name:"f19" number:19 label:LABEL_OPTIONAL type:TYPE_STRING},
			{name:"f20" number:20 label:LABEL_OPTIONAL type:TYPE_STRING},
			{name:"f21" number:21 label:LABEL_OPTIONAL type:TYPE_BYTES},
			{name:"f22" number:22 label:LABEL_OPTIONAL type:TYPE_BYTES}
		]
	`, nil),
}

func (m *ScalarProto3) ProtoReflect() protoreflect.Message { return scalarProto3Type.MessageOf(m) }

func TestScalarProto3(t *testing.T) {
	testMessage(t, nil, new(ScalarProto3).ProtoReflect(), messageOps{
		hasFields{
			1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false,
			12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false,
		},
		getFields{
			1: V(bool(false)), 2: V(int32(0)), 3: V(int64(0)), 4: V(uint32(0)), 5: V(uint64(0)), 6: V(float32(0)), 7: V(float64(0)), 8: V(string("")), 9: V(string("")), 10: V([]byte(nil)), 11: V([]byte(nil)),
			12: V(bool(false)), 13: V(int32(0)), 14: V(int64(0)), 15: V(uint32(0)), 16: V(uint64(0)), 17: V(float32(0)), 18: V(float64(0)), 19: V(string("")), 20: V(string("")), 21: V([]byte(nil)), 22: V([]byte(nil)),
		},
		setFields{
			1: V(bool(false)), 2: V(int32(0)), 3: V(int64(0)), 4: V(uint32(0)), 5: V(uint64(0)), 6: V(float32(0)), 7: V(float64(0)), 8: V(string("")), 9: V(string("")), 10: V([]byte(nil)), 11: V([]byte(nil)),
			12: V(bool(false)), 13: V(int32(0)), 14: V(int64(0)), 15: V(uint32(0)), 16: V(uint64(0)), 17: V(float32(0)), 18: V(float64(0)), 19: V(string("")), 20: V(string("")), 21: V([]byte(nil)), 22: V([]byte(nil)),
		},
		hasFields{
			1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false,
			12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false,
		},
		equalMessage{new(ScalarProto3).ProtoReflect()},
		setFields{
			1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V([]byte("10")), 11: V([]byte("11")),
			12: V(bool(true)), 13: V(int32(13)), 14: V(int64(14)), 15: V(uint32(15)), 16: V(uint64(16)), 17: V(float32(17)), 18: V(float64(18)), 19: V(string("19")), 20: V(string("20")), 21: V([]byte("21")), 22: V([]byte("22")),
		},
		hasFields{
			1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true,
			12: true, 13: true, 14: true, 15: true, 16: true, 17: true, 18: true, 19: true, 20: true, 21: true, 22: true,
		},
		equalMessage{(&ScalarProto3{
			true, 2, 3, 4, 5, 6, 7, "8", []byte("9"), []byte("10"), "11",
			true, 13, 14, 15, 16, 17, 18, "19", []byte("20"), []byte("21"), "22",
		}).ProtoReflect()},
		setFields{
			2: V(int32(-2)), 3: V(int64(-3)), 6: V(float32(math.Inf(-1))), 7: V(float64(math.NaN())),
		},
		hasFields{
			2: true, 3: true, 6: true, 7: true,
		},
		clearFields{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
		equalMessage{new(ScalarProto3).ProtoReflect()},

		// Verify that -0 triggers proper Has behavior.
		hasFields{6: false, 7: false},
		setFields{6: V(float32(math.Copysign(0, -1))), 7: V(float64(math.Copysign(0, -1)))},
		hasFields{6: true, 7: true},

		// Setting a bytes field to non-nil empty bytes should not preserve presence.
		setFields{10: V([]byte{}), 11: V([]byte{}), 21: V([]byte{}), 22: V([]byte{})},
		getFields{10: V([]byte(nil)), 11: V([]byte(nil)), 21: V([]byte(nil)), 22: V([]byte(nil))},
		hasFields{10: false, 11: false, 21: false, 22: false},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*ScalarProto3)(nil).ProtoReflect(), messageOps{
		hasFields{
			1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false,
			12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false,
		},
		getFields{
			1: V(bool(false)), 2: V(int32(0)), 3: V(int64(0)), 4: V(uint32(0)), 5: V(uint64(0)), 6: V(float32(0)), 7: V(float64(0)), 8: V(string("")), 9: V(string("")), 10: V([]byte(nil)), 11: V([]byte(nil)),
			12: V(bool(false)), 13: V(int32(0)), 14: V(int64(0)), 15: V(uint32(0)), 16: V(uint64(0)), 17: V(float32(0)), 18: V(float64(0)), 19: V(string("")), 20: V(string("")), 21: V([]byte(nil)), 22: V([]byte(nil)),
		},
	})
}

type ListScalars struct {
	Bools    []bool    `protobuf:"1"`
	Int32s   []int32   `protobuf:"2"`
	Int64s   []int64   `protobuf:"3"`
	Uint32s  []uint32  `protobuf:"4"`
	Uint64s  []uint64  `protobuf:"5"`
	Float32s []float32 `protobuf:"6"`
	Float64s []float64 `protobuf:"7"`
	Strings  []string  `protobuf:"8"`
	StringsA [][]byte  `protobuf:"9"`
	Bytes    [][]byte  `protobuf:"10"`
	BytesA   []string  `protobuf:"11"`

	MyStrings1 []MyString `protobuf:"12"`
	MyStrings2 []MyBytes  `protobuf:"13"`
	MyBytes1   []MyBytes  `protobuf:"14"`
	MyBytes2   []MyString `protobuf:"15"`

	MyStrings3 ListStrings `protobuf:"16"`
	MyStrings4 ListBytes   `protobuf:"17"`
	MyBytes3   ListBytes   `protobuf:"18"`
	MyBytes4   ListStrings `protobuf:"19"`
}

var listScalarsType = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(ListScalars)), Desc: mustMakeMessageDesc("list-scalars.proto", protoreflect.Proto2, "", `
		name: "ListScalars"
		field: [
			{name:"f1"  number:1  label:LABEL_REPEATED type:TYPE_BOOL},
			{name:"f2"  number:2  label:LABEL_REPEATED type:TYPE_INT32},
			{name:"f3"  number:3  label:LABEL_REPEATED type:TYPE_INT64},
			{name:"f4"  number:4  label:LABEL_REPEATED type:TYPE_UINT32},
			{name:"f5"  number:5  label:LABEL_REPEATED type:TYPE_UINT64},
			{name:"f6"  number:6  label:LABEL_REPEATED type:TYPE_FLOAT},
			{name:"f7"  number:7  label:LABEL_REPEATED type:TYPE_DOUBLE},
			{name:"f8"  number:8  label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f9"  number:9  label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f10" number:10 label:LABEL_REPEATED type:TYPE_BYTES},
			{name:"f11" number:11 label:LABEL_REPEATED type:TYPE_BYTES},

			{name:"f12" number:12 label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f13" number:13 label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f14" number:14 label:LABEL_REPEATED type:TYPE_BYTES},
			{name:"f15" number:15 label:LABEL_REPEATED type:TYPE_BYTES},

			{name:"f16" number:16 label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f17" number:17 label:LABEL_REPEATED type:TYPE_STRING},
			{name:"f18" number:18 label:LABEL_REPEATED type:TYPE_BYTES},
			{name:"f19" number:19 label:LABEL_REPEATED type:TYPE_BYTES}
		]
	`, nil),
}

func (m *ListScalars) ProtoReflect() protoreflect.Message { return listScalarsType.MessageOf(m) }

func TestListScalars(t *testing.T) {
	empty := new(ListScalars).ProtoReflect()
	want := (&ListScalars{
		Bools:    []bool{true, false, true},
		Int32s:   []int32{2, math.MinInt32, math.MaxInt32},
		Int64s:   []int64{3, math.MinInt64, math.MaxInt64},
		Uint32s:  []uint32{4, math.MaxUint32 / 2, math.MaxUint32},
		Uint64s:  []uint64{5, math.MaxUint64 / 2, math.MaxUint64},
		Float32s: []float32{6, math.SmallestNonzeroFloat32, float32(math.NaN()), math.MaxFloat32},
		Float64s: []float64{7, math.SmallestNonzeroFloat64, float64(math.NaN()), math.MaxFloat64},
		Strings:  []string{"8", "", "eight"},
		StringsA: [][]byte{[]byte("9"), nil, []byte("nine")},
		Bytes:    [][]byte{[]byte("10"), nil, []byte("ten")},
		BytesA:   []string{"11", "", "eleven"},

		MyStrings1: []MyString{"12", "", "twelve"},
		MyStrings2: []MyBytes{[]byte("13"), nil, []byte("thirteen")},
		MyBytes1:   []MyBytes{[]byte("14"), nil, []byte("fourteen")},
		MyBytes2:   []MyString{"15", "", "fifteen"},

		MyStrings3: ListStrings{"16", "", "sixteen"},
		MyStrings4: ListBytes{[]byte("17"), nil, []byte("seventeen")},
		MyBytes3:   ListBytes{[]byte("18"), nil, []byte("eighteen")},
		MyBytes4:   ListStrings{"19", "", "nineteen"},
	}).ProtoReflect()

	testMessage(t, nil, new(ListScalars).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false},
		getFields{1: getField(empty, 1), 3: getField(empty, 3), 5: getField(empty, 5), 7: getField(empty, 7), 9: getField(empty, 9), 11: getField(empty, 11), 13: getField(empty, 13), 15: getField(empty, 15), 17: getField(empty, 17), 19: getField(empty, 19)},
		setFields{1: getField(want, 1), 3: getField(want, 3), 5: getField(want, 5), 7: getField(want, 7), 9: getField(want, 9), 11: getField(want, 11), 13: getField(want, 13), 15: getField(want, 15), 17: getField(want, 17), 19: getField(want, 19)},
		listFieldsMutable{
			2: {
				lenList(0),
				appendList{V(int32(2)), V(int32(math.MinInt32)), V(int32(math.MaxInt32))},
				getList{0: V(int32(2)), 1: V(int32(math.MinInt32)), 2: V(int32(math.MaxInt32))},
				equalList{getField(want, 2).List()},
			},
			4: {
				appendList{V(uint32(0)), V(uint32(0)), V(uint32(0))},
				setList{0: V(uint32(4)), 1: V(uint32(math.MaxUint32 / 2)), 2: V(uint32(math.MaxUint32))},
				lenList(3),
			},
			6: {
				appendList{V(float32(6)), V(float32(math.SmallestNonzeroFloat32)), V(float32(math.NaN())), V(float32(math.MaxFloat32))},
				equalList{getField(want, 6).List()},
			},
			8: {
				appendList{V(""), V(""), V(""), V(""), V(""), V("")},
				lenList(6),
				setList{0: V("8"), 2: V("eight")},
				truncList(3),
				equalList{getField(want, 8).List()},
			},
			10: {
				appendList{V([]byte(nil)), V([]byte(nil))},
				setList{0: V([]byte("10"))},
				appendList{V([]byte("wrong"))},
				setList{2: V([]byte("ten"))},
				equalList{getField(want, 10).List()},
			},
			12: {
				appendList{V("12"), V("wrong"), V("twelve")},
				setList{1: V("")},
				equalList{getField(want, 12).List()},
			},
			14: {
				appendList{V([]byte("14")), V([]byte(nil)), V([]byte("fourteen"))},
				equalList{getField(want, 14).List()},
			},
			16: {
				appendList{V("16"), V(""), V("sixteen"), V("extra")},
				truncList(3),
				equalList{getField(want, 16).List()},
			},
			18: {
				appendList{V([]byte("18")), V([]byte(nil)), V([]byte("eighteen"))},
				equalList{getField(want, 18).List()},
			},
		},
		hasFields{1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true, 12: true, 13: true, 14: true, 15: true, 16: true, 17: true, 18: true, 19: true},
		equalMessage{want},
		clearFields{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
		equalMessage{empty},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*ListScalars)(nil).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false},
		listFields{2: {lenList(0)}, 4: {lenList(0)}, 6: {lenList(0)}, 8: {lenList(0)}, 10: {lenList(0)}, 12: {lenList(0)}, 14: {lenList(0)}, 16: {lenList(0)}, 18: {lenList(0)}},
	})
}

type MapScalars struct {
	KeyBools   map[bool]string   `protobuf:"1"`
	KeyInt32s  map[int32]string  `protobuf:"2"`
	KeyInt64s  map[int64]string  `protobuf:"3"`
	KeyUint32s map[uint32]string `protobuf:"4"`
	KeyUint64s map[uint64]string `protobuf:"5"`
	KeyStrings map[string]string `protobuf:"6"`

	ValBools    map[string]bool    `protobuf:"7"`
	ValInt32s   map[string]int32   `protobuf:"8"`
	ValInt64s   map[string]int64   `protobuf:"9"`
	ValUint32s  map[string]uint32  `protobuf:"10"`
	ValUint64s  map[string]uint64  `protobuf:"11"`
	ValFloat32s map[string]float32 `protobuf:"12"`
	ValFloat64s map[string]float64 `protobuf:"13"`
	ValStrings  map[string]string  `protobuf:"14"`
	ValStringsA map[string][]byte  `protobuf:"15"`
	ValBytes    map[string][]byte  `protobuf:"16"`
	ValBytesA   map[string]string  `protobuf:"17"`

	MyStrings1 map[MyString]MyString `protobuf:"18"`
	MyStrings2 map[MyString]MyBytes  `protobuf:"19"`
	MyBytes1   map[MyString]MyBytes  `protobuf:"20"`
	MyBytes2   map[MyString]MyString `protobuf:"21"`

	MyStrings3 MapStrings `protobuf:"22"`
	MyStrings4 MapBytes   `protobuf:"23"`
	MyBytes3   MapBytes   `protobuf:"24"`
	MyBytes4   MapStrings `protobuf:"25"`
}

var mapScalarsType = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(MapScalars)), Desc: mustMakeMessageDesc("map-scalars.proto", protoreflect.Proto2, "", `
		name: "MapScalars"
		field: [
			{name:"f1"  number:1  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F1Entry"},
			{name:"f2"  number:2  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F2Entry"},
			{name:"f3"  number:3  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F3Entry"},
			{name:"f4"  number:4  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F4Entry"},
			{name:"f5"  number:5  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F5Entry"},
			{name:"f6"  number:6  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F6Entry"},

			{name:"f7"  number:7  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F7Entry"},
			{name:"f8"  number:8  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F8Entry"},
			{name:"f9"  number:9  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F9Entry"},
			{name:"f10" number:10 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F10Entry"},
			{name:"f11" number:11 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F11Entry"},
			{name:"f12" number:12 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F12Entry"},
			{name:"f13" number:13 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F13Entry"},
			{name:"f14" number:14 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F14Entry"},
			{name:"f15" number:15 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F15Entry"},
			{name:"f16" number:16 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F16Entry"},
			{name:"f17" number:17 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F17Entry"},

			{name:"f18" number:18 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F18Entry"},
			{name:"f19" number:19 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F19Entry"},
			{name:"f20" number:20 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F20Entry"},
			{name:"f21" number:21 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F21Entry"},

			{name:"f22" number:22 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F22Entry"},
			{name:"f23" number:23 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F23Entry"},
			{name:"f24" number:24 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F24Entry"},
			{name:"f25" number:25 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".MapScalars.F25Entry"}
		]
		nested_type: [
			{name:"F1Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_BOOL},   {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F2Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_INT32},  {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F3Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_INT64},  {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F4Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_UINT32}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F5Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_UINT64}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F6Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},

			{name:"F7Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BOOL}]   options:{map_entry:true}},
			{name:"F8Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_INT32}]  options:{map_entry:true}},
			{name:"F9Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_INT64}]  options:{map_entry:true}},
			{name:"F10Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_UINT32}] options:{map_entry:true}},
			{name:"F11Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_UINT64}] options:{map_entry:true}},
			{name:"F12Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_FLOAT}]  options:{map_entry:true}},
			{name:"F13Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_DOUBLE}] options:{map_entry:true}},
			{name:"F14Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F15Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F16Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}},
			{name:"F17Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}},

			{name:"F18Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F19Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F20Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}},
			{name:"F21Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}},

			{name:"F22Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F23Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}] options:{map_entry:true}},
			{name:"F24Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}},
			{name:"F25Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]  options:{map_entry:true}}
		]
	`, nil),
}

func (m *MapScalars) ProtoReflect() protoreflect.Message { return mapScalarsType.MessageOf(m) }

func TestMapScalars(t *testing.T) {
	empty := new(MapScalars).ProtoReflect()
	want := (&MapScalars{
		KeyBools:   map[bool]string{true: "true", false: "false"},
		KeyInt32s:  map[int32]string{0: "zero", -1: "one", 2: "two"},
		KeyInt64s:  map[int64]string{0: "zero", -10: "ten", 20: "twenty"},
		KeyUint32s: map[uint32]string{0: "zero", 1: "one", 2: "two"},
		KeyUint64s: map[uint64]string{0: "zero", 10: "ten", 20: "twenty"},
		KeyStrings: map[string]string{"": "", "foo": "bar"},

		ValBools:    map[string]bool{"true": true, "false": false},
		ValInt32s:   map[string]int32{"one": 1, "two": 2, "three": 3},
		ValInt64s:   map[string]int64{"ten": 10, "twenty": -20, "thirty": 30},
		ValUint32s:  map[string]uint32{"0x00": 0x00, "0xff": 0xff, "0xdead": 0xdead},
		ValUint64s:  map[string]uint64{"0x00": 0x00, "0xff": 0xff, "0xdead": 0xdead},
		ValFloat32s: map[string]float32{"nan": float32(math.NaN()), "pi": float32(math.Pi)},
		ValFloat64s: map[string]float64{"nan": float64(math.NaN()), "pi": float64(math.Pi)},
		ValStrings:  map[string]string{"s1": "s1", "s2": "s2"},
		ValStringsA: map[string][]byte{"s1": []byte("s1"), "s2": []byte("s2")},
		ValBytes:    map[string][]byte{"s1": []byte("s1"), "s2": []byte("s2")},
		ValBytesA:   map[string]string{"s1": "s1", "s2": "s2"},

		MyStrings1: map[MyString]MyString{"s1": "s1", "s2": "s2"},
		MyStrings2: map[MyString]MyBytes{"s1": []byte("s1"), "s2": []byte("s2")},
		MyBytes1:   map[MyString]MyBytes{"s1": []byte("s1"), "s2": []byte("s2")},
		MyBytes2:   map[MyString]MyString{"s1": "s1", "s2": "s2"},

		MyStrings3: MapStrings{"s1": "s1", "s2": "s2"},
		MyStrings4: MapBytes{"s1": []byte("s1"), "s2": []byte("s2")},
		MyBytes3:   MapBytes{"s1": []byte("s1"), "s2": []byte("s2")},
		MyBytes4:   MapStrings{"s1": "s1", "s2": "s2"},
	}).ProtoReflect()

	testMessage(t, nil, new(MapScalars).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false, 23: false, 24: false, 25: false},
		getFields{1: getField(empty, 1), 3: getField(empty, 3), 5: getField(empty, 5), 7: getField(empty, 7), 9: getField(empty, 9), 11: getField(empty, 11), 13: getField(empty, 13), 15: getField(empty, 15), 17: getField(empty, 17), 19: getField(empty, 19), 21: getField(empty, 21), 23: getField(empty, 23), 25: getField(empty, 25)},
		setFields{1: getField(want, 1), 3: getField(want, 3), 5: getField(want, 5), 7: getField(want, 7), 9: getField(want, 9), 11: getField(want, 11), 13: getField(want, 13), 15: getField(want, 15), 17: getField(want, 17), 19: getField(want, 19), 21: getField(want, 21), 23: getField(want, 23), 25: getField(want, 25)},
		mapFieldsMutable{
			2: {
				lenMap(0),
				hasMap{int32(0): false, int32(-1): false, int32(2): false},
				setMap{int32(0): V("zero")},
				lenMap(1),
				hasMap{int32(0): true, int32(-1): false, int32(2): false},
				setMap{int32(-1): V("one")},
				lenMap(2),
				hasMap{int32(0): true, int32(-1): true, int32(2): false},
				setMap{int32(2): V("two")},
				lenMap(3),
				hasMap{int32(0): true, int32(-1): true, int32(2): true},
			},
			4: {
				setMap{uint32(0): V("zero"), uint32(1): V("one"), uint32(2): V("two")},
				equalMap{getField(want, 4).Map()},
			},
			6: {
				clearMap{"noexist"},
				setMap{"foo": V("bar")},
				setMap{"": V("empty")},
				getMap{"": V("empty"), "foo": V("bar"), "noexist": V(nil)},
				setMap{"": V(""), "extra": V("extra")},
				clearMap{"extra", "noexist"},
			},
			8: {
				equalMap{getField(empty, 8).Map()},
				setMap{"one": V(int32(1)), "two": V(int32(2)), "three": V(int32(3))},
			},
			10: {
				setMap{"0x00": V(uint32(0x00)), "0xff": V(uint32(0xff)), "0xdead": V(uint32(0xdead))},
				lenMap(3),
				equalMap{getField(want, 10).Map()},
				getMap{"0x00": V(uint32(0x00)), "0xff": V(uint32(0xff)), "0xdead": V(uint32(0xdead)), "0xdeadbeef": V(nil)},
			},
			12: {
				setMap{"nan": V(float32(math.NaN())), "pi": V(float32(math.Pi)), "e": V(float32(math.E))},
				clearMap{"e", "phi"},
				rangeMap{"nan": V(float32(math.NaN())), "pi": V(float32(math.Pi))},
			},
			14: {
				equalMap{getField(empty, 14).Map()},
				setMap{"s1": V("s1"), "s2": V("s2")},
			},
			16: {
				setMap{"s1": V([]byte("s1")), "s2": V([]byte("s2"))},
				equalMap{getField(want, 16).Map()},
			},
			18: {
				hasMap{"s1": false, "s2": false, "s3": false},
				setMap{"s1": V("s1"), "s2": V("s2")},
				hasMap{"s1": true, "s2": true, "s3": false},
			},
			20: {
				equalMap{getField(empty, 20).Map()},
				setMap{"s1": V([]byte("s1")), "s2": V([]byte("s2"))},
			},
			22: {
				rangeMap{},
				setMap{"s1": V("s1"), "s2": V("s2")},
				rangeMap{"s1": V("s1"), "s2": V("s2")},
				lenMap(2),
			},
			24: {
				setMap{"s1": V([]byte("s1")), "s2": V([]byte("s2"))},
				equalMap{getField(want, 24).Map()},
			},
		},
		hasFields{1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true, 12: true, 13: true, 14: true, 15: true, 16: true, 17: true, 18: true, 19: true, 20: true, 21: true, 22: true, 23: true, 24: true, 25: true},
		equalMessage{want},
		clearFields{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
		equalMessage{empty},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*MapScalars)(nil).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false, 14: false, 15: false, 16: false, 17: false, 18: false, 19: false, 20: false, 21: false, 22: false, 23: false, 24: false, 25: false},
		mapFields{2: {lenMap(0)}, 4: {lenMap(0)}, 6: {lenMap(0)}, 8: {lenMap(0)}, 10: {lenMap(0)}, 12: {lenMap(0)}, 14: {lenMap(0)}, 16: {lenMap(0)}, 18: {lenMap(0)}, 20: {lenMap(0)}, 22: {lenMap(0)}, 24: {lenMap(0)}},
	})
}

type OneofScalars struct {
	Union isOneofScalars_Union `protobuf_oneof:"union"`
}

var oneofScalarsType = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(OneofScalars)), Desc: mustMakeMessageDesc("oneof-scalars.proto", protoreflect.Proto2, "", `
		name: "OneofScalars"
		field: [
			{name:"f1"  number:1  label:LABEL_OPTIONAL type:TYPE_BOOL   default_value:"true" oneof_index:0},
			{name:"f2"  number:2  label:LABEL_OPTIONAL type:TYPE_INT32  default_value:"2"    oneof_index:0},
			{name:"f3"  number:3  label:LABEL_OPTIONAL type:TYPE_INT64  default_value:"3"    oneof_index:0},
			{name:"f4"  number:4  label:LABEL_OPTIONAL type:TYPE_UINT32 default_value:"4"    oneof_index:0},
			{name:"f5"  number:5  label:LABEL_OPTIONAL type:TYPE_UINT64 default_value:"5"    oneof_index:0},
			{name:"f6"  number:6  label:LABEL_OPTIONAL type:TYPE_FLOAT  default_value:"6"    oneof_index:0},
			{name:"f7"  number:7  label:LABEL_OPTIONAL type:TYPE_DOUBLE default_value:"7"    oneof_index:0},
			{name:"f8"  number:8  label:LABEL_OPTIONAL type:TYPE_STRING default_value:"8"    oneof_index:0},
			{name:"f9"  number:9  label:LABEL_OPTIONAL type:TYPE_STRING default_value:"9"    oneof_index:0},
			{name:"f10" number:10 label:LABEL_OPTIONAL type:TYPE_STRING default_value:"10"   oneof_index:0},
			{name:"f11" number:11 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"11"   oneof_index:0},
			{name:"f12" number:12 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"12"   oneof_index:0},
			{name:"f13" number:13 label:LABEL_OPTIONAL type:TYPE_BYTES  default_value:"13"   oneof_index:0}
		]
		oneof_decl: [{name:"union"}]
	`, nil),
}

func (m *OneofScalars) ProtoReflect() protoreflect.Message { return oneofScalarsType.MessageOf(m) }

func (*OneofScalars) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*OneofScalars_Bool)(nil),
		(*OneofScalars_Int32)(nil),
		(*OneofScalars_Int64)(nil),
		(*OneofScalars_Uint32)(nil),
		(*OneofScalars_Uint64)(nil),
		(*OneofScalars_Float32)(nil),
		(*OneofScalars_Float64)(nil),
		(*OneofScalars_String)(nil),
		(*OneofScalars_StringA)(nil),
		(*OneofScalars_StringB)(nil),
		(*OneofScalars_Bytes)(nil),
		(*OneofScalars_BytesA)(nil),
		(*OneofScalars_BytesB)(nil),
	}
}

type (
	isOneofScalars_Union interface {
		isOneofScalars_Union()
	}
	OneofScalars_Bool struct {
		Bool bool `protobuf:"1"`
	}
	OneofScalars_Int32 struct {
		Int32 MyInt32 `protobuf:"2"`
	}
	OneofScalars_Int64 struct {
		Int64 int64 `protobuf:"3"`
	}
	OneofScalars_Uint32 struct {
		Uint32 MyUint32 `protobuf:"4"`
	}
	OneofScalars_Uint64 struct {
		Uint64 uint64 `protobuf:"5"`
	}
	OneofScalars_Float32 struct {
		Float32 MyFloat32 `protobuf:"6"`
	}
	OneofScalars_Float64 struct {
		Float64 float64 `protobuf:"7"`
	}
	OneofScalars_String struct {
		String string `protobuf:"8"`
	}
	OneofScalars_StringA struct {
		StringA []byte `protobuf:"9"`
	}
	OneofScalars_StringB struct {
		StringB MyString `protobuf:"10"`
	}
	OneofScalars_Bytes struct {
		Bytes []byte `protobuf:"11"`
	}
	OneofScalars_BytesA struct {
		BytesA string `protobuf:"12"`
	}
	OneofScalars_BytesB struct {
		BytesB MyBytes `protobuf:"13"`
	}
)

func (*OneofScalars_Bool) isOneofScalars_Union()    {}
func (*OneofScalars_Int32) isOneofScalars_Union()   {}
func (*OneofScalars_Int64) isOneofScalars_Union()   {}
func (*OneofScalars_Uint32) isOneofScalars_Union()  {}
func (*OneofScalars_Uint64) isOneofScalars_Union()  {}
func (*OneofScalars_Float32) isOneofScalars_Union() {}
func (*OneofScalars_Float64) isOneofScalars_Union() {}
func (*OneofScalars_String) isOneofScalars_Union()  {}
func (*OneofScalars_StringA) isOneofScalars_Union() {}
func (*OneofScalars_StringB) isOneofScalars_Union() {}
func (*OneofScalars_Bytes) isOneofScalars_Union()   {}
func (*OneofScalars_BytesA) isOneofScalars_Union()  {}
func (*OneofScalars_BytesB) isOneofScalars_Union()  {}

func TestOneofs(t *testing.T) {
	empty := &OneofScalars{}
	want1 := &OneofScalars{Union: &OneofScalars_Bool{true}}
	want2 := &OneofScalars{Union: &OneofScalars_Int32{20}}
	want3 := &OneofScalars{Union: &OneofScalars_Int64{30}}
	want4 := &OneofScalars{Union: &OneofScalars_Uint32{40}}
	want5 := &OneofScalars{Union: &OneofScalars_Uint64{50}}
	want6 := &OneofScalars{Union: &OneofScalars_Float32{60}}
	want7 := &OneofScalars{Union: &OneofScalars_Float64{70}}
	want8 := &OneofScalars{Union: &OneofScalars_String{string("80")}}
	want9 := &OneofScalars{Union: &OneofScalars_StringA{[]byte("90")}}
	want10 := &OneofScalars{Union: &OneofScalars_StringB{MyString("100")}}
	want11 := &OneofScalars{Union: &OneofScalars_Bytes{[]byte("110")}}
	want12 := &OneofScalars{Union: &OneofScalars_BytesA{string("120")}}
	want13 := &OneofScalars{Union: &OneofScalars_BytesB{MyBytes("130")}}

	testMessage(t, nil, new(OneofScalars).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false},
		getFields{1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V(string("10")), 11: V([]byte("11")), 12: V([]byte("12")), 13: V([]byte("13"))},
		whichOneofs{"union": 0},

		setFields{1: V(bool(true))}, hasFields{1: true}, equalMessage{want1.ProtoReflect()},
		setFields{2: V(int32(20))}, hasFields{2: true}, equalMessage{want2.ProtoReflect()},
		setFields{3: V(int64(30))}, hasFields{3: true}, equalMessage{want3.ProtoReflect()},
		setFields{4: V(uint32(40))}, hasFields{4: true}, equalMessage{want4.ProtoReflect()},
		setFields{5: V(uint64(50))}, hasFields{5: true}, equalMessage{want5.ProtoReflect()},
		setFields{6: V(float32(60))}, hasFields{6: true}, equalMessage{want6.ProtoReflect()},
		setFields{7: V(float64(70))}, hasFields{7: true}, equalMessage{want7.ProtoReflect()},

		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: true, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false},
		whichOneofs{"union": 7},

		setFields{8: V(string("80"))}, hasFields{8: true}, equalMessage{want8.ProtoReflect()},
		setFields{9: V(string("90"))}, hasFields{9: true}, equalMessage{want9.ProtoReflect()},
		setFields{10: V(string("100"))}, hasFields{10: true}, equalMessage{want10.ProtoReflect()},
		setFields{11: V([]byte("110"))}, hasFields{11: true}, equalMessage{want11.ProtoReflect()},
		setFields{12: V([]byte("120"))}, hasFields{12: true}, equalMessage{want12.ProtoReflect()},
		setFields{13: V([]byte("130"))}, hasFields{13: true}, equalMessage{want13.ProtoReflect()},

		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: true},
		getFields{1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V(string("10")), 11: V([]byte("11")), 12: V([]byte("12")), 13: V([]byte("130"))},
		clearFields{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		whichOneofs{"union": 13},
		equalMessage{want13.ProtoReflect()},
		clearFields{13},
		whichOneofs{"union": 0},
		equalMessage{empty.ProtoReflect()},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*OneofScalars)(nil).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false, 13: false},
		getFields{1: V(bool(true)), 2: V(int32(2)), 3: V(int64(3)), 4: V(uint32(4)), 5: V(uint64(5)), 6: V(float32(6)), 7: V(float64(7)), 8: V(string("8")), 9: V(string("9")), 10: V(string("10")), 11: V([]byte("11")), 12: V([]byte("12")), 13: V([]byte("13"))},
	})
}

type EnumProto2 int32

var enumProto2Desc = mustMakeEnumDesc("enum2.proto", protoreflect.Proto2, `
	name:  "EnumProto2"
	value: [{name:"DEAD" number:0xdead}, {name:"BEEF" number:0xbeef}]
`)

func (e EnumProto2) Descriptor() protoreflect.EnumDescriptor         { return enumProto2Desc }
func (e EnumProto2) Type() protoreflect.EnumType                     { return e }
func (e EnumProto2) Enum() *EnumProto2                               { return &e }
func (e EnumProto2) Number() protoreflect.EnumNumber                 { return protoreflect.EnumNumber(e) }
func (t EnumProto2) New(n protoreflect.EnumNumber) protoreflect.Enum { return EnumProto2(n) }

type EnumProto3 int32

var enumProto3Desc = mustMakeEnumDesc("enum3.proto", protoreflect.Proto3, `
	name:  "EnumProto3",
	value: [{name:"ALPHA" number:0}, {name:"BRAVO" number:1}]
`)

func (e EnumProto3) Descriptor() protoreflect.EnumDescriptor         { return enumProto3Desc }
func (e EnumProto3) Type() protoreflect.EnumType                     { return e }
func (e EnumProto3) Enum() *EnumProto3                               { return &e }
func (e EnumProto3) Number() protoreflect.EnumNumber                 { return protoreflect.EnumNumber(e) }
func (t EnumProto3) New(n protoreflect.EnumNumber) protoreflect.Enum { return EnumProto3(n) }

type EnumMessages struct {
	EnumP2        *EnumProto2              `protobuf:"1"`
	EnumP3        *EnumProto3              `protobuf:"2"`
	MessageLegacy *proto2_20180125.Message `protobuf:"3"`
	MessageCycle  *EnumMessages            `protobuf:"4"`
	EnumList      []EnumProto2             `protobuf:"5"`
	MessageList   []*ScalarProto2          `protobuf:"6"`
	EnumMap       map[string]EnumProto3    `protobuf:"7"`
	MessageMap    map[string]*ScalarProto3 `protobuf:"8"`
	Union         isEnumMessages_Union     `protobuf_oneof:"union"`
}

var enumMessagesType = pimpl.MessageInfo{GoReflectType: reflect.TypeOf(new(EnumMessages)), Desc: mustMakeMessageDesc("enum-messages.proto", protoreflect.Proto2, `
		dependency: ["enum2.proto", "enum3.proto", "scalar2.proto", "scalar3.proto", "proto2_20180125_92554152/test.proto"]
	`, `
		name: "EnumMessages"
		field: [
			{name:"f1"  number:1  label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".EnumProto2" default_value:"BEEF"},
			{name:"f2"  number:2  label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".EnumProto3" default_value:"BRAVO"},
			{name:"f3"  number:3  label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".google.golang.org.proto2_20180125.Message"},
			{name:"f4"  number:4  label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".EnumMessages"},
			{name:"f5"  number:5  label:LABEL_REPEATED type:TYPE_ENUM    type_name:".EnumProto2"},
			{name:"f6"  number:6  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".ScalarProto2"},
			{name:"f7"  number:7  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".EnumMessages.F7Entry"},
			{name:"f8"  number:8  label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".EnumMessages.F8Entry"},
			{name:"f9"  number:9  label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".EnumProto2"   oneof_index:0 default_value:"BEEF"},
			{name:"f10" number:10 label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".EnumProto3"   oneof_index:0 default_value:"BRAVO"},
			{name:"f11" number:11 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".ScalarProto2" oneof_index:0},
			{name:"f12" number:12 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".ScalarProto3" oneof_index:0}
		]
		oneof_decl: [{name:"union"}]
		nested_type: [
			{name:"F7Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".EnumProto3"}]   options:{map_entry:true}},
			{name:"F8Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".ScalarProto3"}] options:{map_entry:true}}
		]
	`, newFileRegistry(
	EnumProto2(0).Descriptor().ParentFile(),
	EnumProto3(0).Descriptor().ParentFile(),
	((*ScalarProto2)(nil)).ProtoReflect().Descriptor().ParentFile(),
	((*ScalarProto3)(nil)).ProtoReflect().Descriptor().ParentFile(),
	pimpl.Export{}.MessageDescriptorOf((*proto2_20180125.Message)(nil)).ParentFile(),
)),
}

func newFileRegistry(files ...protoreflect.FileDescriptor) *protoregistry.Files {
	r := new(protoregistry.Files)
	for _, file := range files {
		r.RegisterFile(file)
	}
	return r
}

func (m *EnumMessages) ProtoReflect() protoreflect.Message { return enumMessagesType.MessageOf(m) }

func (*EnumMessages) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*EnumMessages_OneofE2)(nil),
		(*EnumMessages_OneofE3)(nil),
		(*EnumMessages_OneofM2)(nil),
		(*EnumMessages_OneofM3)(nil),
	}
}

type (
	isEnumMessages_Union interface {
		isEnumMessages_Union()
	}
	EnumMessages_OneofE2 struct {
		OneofE2 EnumProto2 `protobuf:"9"`
	}
	EnumMessages_OneofE3 struct {
		OneofE3 EnumProto3 `protobuf:"10"`
	}
	EnumMessages_OneofM2 struct {
		OneofM2 *ScalarProto2 `protobuf:"11"`
	}
	EnumMessages_OneofM3 struct {
		OneofM3 *ScalarProto3 `protobuf:"12"`
	}
)

func (*EnumMessages_OneofE2) isEnumMessages_Union() {}
func (*EnumMessages_OneofE3) isEnumMessages_Union() {}
func (*EnumMessages_OneofM2) isEnumMessages_Union() {}
func (*EnumMessages_OneofM3) isEnumMessages_Union() {}

func TestEnumMessages(t *testing.T) {
	emptyL := pimpl.Export{}.MessageOf(new(proto2_20180125.Message))
	emptyM := new(EnumMessages).ProtoReflect()
	emptyM2 := new(ScalarProto2).ProtoReflect()
	emptyM3 := new(ScalarProto3).ProtoReflect()

	wantL := pimpl.Export{}.MessageOf(&proto2_20180125.Message{OptionalFloat: proto.Float32(math.E)})
	wantM := (&EnumMessages{EnumP2: EnumProto2(1234).Enum()}).ProtoReflect()
	wantM2a := &ScalarProto2{Float32: proto.Float32(math.Pi)}
	wantM2b := &ScalarProto2{Float32: proto.Float32(math.Phi)}
	wantM3a := &ScalarProto3{Float32: math.Pi}
	wantM3b := &ScalarProto3{Float32: math.Ln2}

	wantList5 := getField((&EnumMessages{EnumList: []EnumProto2{333, 222}}).ProtoReflect(), 5)
	wantList6 := getField((&EnumMessages{MessageList: []*ScalarProto2{wantM2a, wantM2b}}).ProtoReflect(), 6)

	wantMap7 := getField((&EnumMessages{EnumMap: map[string]EnumProto3{"one": 1, "two": 2}}).ProtoReflect(), 7)
	wantMap8 := getField((&EnumMessages{MessageMap: map[string]*ScalarProto3{"pi": wantM3a, "ln2": wantM3b}}).ProtoReflect(), 8)

	testMessage(t, nil, new(EnumMessages).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false},
		getFields{1: VE(0xbeef), 2: VE(1), 3: V(emptyL), 4: V(emptyM), 9: VE(0xbeef), 10: VE(1)},

		// Test singular enums.
		setFields{1: VE(0xdead), 2: VE(0)},
		getFields{1: VE(0xdead), 2: VE(0)},
		hasFields{1: true, 2: true},

		// Test singular messages.
		messageFieldsMutable{3: messageOps{setFields{109: V(float32(math.E))}}},
		messageFieldsMutable{4: messageOps{setFields{1: VE(1234)}}},
		getFields{3: V(wantL), 4: V(wantM)},
		clearFields{3, 4},
		hasFields{3: false, 4: false},
		setFields{3: V(wantL), 4: V(wantM)},
		hasFields{3: true, 4: true},

		// Test list of enums and messages.
		listFieldsMutable{
			5: listOps{
				appendList{VE(111), VE(222)},
				setList{0: VE(333)},
				getList{0: VE(333), 1: VE(222)},
				lenList(2),
			},
			6: listOps{
				appendMessageList{setFields{4: V(uint32(1e6))}},
				appendMessageList{setFields{6: V(float32(math.Phi))}},
				setList{0: V(wantM2a.ProtoReflect())},
				getList{0: V(wantM2a.ProtoReflect()), 1: V(wantM2b.ProtoReflect())},
			},
		},
		getFields{5: wantList5, 6: wantList6},
		hasFields{5: true, 6: true},
		listFields{5: listOps{truncList(0)}},
		hasFields{5: false, 6: true},

		// Test maps of enums and messages.
		mapFieldsMutable{
			7: mapOps{
				setMap{"one": VE(1), "two": VE(2)},
				hasMap{"one": true, "two": true, "three": false},
				lenMap(2),
			},
			8: mapOps{
				messageMap{"pi": messageOps{setFields{6: V(float32(math.Pi))}}},
				setMap{"ln2": V(wantM3b.ProtoReflect())},
				getMap{"pi": V(wantM3a.ProtoReflect()), "ln2": V(wantM3b.ProtoReflect()), "none": V(nil)},
				lenMap(2),
			},
		},
		getFields{7: wantMap7, 8: wantMap8},
		hasFields{7: true, 8: true},
		mapFields{8: mapOps{clearMap{"pi", "ln2", "none"}}},
		hasFields{7: true, 8: false},

		// Test oneofs of enums and messages.
		setFields{9: VE(0xdead)},
		hasFields{1: true, 2: true, 9: true, 10: false, 11: false, 12: false},
		setFields{10: VE(0)},
		hasFields{1: true, 2: true, 9: false, 10: true, 11: false, 12: false},
		messageFieldsMutable{11: messageOps{setFields{6: V(float32(math.Pi))}}},
		getFields{11: V(wantM2a.ProtoReflect())},
		hasFields{1: true, 2: true, 9: false, 10: false, 11: true, 12: false},
		messageFieldsMutable{12: messageOps{setFields{6: V(float32(math.Pi))}}},
		getFields{12: V(wantM3a.ProtoReflect())},
		hasFields{1: true, 2: true, 9: false, 10: false, 11: false, 12: true},

		// Check entire message.
		rangeFields{1: VE(0xdead), 2: VE(0), 3: V(wantL), 4: V(wantM), 6: wantList6, 7: wantMap7, 12: V(wantM3a.ProtoReflect())},
		equalMessage{(&EnumMessages{
			EnumP2:        EnumProto2(0xdead).Enum(),
			EnumP3:        EnumProto3(0).Enum(),
			MessageLegacy: &proto2_20180125.Message{OptionalFloat: proto.Float32(math.E)},
			MessageCycle:  wantM.Interface().(*EnumMessages),
			MessageList:   []*ScalarProto2{wantM2a, wantM2b},
			EnumMap:       map[string]EnumProto3{"one": 1, "two": 2},
			Union:         &EnumMessages_OneofM3{wantM3a},
		}).ProtoReflect()},
		clearFields{1, 2, 3, 4, 6, 7, 12},
		equalMessage{new(EnumMessages).ProtoReflect()},
	})

	// Test read-only operations on nil message.
	testMessage(t, nil, (*EnumMessages)(nil).ProtoReflect(), messageOps{
		hasFields{1: false, 2: false, 3: false, 4: false, 5: false, 6: false, 7: false, 8: false, 9: false, 10: false, 11: false, 12: false},
		getFields{1: VE(0xbeef), 2: VE(1), 3: V(emptyL), 4: V(emptyM), 9: VE(0xbeef), 10: VE(1), 11: V(emptyM2), 12: V(emptyM3)},
		listFields{5: {lenList(0)}, 6: {lenList(0)}},
		mapFields{7: {lenMap(0)}, 8: {lenMap(0)}},
	})
}

var cmpOpts = cmp.Options{
	cmp.Comparer(func(x, y *proto2_20180125.Message) bool {
		mx := pimpl.Export{}.MessageOf(x).Interface()
		my := pimpl.Export{}.MessageOf(y).Interface()
		return proto.Equal(mx, my)
	}),
	cmp.Transformer("UnwrapValue", func(pv protoreflect.Value) interface{} {
		switch v := pv.Interface().(type) {
		case protoreflect.Message:
			out := make(map[protoreflect.FieldNumber]protoreflect.Value)
			v.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
				out[fd.Number()] = v
				return true
			})
			return out
		case protoreflect.List:
			var out []protoreflect.Value
			for i := 0; i < v.Len(); i++ {
				out = append(out, v.Get(i))
			}
			return out
		case protoreflect.Map:
			out := make(map[interface{}]protoreflect.Value)
			v.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
				out[k.Interface()] = v
				return true
			})
			return out
		default:
			return v
		}
	}),
	cmpopts.EquateNaNs(),
}

func testMessage(t *testing.T, p path, m protoreflect.Message, tt messageOps) {
	fieldDescs := m.Descriptor().Fields()
	oneofDescs := m.Descriptor().Oneofs()
	for i, op := range tt {
		p.Push(i)
		switch op := op.(type) {
		case equalMessage:
			if diff := cmp.Diff(V(op.Message), V(m), cmpOpts); diff != "" {
				t.Errorf("operation %v, message mismatch (-want, +got):\n%s", p, diff)
			}
		case hasFields:
			got := map[protoreflect.FieldNumber]bool{}
			want := map[protoreflect.FieldNumber]bool(op)
			for n := range want {
				fd := fieldDescs.ByNumber(n)
				got[n] = m.Has(fd)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("operation %v, Message.Has mismatch (-want, +got):\n%s", p, diff)
			}
		case getFields:
			got := map[protoreflect.FieldNumber]protoreflect.Value{}
			want := map[protoreflect.FieldNumber]protoreflect.Value(op)
			for n := range want {
				fd := fieldDescs.ByNumber(n)
				got[n] = m.Get(fd)
			}
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, Message.Get mismatch (-want, +got):\n%s", p, diff)
			}
		case setFields:
			for n, v := range op {
				fd := fieldDescs.ByNumber(n)
				m.Set(fd, v)
			}
		case clearFields:
			for _, n := range op {
				fd := fieldDescs.ByNumber(n)
				m.Clear(fd)
			}
		case whichOneofs:
			got := map[protoreflect.Name]protoreflect.FieldNumber{}
			want := map[protoreflect.Name]protoreflect.FieldNumber(op)
			for s := range want {
				od := oneofDescs.ByName(s)
				fd := m.WhichOneof(od)
				if fd == nil {
					got[s] = 0
				} else {
					got[s] = fd.Number()
				}
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("operation %v, Message.WhichOneof mismatch (-want, +got):\n%s", p, diff)
			}
		case messageFields:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testMessage(t, p, m.Get(fd).Message(), tt)
				p.Pop()
			}
		case messageFieldsMutable:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testMessage(t, p, m.Mutable(fd).Message(), tt)
				p.Pop()
			}
		case listFields:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testLists(t, p, m.Get(fd).List(), tt)
				p.Pop()
			}
		case listFieldsMutable:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testLists(t, p, m.Mutable(fd).List(), tt)
				p.Pop()
			}
		case mapFields:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testMaps(t, p, m.Get(fd).Map(), tt)
				p.Pop()
			}
		case mapFieldsMutable:
			for n, tt := range op {
				p.Push(int(n))
				fd := fieldDescs.ByNumber(n)
				testMaps(t, p, m.Mutable(fd).Map(), tt)
				p.Pop()
			}
		case rangeFields:
			got := map[protoreflect.FieldNumber]protoreflect.Value{}
			want := map[protoreflect.FieldNumber]protoreflect.Value(op)
			m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
				got[fd.Number()] = v
				return true
			})
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, Message.Range mismatch (-want, +got):\n%s", p, diff)
			}
		default:
			t.Fatalf("operation %v, invalid operation: %T", p, op)
		}
		p.Pop()
	}
}

func testLists(t *testing.T, p path, v protoreflect.List, tt listOps) {
	for i, op := range tt {
		p.Push(i)
		switch op := op.(type) {
		case equalList:
			if diff := cmp.Diff(V(op.List), V(v), cmpOpts); diff != "" {
				t.Errorf("operation %v, list mismatch (-want, +got):\n%s", p, diff)
			}
		case lenList:
			if got, want := v.Len(), int(op); got != want {
				t.Errorf("operation %v, List.Len = %d, want %d", p, got, want)
			}
		case getList:
			got := map[int]protoreflect.Value{}
			want := map[int]protoreflect.Value(op)
			for n := range want {
				got[n] = v.Get(n)
			}
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, List.Get mismatch (-want, +got):\n%s", p, diff)
			}
		case setList:
			for n, e := range op {
				v.Set(n, e)
			}
		case appendList:
			for _, e := range op {
				v.Append(e)
			}
		case appendMessageList:
			e := v.NewElement()
			v.Append(e)
			testMessage(t, p, e.Message(), messageOps(op))
		case truncList:
			v.Truncate(int(op))
		default:
			t.Fatalf("operation %v, invalid operation: %T", p, op)
		}
		p.Pop()
	}
}

func testMaps(t *testing.T, p path, m protoreflect.Map, tt mapOps) {
	for i, op := range tt {
		p.Push(i)
		switch op := op.(type) {
		case equalMap:
			if diff := cmp.Diff(V(op.Map), V(m), cmpOpts); diff != "" {
				t.Errorf("operation %v, map mismatch (-want, +got):\n%s", p, diff)
			}
		case lenMap:
			if got, want := m.Len(), int(op); got != want {
				t.Errorf("operation %v, Map.Len = %d, want %d", p, got, want)
			}
		case hasMap:
			got := map[interface{}]bool{}
			want := map[interface{}]bool(op)
			for k := range want {
				got[k] = m.Has(V(k).MapKey())
			}
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, Map.Has mismatch (-want, +got):\n%s", p, diff)
			}
		case getMap:
			got := map[interface{}]protoreflect.Value{}
			want := map[interface{}]protoreflect.Value(op)
			for k := range want {
				got[k] = m.Get(V(k).MapKey())
			}
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, Map.Get mismatch (-want, +got):\n%s", p, diff)
			}
		case setMap:
			for k, v := range op {
				m.Set(V(k).MapKey(), v)
			}
		case clearMap:
			for _, k := range op {
				m.Clear(V(k).MapKey())
			}
		case messageMap:
			for k, tt := range op {
				mk := V(k).MapKey()
				if !m.Has(mk) {
					m.Set(mk, m.NewValue())
				}
				testMessage(t, p, m.Get(mk).Message(), tt)
			}
		case rangeMap:
			got := map[interface{}]protoreflect.Value{}
			want := map[interface{}]protoreflect.Value(op)
			m.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
				got[k.Interface()] = v
				return true
			})
			if diff := cmp.Diff(want, got, cmpOpts); diff != "" {
				t.Errorf("operation %v, Map.Range mismatch (-want, +got):\n%s", p, diff)
			}
		default:
			t.Fatalf("operation %v, invalid operation: %T", p, op)
		}
		p.Pop()
	}
}

func getField(m protoreflect.Message, n protoreflect.FieldNumber) protoreflect.Value {
	fd := m.Descriptor().Fields().ByNumber(n)
	return m.Get(fd)
}

type path []int

func (p *path) Push(i int) { *p = append(*p, i) }
func (p *path) Pop()       { *p = (*p)[:len(*p)-1] }
func (p path) String() string {
	var ss []string
	for _, i := range p {
		ss = append(ss, fmt.Sprint(i))
	}
	return strings.Join(ss, ".")
}

type UnknownFieldsA struct {
	XXX_unrecognized []byte
}

var unknownFieldsAType = pimpl.MessageInfo{
	GoReflectType: reflect.TypeOf(new(UnknownFieldsA)),
	Desc:          mustMakeMessageDesc("unknown.proto", protoreflect.Proto2, "", `name: "UnknownFieldsA"`, nil),
}

func (m *UnknownFieldsA) ProtoReflect() protoreflect.Message { return unknownFieldsAType.MessageOf(m) }

type UnknownFieldsB struct {
	XXX_unrecognized *[]byte
}

var unknownFieldsBType = pimpl.MessageInfo{
	GoReflectType: reflect.TypeOf(new(UnknownFieldsB)),
	Desc:          mustMakeMessageDesc("unknown.proto", protoreflect.Proto2, "", `name: "UnknownFieldsB"`, nil),
}

func (m *UnknownFieldsB) ProtoReflect() protoreflect.Message { return unknownFieldsBType.MessageOf(m) }

func TestUnknownFields(t *testing.T) {
	for _, m := range []proto.Message{new(UnknownFieldsA), new(UnknownFieldsB)} {
		t.Run(reflect.TypeOf(m).Elem().Name(), func(t *testing.T) {
			want := protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("Hello, world!"),
			}.Marshal()
			m.ProtoReflect().SetUnknown(want)
			got := []byte(m.ProtoReflect().GetUnknown())
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("UnknownFields mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestReset(t *testing.T) {
	mi := new(testpb.TestAllTypes)

	// ProtoReflect is implemented using a messageState cache.
	m := mi.ProtoReflect()

	// Reset must not clear the messageState cache.
	mi.Reset()

	// If Reset accidentally cleared the messageState cache, this panics.
	m.Descriptor()
}

func TestIsValid(t *testing.T) {
	var m *testpb.TestAllTypes
	if got, want := m.ProtoReflect().IsValid(), false; got != want {
		t.Errorf("((*M)(nil)).ProtoReflect().IsValid() = %v, want %v", got, want)
	}
	m = &testpb.TestAllTypes{}
	if got, want := m.ProtoReflect().IsValid(), true; got != want {
		t.Errorf("(&M{}).ProtoReflect().IsValid() = %v, want %v", got, want)
	}
}

// The MessageState implementation makes the assumption that when a
// concrete message is unsafe casted as a *MessageState, the Go GC does
// not reclaim the memory for the remainder of the concrete message.
func TestUnsafeAssumptions(t *testing.T) {
	if !pimpl.UnsafeEnabled {
		t.Skip()
	}

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			var ms [10]protoreflect.Message

			// Store the message only in its reflective form.
			// Trigger the GC after each iteration.
			for j := 0; j < 10; j++ {
				ms[j] = (&testpb.TestAllTypes{
					OptionalInt32: proto.Int32(int32(j)),
					OptionalFloat: proto.Float32(float32(j)),
					RepeatedInt32: []int32{int32(j)},
					RepeatedFloat: []float32{float32(j)},
					DefaultInt32:  proto.Int32(int32(j)),
					DefaultFloat:  proto.Float32(float32(j)),
				}).ProtoReflect()
				runtime.GC()
			}

			// Convert the reflective form back into a concrete form.
			// Verify that the values written previously are still the same.
			for j := 0; j < 10; j++ {
				switch m := ms[j].Interface().(*testpb.TestAllTypes); {
				case m.GetOptionalInt32() != int32(j):
				case m.GetOptionalFloat() != float32(j):
				case m.GetRepeatedInt32()[0] != int32(j):
				case m.GetRepeatedFloat()[0] != float32(j):
				case m.GetDefaultInt32() != int32(j):
				case m.GetDefaultFloat() != float32(j):
				default:
					continue
				}
				t.Error("memory corrupted detected")
			}
			defer wg.Done()
		}()
	}
	wg.Wait()
}

func BenchmarkName(b *testing.B) {
	var sink protoreflect.FullName
	b.Run("Value", func(b *testing.B) {
		b.ReportAllocs()
		m := new(descriptorpb.FileDescriptorProto)
		for i := 0; i < b.N; i++ {
			sink = m.ProtoReflect().Descriptor().FullName()
		}
	})
	b.Run("Nil", func(b *testing.B) {
		b.ReportAllocs()
		m := (*descriptorpb.FileDescriptorProto)(nil)
		for i := 0; i < b.N; i++ {
			sink = m.ProtoReflect().Descriptor().FullName()
		}
	})
	runtime.KeepAlive(sink)
}

func BenchmarkReflect(b *testing.B) {
	m := new(testpb.TestAllTypes).ProtoReflect()
	fds := m.Descriptor().Fields()
	vs := make([]protoreflect.Value, fds.Len())
	for i := range vs {
		vs[i] = m.NewField(fds.Get(i))
	}

	b.Run("Has", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < fds.Len(); j++ {
				m.Has(fds.Get(j))
			}
		}
	})
	b.Run("Get", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < fds.Len(); j++ {
				m.Get(fds.Get(j))
			}
		}
	})
	b.Run("Set", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < fds.Len(); j++ {
				m.Set(fds.Get(j), vs[j])
			}
		}
	})
	b.Run("Clear", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := 0; j < fds.Len(); j++ {
				m.Clear(fds.Get(j))
			}
		}
	})
	b.Run("Range", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			m.Range(func(protoreflect.FieldDescriptor, protoreflect.Value) bool {
				return true
			})
		}
	})
}
