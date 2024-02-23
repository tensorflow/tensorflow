// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl_test

import (
	"io"
	"reflect"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
	"google.golang.org/protobuf/testing/protocmp"

	"google.golang.org/protobuf/types/descriptorpb"
)

type AberrantMessage struct {
	OptionalBool     *bool            `protobuf:"varint,1,opt,name=opt_bool,def=1"`
	OptionalInt32    *int32           `protobuf:"varint,2,opt,name=opt_int32,def=-12345"`
	OptionalSint32   *int32           `protobuf:"zigzag32,3,opt,name=opt_sint32,def=-3200"`
	OptionalUint32   *uint32          `protobuf:"varint,4,opt,name=opt_uint32,def=3200"`
	OptionalInt64    *int64           `protobuf:"varint,5,opt,name=opt_int64,def=-123456789"`
	OptionalSint64   *int64           `protobuf:"zigzag64,6,opt,name=opt_sint64,def=-6400"`
	OptionalUint64   *uint64          `protobuf:"varint,7,opt,name=opt_uint64,def=6400"`
	OptionalFixed32  *uint32          `protobuf:"fixed32,8,opt,name=opt_fixed32,def=320000"`
	OptionalSfixed32 *int32           `protobuf:"fixed32,9,opt,name=opt_sfixed32,def=-320000"`
	OptionalFloat    *float32         `protobuf:"fixed32,10,opt,name=opt_float,def=3.14159"`
	OptionalFixed64  *uint64          `protobuf:"fixed64,11,opt,name=opt_fixed64,def=640000"`
	OptionalSfixed64 *int64           `protobuf:"fixed64,12,opt,name=opt_sfixed64,def=-640000"`
	OptionalDouble   *float64         `protobuf:"fixed64,13,opt,name=opt_double,def=3.14159265359"`
	OptionalString   *string          `protobuf:"bytes,14,opt,name=opt_string,def=hello, \"world!\"\n"`
	OptionalBytes    []byte           `protobuf:"bytes,15,opt,name=opt_bytes,def=dead\\336\\255\\276\\357beef"`
	OptionalEnum     *AberrantEnum    `protobuf:"varint,16,opt,name=opt_enum,enum=google.golang.org.example.AberrantEnum,def=0"`
	OptionalMessage  *AberrantMessage `protobuf:"bytes,17,opt,name=opt_message"`

	RepeatedBool     []bool             `protobuf:"varint,18,rep,packed,name=rep_bool"`
	RepeatedInt32    []int32            `protobuf:"varint,19,rep,packed,name=rep_int32"`
	RepeatedSint32   []int32            `protobuf:"zigzag32,20,rep,packed,name=rep_sint32"`
	RepeatedUint32   []uint32           `protobuf:"varint,21,rep,packed,name=rep_uint32"`
	RepeatedInt64    []int64            `protobuf:"varint,22,rep,packed,name=rep_int64"`
	RepeatedSint64   []int64            `protobuf:"zigzag64,23,rep,packed,name=rep_sint64"`
	RepeatedUint64   []uint64           `protobuf:"varint,24,rep,packed,name=rep_uint64"`
	RepeatedFixed32  []uint32           `protobuf:"fixed32,25,rep,packed,name=rep_fixed32"`
	RepeatedSfixed32 []int32            `protobuf:"fixed32,26,rep,packed,name=rep_sfixed32"`
	RepeatedFloat    []float32          `protobuf:"fixed32,27,rep,packed,name=rep_float"`
	RepeatedFixed64  []uint64           `protobuf:"fixed64,28,rep,packed,name=rep_fixed64"`
	RepeatedSfixed64 []int64            `protobuf:"fixed64,29,rep,packed,name=rep_sfixed64"`
	RepeatedDouble   []float64          `protobuf:"fixed64,30,rep,packed,name=rep_double"`
	RepeatedString   []string           `protobuf:"bytes,31,rep,name=rep_string"`
	RepeatedBytes    [][]byte           `protobuf:"bytes,32,rep,name=rep_bytes"`
	RepeatedEnum     []AberrantEnum     `protobuf:"varint,33,rep,name=rep_enum,enum=google.golang.org.example.AberrantEnum"`
	RepeatedMessage  []*AberrantMessage `protobuf:"bytes,34,rep,name=rep_message"`

	MapStringBool     map[string]bool             `protobuf:"bytes,35,rep,name=map_string_bool"     protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapStringInt32    map[string]int32            `protobuf:"bytes,36,rep,name=map_string_int32"    protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapStringSint32   map[string]int32            `protobuf:"bytes,37,rep,name=map_string_sint32"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"zigzag32,2,opt,name=value"`
	MapStringUint32   map[string]uint32           `protobuf:"bytes,38,rep,name=map_string_uint32"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapStringInt64    map[string]int64            `protobuf:"bytes,39,rep,name=map_string_int64"    protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapStringSint64   map[string]int64            `protobuf:"bytes,40,rep,name=map_string_sint64"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"zigzag64,2,opt,name=value"`
	MapStringUint64   map[string]uint64           `protobuf:"bytes,41,rep,name=map_string_uint64"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapStringFixed32  map[string]uint32           `protobuf:"bytes,42,rep,name=map_string_fixed32"  protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed32,2,opt,name=value"`
	MapStringSfixed32 map[string]int32            `protobuf:"bytes,43,rep,name=map_string_sfixed32" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed32,2,opt,name=value"`
	MapStringFloat    map[string]float32          `protobuf:"bytes,44,rep,name=map_string_float"    protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed32,2,opt,name=value"`
	MapStringFixed64  map[string]uint64           `protobuf:"bytes,45,rep,name=map_string_fixed64"  protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed64,2,opt,name=value"`
	MapStringSfixed64 map[string]int64            `protobuf:"bytes,46,rep,name=map_string_sfixed64" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed64,2,opt,name=value"`
	MapStringDouble   map[string]float64          `protobuf:"bytes,47,rep,name=map_string_double"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed64,2,opt,name=value"`
	MapStringString   map[string]string           `protobuf:"bytes,48,rep,name=map_string_string"   protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	MapStringBytes    map[string][]byte           `protobuf:"bytes,49,rep,name=map_string_bytes"    protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	MapStringEnum     map[string]AberrantEnum     `protobuf:"bytes,50,rep,name=map_string_enum"     protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value,enum=google.golang.org.example.AberrantEnum"`
	MapStringMessage  map[string]*AberrantMessage `protobuf:"bytes,51,rep,name=map_string_message"  protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`

	OneofUnion isOneofUnion `protobuf_oneof:"oneof_union"`

	Ignored io.Reader
}

func (m *AberrantMessage) ExtensionRangeArray() []protoiface.ExtensionRangeV1 {
	return []protoiface.ExtensionRangeV1{{Start: 10, End: 100}}
}

func (m *AberrantMessage) XXX_OneofFuncs() []interface{} {
	return []interface{}{
		(*OneofBool)(nil),
		(*OneofInt32)(nil),
		(*OneofSint32)(nil),
		(*OneofUint32)(nil),
		(*OneofInt64)(nil),
		(*OneofSint64)(nil),
		(*OneofUint64)(nil),
		(*OneofFixed32)(nil),
		(*OneofSfixed32)(nil),
		(*OneofFloat)(nil),
		(*OneofFixed64)(nil),
		(*OneofSfixed64)(nil),
		(*OneofDouble)(nil),
		(*OneofString)(nil),
		(*OneofBytes)(nil),
		(*OneofEnum)(nil),
		(*OneofMessage)(nil),
	}
}

type isOneofUnion interface{ isOneofUnion() }

type OneofBool struct {
	OneofBool bool `protobuf:"varint,52,opt,name=oneof_bool,oneof,def=1"`
}
type OneofInt32 struct {
	OneofInt32 int32 `protobuf:"varint,53,opt,name=oneof_int32,oneof,def=-12345"`
}
type OneofSint32 struct {
	OneofSint32 int32 `protobuf:"zigzag32,54,opt,name=oneof_sint32,oneof,def=-3200"`
}
type OneofUint32 struct {
	OneofUint32 uint32 `protobuf:"varint,55,opt,name=oneof_uint32,oneof,def=3200"`
}
type OneofInt64 struct {
	OneofInt64 int64 `protobuf:"varint,56,opt,name=oneof_int64,oneof,def=-123456789"`
}
type OneofSint64 struct {
	OneofSint64 int64 `protobuf:"zigzag64,57,opt,name=oneof_sint64,oneof,def=-6400"`
}
type OneofUint64 struct {
	OneofUint64 uint64 `protobuf:"varint,58,opt,name=oneof_uint64,oneof,def=6400"`
}
type OneofFixed32 struct {
	OneofFixed32 uint32 `protobuf:"fixed32,59,opt,name=oneof_fixed32,oneof,def=320000"`
}
type OneofSfixed32 struct {
	OneofSfixed32 int32 `protobuf:"fixed32,60,opt,name=oneof_sfixed32,oneof,def=-320000"`
}
type OneofFloat struct {
	OneofFloat float32 `protobuf:"fixed32,61,opt,name=oneof_float,oneof,def=3.14159"`
}
type OneofFixed64 struct {
	OneofFixed64 uint64 `protobuf:"fixed64,62,opt,name=oneof_fixed64,oneof,def=640000"`
}
type OneofSfixed64 struct {
	OneofSfixed64 int64 `protobuf:"fixed64,63,opt,name=oneof_sfixed64,oneof,def=-640000"`
}
type OneofDouble struct {
	OneofDouble float64 `protobuf:"fixed64,64,opt,name=oneof_double,oneof,def=3.14159265359"`
}
type OneofString struct {
	OneofString string `protobuf:"bytes,65,opt,name=oneof_string,oneof,def=hello, \"world!\"\n"`
}
type OneofBytes struct {
	OneofBytes []byte `protobuf:"bytes,66,opt,name=oneof_bytes,oneof,def=dead\\336\\255\\276\\357beef"`
}
type OneofEnum struct {
	OneofEnum AberrantEnum `protobuf:"varint,67,opt,name=oneof_enum,enum=google.golang.org.example.AberrantEnum,oneof,def=0"`
}
type OneofMessage struct {
	OneofMessage *AberrantMessage `protobuf:"bytes,68,opt,name=oneof_message,oneof"`
}

func (OneofBool) isOneofUnion()     {}
func (OneofInt32) isOneofUnion()    {}
func (OneofSint32) isOneofUnion()   {}
func (OneofUint32) isOneofUnion()   {}
func (OneofInt64) isOneofUnion()    {}
func (OneofSint64) isOneofUnion()   {}
func (OneofUint64) isOneofUnion()   {}
func (OneofFixed32) isOneofUnion()  {}
func (OneofSfixed32) isOneofUnion() {}
func (OneofFloat) isOneofUnion()    {}
func (OneofFixed64) isOneofUnion()  {}
func (OneofSfixed64) isOneofUnion() {}
func (OneofDouble) isOneofUnion()   {}
func (OneofString) isOneofUnion()   {}
func (OneofBytes) isOneofUnion()    {}
func (OneofEnum) isOneofUnion()     {}
func (OneofMessage) isOneofUnion()  {}

type AberrantEnum int32

func TestAberrantMessages(t *testing.T) {
	enumName := impl.AberrantDeriveFullName(reflect.TypeOf(AberrantEnum(0)))
	messageName := impl.AberrantDeriveFullName(reflect.TypeOf(AberrantMessage{}))

	want := new(descriptorpb.DescriptorProto)
	if err := prototext.Unmarshal([]byte(`
		name: "AberrantMessage"
		field: [
			{name:"opt_bool"     number:1  label:LABEL_OPTIONAL type:TYPE_BOOL     default_value:"true"},
			{name:"opt_int32"    number:2  label:LABEL_OPTIONAL type:TYPE_INT32    default_value:"-12345"},
			{name:"opt_sint32"   number:3  label:LABEL_OPTIONAL type:TYPE_SINT32   default_value:"-3200"},
			{name:"opt_uint32"   number:4  label:LABEL_OPTIONAL type:TYPE_UINT32   default_value:"3200"},
			{name:"opt_int64"    number:5  label:LABEL_OPTIONAL type:TYPE_INT64    default_value:"-123456789"},
			{name:"opt_sint64"   number:6  label:LABEL_OPTIONAL type:TYPE_SINT64   default_value:"-6400"},
			{name:"opt_uint64"   number:7  label:LABEL_OPTIONAL type:TYPE_UINT64   default_value:"6400"},
			{name:"opt_fixed32"  number:8  label:LABEL_OPTIONAL type:TYPE_FIXED32  default_value:"320000"},
			{name:"opt_sfixed32" number:9  label:LABEL_OPTIONAL type:TYPE_SFIXED32 default_value:"-320000"},
			{name:"opt_float"    number:10 label:LABEL_OPTIONAL type:TYPE_FLOAT    default_value:"3.14159"},
			{name:"opt_fixed64"  number:11 label:LABEL_OPTIONAL type:TYPE_FIXED64  default_value:"640000"},
			{name:"opt_sfixed64" number:12 label:LABEL_OPTIONAL type:TYPE_SFIXED64 default_value:"-640000"},
			{name:"opt_double"   number:13 label:LABEL_OPTIONAL type:TYPE_DOUBLE   default_value:"3.14159265359"},
			{name:"opt_string"   number:14 label:LABEL_OPTIONAL type:TYPE_STRING   default_value:"hello, \"world!\"\n"},
			{name:"opt_bytes"    number:15 label:LABEL_OPTIONAL type:TYPE_BYTES    default_value:"dead\\336\\255\\276\\357beef"},
			{name:"opt_enum"     number:16 label:LABEL_OPTIONAL type:TYPE_ENUM     type_name:".`+enumName+`" default_value:"UNKNOWN_0"},
			{name:"opt_message"  number:17 label:LABEL_OPTIONAL type:TYPE_MESSAGE  type_name:".`+messageName+`"},

			{name:"rep_bool"     number:18 label:LABEL_REPEATED type:TYPE_BOOL     options:{packed:true}},
			{name:"rep_int32"    number:19 label:LABEL_REPEATED type:TYPE_INT32    options:{packed:true}},
			{name:"rep_sint32"   number:20 label:LABEL_REPEATED type:TYPE_SINT32   options:{packed:true}},
			{name:"rep_uint32"   number:21 label:LABEL_REPEATED type:TYPE_UINT32   options:{packed:true}},
			{name:"rep_int64"    number:22 label:LABEL_REPEATED type:TYPE_INT64    options:{packed:true}},
			{name:"rep_sint64"   number:23 label:LABEL_REPEATED type:TYPE_SINT64   options:{packed:true}},
			{name:"rep_uint64"   number:24 label:LABEL_REPEATED type:TYPE_UINT64   options:{packed:true}},
			{name:"rep_fixed32"  number:25 label:LABEL_REPEATED type:TYPE_FIXED32  options:{packed:true}},
			{name:"rep_sfixed32" number:26 label:LABEL_REPEATED type:TYPE_SFIXED32 options:{packed:true}},
			{name:"rep_float"    number:27 label:LABEL_REPEATED type:TYPE_FLOAT    options:{packed:true}},
			{name:"rep_fixed64"  number:28 label:LABEL_REPEATED type:TYPE_FIXED64  options:{packed:true}},
			{name:"rep_sfixed64" number:29 label:LABEL_REPEATED type:TYPE_SFIXED64 options:{packed:true}},
			{name:"rep_double"   number:30 label:LABEL_REPEATED type:TYPE_DOUBLE   options:{packed:true}},
			{name:"rep_string"   number:31 label:LABEL_REPEATED type:TYPE_STRING},
			{name:"rep_bytes"    number:32 label:LABEL_REPEATED type:TYPE_BYTES},
			{name:"rep_enum"     number:33 label:LABEL_REPEATED type:TYPE_ENUM     type_name:".`+enumName+`"},
			{name:"rep_message"  number:34 label:LABEL_REPEATED type:TYPE_MESSAGE  type_name:".`+messageName+`"},

			{name:"map_string_bool"     number:35 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringBoolEntry"},
			{name:"map_string_int32"    number:36 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringInt32Entry"},
			{name:"map_string_sint32"   number:37 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringSint32Entry"},
			{name:"map_string_uint32"   number:38 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringUint32Entry"},
			{name:"map_string_int64"    number:39 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringInt64Entry"},
			{name:"map_string_sint64"   number:40 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringSint64Entry"},
			{name:"map_string_uint64"   number:41 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringUint64Entry"},
			{name:"map_string_fixed32"  number:42 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringFixed32Entry"},
			{name:"map_string_sfixed32" number:43 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringSfixed32Entry"},
			{name:"map_string_float"    number:44 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringFloatEntry"},
			{name:"map_string_fixed64"  number:45 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringFixed64Entry"},
			{name:"map_string_sfixed64" number:46 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringSfixed64Entry"},
			{name:"map_string_double"   number:47 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringDoubleEntry"},
			{name:"map_string_string"   number:48 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringStringEntry"},
			{name:"map_string_bytes"    number:49 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringBytesEntry"},
			{name:"map_string_enum"     number:50 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringEnumEntry"},
			{name:"map_string_message"  number:51 label:LABEL_REPEATED type:TYPE_MESSAGE type_name:".`+messageName+`.MapStringMessageEntry"},

			{name:"oneof_bool"     number:52 label:LABEL_OPTIONAL type:TYPE_BOOL     oneof_index:0 default_value:"true"},
			{name:"oneof_int32"    number:53 label:LABEL_OPTIONAL type:TYPE_INT32    oneof_index:0 default_value:"-12345"},
			{name:"oneof_sint32"   number:54 label:LABEL_OPTIONAL type:TYPE_SINT32   oneof_index:0 default_value:"-3200"},
			{name:"oneof_uint32"   number:55 label:LABEL_OPTIONAL type:TYPE_UINT32   oneof_index:0 default_value:"3200"},
			{name:"oneof_int64"    number:56 label:LABEL_OPTIONAL type:TYPE_INT64    oneof_index:0 default_value:"-123456789"},
			{name:"oneof_sint64"   number:57 label:LABEL_OPTIONAL type:TYPE_SINT64   oneof_index:0 default_value:"-6400"},
			{name:"oneof_uint64"   number:58 label:LABEL_OPTIONAL type:TYPE_UINT64   oneof_index:0 default_value:"6400"},
			{name:"oneof_fixed32"  number:59 label:LABEL_OPTIONAL type:TYPE_FIXED32  oneof_index:0 default_value:"320000"},
			{name:"oneof_sfixed32" number:60 label:LABEL_OPTIONAL type:TYPE_SFIXED32 oneof_index:0 default_value:"-320000"},
			{name:"oneof_float"    number:61 label:LABEL_OPTIONAL type:TYPE_FLOAT    oneof_index:0 default_value:"3.14159"},
			{name:"oneof_fixed64"  number:62 label:LABEL_OPTIONAL type:TYPE_FIXED64  oneof_index:0 default_value:"640000"},
			{name:"oneof_sfixed64" number:63 label:LABEL_OPTIONAL type:TYPE_SFIXED64 oneof_index:0 default_value:"-640000"},
			{name:"oneof_double"   number:64 label:LABEL_OPTIONAL type:TYPE_DOUBLE   oneof_index:0 default_value:"3.14159265359"},
			{name:"oneof_string"   number:65 label:LABEL_OPTIONAL type:TYPE_STRING   oneof_index:0 default_value:"hello, \"world!\"\n"},
			{name:"oneof_bytes"    number:66 label:LABEL_OPTIONAL type:TYPE_BYTES    oneof_index:0 default_value:"dead\\336\\255\\276\\357beef"},
			{name:"oneof_enum"     number:67 label:LABEL_OPTIONAL type:TYPE_ENUM     oneof_index:0 type_name:".`+enumName+`" default_value:"UNKNOWN_0"},
			{name:"oneof_message"  number:68 label:LABEL_OPTIONAL type:TYPE_MESSAGE  oneof_index:0 type_name:".`+messageName+`"}
		]
		oneof_decl: [{name:"oneof_union"}]
		extension_range: [{start:10 end:101}]
		nested_type: [
			{name:"MapStringBoolEntry"     field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BOOL}]     options:{map_entry:true}},
			{name:"MapStringInt32Entry"    field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_INT32}]    options:{map_entry:true}},
			{name:"MapStringSint32Entry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_SINT32}]   options:{map_entry:true}},
			{name:"MapStringUint32Entry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_UINT32}]   options:{map_entry:true}},
			{name:"MapStringInt64Entry"    field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_INT64}]    options:{map_entry:true}},
			{name:"MapStringSint64Entry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_SINT64}]   options:{map_entry:true}},
			{name:"MapStringUint64Entry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_UINT64}]   options:{map_entry:true}},
			{name:"MapStringFixed32Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_FIXED32}]  options:{map_entry:true}},
			{name:"MapStringSfixed32Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_SFIXED32}] options:{map_entry:true}},
			{name:"MapStringFloatEntry"    field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_FLOAT}]    options:{map_entry:true}},
			{name:"MapStringFixed64Entry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_FIXED64}]  options:{map_entry:true}},
			{name:"MapStringSfixed64Entry" field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_SFIXED64}] options:{map_entry:true}},
			{name:"MapStringDoubleEntry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_DOUBLE}]   options:{map_entry:true}},
			{name:"MapStringStringEntry"   field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}]   options:{map_entry:true}},
			{name:"MapStringBytesEntry"    field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_BYTES}]    options:{map_entry:true}},
			{name:"MapStringEnumEntry"     field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_ENUM    type_name:".`+enumName+`"}] options:{map_entry:true}},
			{name:"MapStringMessageEntry"  field:[{name:"key" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}, {name:"value" number:2 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".`+messageName+`"}]                  options:{map_entry:true}}
		]
	`), want); err != nil {
		t.Fatalf("prototext.Unmarshal() error: %v", err)
	}

	md := impl.LegacyLoadMessageDesc(reflect.TypeOf(&AberrantMessage{}))
	got := protodesc.ToDescriptorProto(md)
	if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
		t.Errorf("mismatching descriptor (-want +got):\n%s", diff)
	}
}

type AberrantMessage1 struct {
	M *AberrantMessage2 `protobuf:"bytes,1,opt,name=message"`
}

type AberrantMessage2 struct {
	M *AberrantMessage1 `protobuf:"bytes,1,opt,name=message"`
}

func TestAberrantRace(t *testing.T) {
	var gotMD1, wantMD1, gotMD2, wantMD2 protoreflect.MessageDescriptor

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		md := impl.LegacyLoadMessageDesc(reflect.TypeOf(&AberrantMessage1{}))
		wantMD2 = md.Fields().Get(0).Message()
		gotMD2 = wantMD2.Fields().Get(0).Message().Fields().Get(0).Message()
	}()
	go func() {
		defer wg.Done()
		md := impl.LegacyLoadMessageDesc(reflect.TypeOf(&AberrantMessage2{}))
		wantMD1 = md.Fields().Get(0).Message()
		gotMD1 = wantMD1.Fields().Get(0).Message().Fields().Get(0).Message()
	}()
	wg.Wait()

	if gotMD1 != wantMD1 || gotMD2 != wantMD2 {
		t.Errorf("mismatching exact message descriptors")
	}
}

func TestAberrantExtensions(t *testing.T) {
	tests := []struct {
		in              *impl.ExtensionInfo
		wantName        protoreflect.FullName
		wantNumber      protoreflect.FieldNumber
		wantPlaceholder bool
	}{{
		in:              &impl.ExtensionInfo{Field: 500},
		wantNumber:      500,
		wantPlaceholder: true,
	}, {
		in:              &impl.ExtensionInfo{Name: "foo.bar.baz"},
		wantName:        "foo.bar.baz",
		wantPlaceholder: true,
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			xtd := tt.in.TypeDescriptor()
			switch {
			case xtd.FullName() != tt.wantName:
				t.Errorf("FullName() = %v, want %v", xtd.FullName(), tt.wantName)
			case xtd.Number() != tt.wantNumber:
				t.Errorf("Number() = %v, want %v", xtd.Number(), tt.wantNumber)
			case xtd.IsPlaceholder() != tt.wantPlaceholder:
				t.Errorf("IsPlaceholder() = %v, want %v", xtd.IsPlaceholder(), tt.wantPlaceholder)
			}
		})
	}
}
