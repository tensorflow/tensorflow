// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nullable

import (
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/runtime/protoimpl"
	"google.golang.org/protobuf/types/descriptorpb"
)

type Proto2 struct {
	OptionalBool    bool                                   `protobuf:"varint,100,opt,name=optional_bool"`
	OptionalInt32   int32                                  `protobuf:"varint,101,opt,name=optional_int32"`
	OptionalInt64   int64                                  `protobuf:"varint,102,opt,name=optional_int64"`
	OptionalUint32  uint32                                 `protobuf:"varint,103,opt,name=optional_uint32"`
	OptionalUint64  uint64                                 `protobuf:"varint,104,opt,name=optional_uint64"`
	OptionalFloat   float32                                `protobuf:"fixed32,105,opt,name=optional_float"`
	OptionalDouble  float64                                `protobuf:"fixed64,106,opt,name=optional_double"`
	OptionalString  string                                 `protobuf:"bytes,107,opt,name=optional_string"`
	OptionalBytes   []byte                                 `protobuf:"bytes,108,opt,name=optional_bytes"`
	OptionalEnum    descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,109,req,name=optional_enum"`
	OptionalMessage descriptorpb.FieldOptions              `protobuf:"bytes,110,req,name=optional_message"`

	RepeatedBool    []bool                                   `protobuf:"varint,200,rep,name=repeated_bool"`
	RepeatedInt32   []int32                                  `protobuf:"varint,201,rep,name=repeated_int32"`
	RepeatedInt64   []int64                                  `protobuf:"varint,202,rep,name=repeated_int64"`
	RepeatedUint32  []uint32                                 `protobuf:"varint,203,rep,name=repeated_uint32"`
	RepeatedUint64  []uint64                                 `protobuf:"varint,204,rep,name=repeated_uint64"`
	RepeatedFloat   []float32                                `protobuf:"fixed32,205,rep,name=repeated_float"`
	RepeatedDouble  []float64                                `protobuf:"fixed64,206,rep,name=repeated_double"`
	RepeatedString  []string                                 `protobuf:"bytes,207,rep,name=repeated_string"`
	RepeatedBytes   [][]byte                                 `protobuf:"bytes,208,rep,name=repeated_bytes"`
	RepeatedEnum    []descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,209,rep,name=repeated_enum"`
	RepeatedMessage []descriptorpb.FieldOptions              `protobuf:"bytes,210,rep,name=repeated_message"`

	MapBool    map[string]bool                                   `protobuf:"bytes,300,rep,name=map_bool" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapInt32   map[string]int32                                  `protobuf:"bytes,301,rep,name=map_int32" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapInt64   map[string]int64                                  `protobuf:"bytes,302,rep,name=map_int64" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapUint32  map[string]uint32                                 `protobuf:"bytes,303,rep,name=map_uint32" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapUint64  map[string]uint64                                 `protobuf:"bytes,304,rep,name=map_uint64" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapFloat   map[string]float32                                `protobuf:"bytes,305,rep,name=map_float" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed32,2,opt,name=value"`
	MapDouble  map[string]float64                                `protobuf:"bytes,306,rep,name=map_double" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"fixed64,2,opt,name=value"`
	MapString  map[string]string                                 `protobuf:"bytes,307,rep,name=map_string" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	MapBytes   map[string][]byte                                 `protobuf:"bytes,308,rep,name=map_bytes" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	MapEnum    map[string]descriptorpb.FieldDescriptorProto_Type `protobuf:"bytes,309,rep,name=map_enum" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"varint,2,opt,name=value"`
	MapMessage map[string]descriptorpb.FieldOptions              `protobuf:"bytes,310,rep,name=map_message" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`

	OneofUnion isProto2_OneofUnion `protobuf_oneof:"oneof_union"`
}

func (x *Proto2) ProtoMessage()  {}
func (x *Proto2) Reset()         { *x = Proto2{} }
func (x *Proto2) String() string { return prototext.Format(protoimpl.X.ProtoMessageV2Of(x)) }
func (x *Proto2) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Proto2_OneofBool)(nil),
		(*Proto2_OneofInt32)(nil),
		(*Proto2_OneofInt64)(nil),
		(*Proto2_OneofUint32)(nil),
		(*Proto2_OneofUint64)(nil),
		(*Proto2_OneofFloat)(nil),
		(*Proto2_OneofDouble)(nil),
		(*Proto2_OneofString)(nil),
		(*Proto2_OneofBytes)(nil),
		(*Proto2_OneofEnum)(nil),
		(*Proto2_OneofMessage)(nil),
	}
}

type isProto2_OneofUnion interface{ isProto2_OneofUnion() }

type Proto2_OneofBool struct {
	OneofBool bool `protobuf:"varint,400,opt,name=oneof_bool,oneof"`
}
type Proto2_OneofInt32 struct {
	OneofInt32 int32 `protobuf:"varint,401,opt,name=oneof_int32,oneof"`
}
type Proto2_OneofInt64 struct {
	OneofInt64 int64 `protobuf:"varint,402,opt,name=oneof_int64,oneof"`
}
type Proto2_OneofUint32 struct {
	OneofUint32 uint32 `protobuf:"varint,403,opt,name=oneof_uint32,oneof"`
}
type Proto2_OneofUint64 struct {
	OneofUint64 uint64 `protobuf:"varint,404,opt,name=oneof_uint64,oneof"`
}
type Proto2_OneofFloat struct {
	OneofFloat float32 `protobuf:"fixed32,405,opt,name=oneof_float,oneof"`
}
type Proto2_OneofDouble struct {
	OneofDouble float64 `protobuf:"fixed64,406,opt,name=oneof_double,oneof"`
}
type Proto2_OneofString struct {
	OneofString string `protobuf:"bytes,407,opt,name=oneof_string,oneof"`
}
type Proto2_OneofBytes struct {
	OneofBytes []byte `protobuf:"bytes,408,opt,name=oneof_bytes,oneof"`
}
type Proto2_OneofEnum struct {
	OneofEnum descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,409,opt,name=oneof_enum,oneof"`
}
type Proto2_OneofMessage struct {
	OneofMessage descriptorpb.FieldOptions `protobuf:"bytes,410,opt,name=oneof_message,oneof"`
}

func (*Proto2_OneofBool) isProto2_OneofUnion()    {}
func (*Proto2_OneofInt32) isProto2_OneofUnion()   {}
func (*Proto2_OneofInt64) isProto2_OneofUnion()   {}
func (*Proto2_OneofUint32) isProto2_OneofUnion()  {}
func (*Proto2_OneofUint64) isProto2_OneofUnion()  {}
func (*Proto2_OneofFloat) isProto2_OneofUnion()   {}
func (*Proto2_OneofDouble) isProto2_OneofUnion()  {}
func (*Proto2_OneofString) isProto2_OneofUnion()  {}
func (*Proto2_OneofBytes) isProto2_OneofUnion()   {}
func (*Proto2_OneofEnum) isProto2_OneofUnion()    {}
func (*Proto2_OneofMessage) isProto2_OneofUnion() {}

type Proto3 struct {
	OptionalBool    bool                                   `protobuf:"varint,100,opt,name=optional_bool,proto3"`
	OptionalInt32   int32                                  `protobuf:"varint,101,opt,name=optional_int32,proto3"`
	OptionalInt64   int64                                  `protobuf:"varint,102,opt,name=optional_int64,proto3"`
	OptionalUint32  uint32                                 `protobuf:"varint,103,opt,name=optional_uint32,proto3"`
	OptionalUint64  uint64                                 `protobuf:"varint,104,opt,name=optional_uint64,proto3"`
	OptionalFloat   float32                                `protobuf:"fixed32,105,opt,name=optional_float,proto3"`
	OptionalDouble  float64                                `protobuf:"fixed64,106,opt,name=optional_double,proto3"`
	OptionalString  string                                 `protobuf:"bytes,107,opt,name=optional_string,proto3"`
	OptionalBytes   []byte                                 `protobuf:"bytes,108,opt,name=optional_bytes,proto3"`
	OptionalEnum    descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,109,req,name=optional_enum,proto3"`
	OptionalMessage descriptorpb.FieldOptions              `protobuf:"bytes,110,req,name=optional_message,proto3"`

	RepeatedBool    []bool                                   `protobuf:"varint,200,rep,name=repeated_bool,proto3"`
	RepeatedInt32   []int32                                  `protobuf:"varint,201,rep,name=repeated_int32,proto3"`
	RepeatedInt64   []int64                                  `protobuf:"varint,202,rep,name=repeated_int64,proto3"`
	RepeatedUint32  []uint32                                 `protobuf:"varint,203,rep,name=repeated_uint32,proto3"`
	RepeatedUint64  []uint64                                 `protobuf:"varint,204,rep,name=repeated_uint64,proto3"`
	RepeatedFloat   []float32                                `protobuf:"fixed32,205,rep,name=repeated_float,proto3"`
	RepeatedDouble  []float64                                `protobuf:"fixed64,206,rep,name=repeated_double,proto3"`
	RepeatedString  []string                                 `protobuf:"bytes,207,rep,name=repeated_string,proto3"`
	RepeatedBytes   [][]byte                                 `protobuf:"bytes,208,rep,name=repeated_bytes,proto3"`
	RepeatedEnum    []descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,209,rep,name=repeated_enum,proto3"`
	RepeatedMessage []descriptorpb.FieldOptions              `protobuf:"bytes,210,rep,name=repeated_message,proto3"`

	MapBool    map[string]bool                                   `protobuf:"bytes,300,rep,name=map_bool,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapInt32   map[string]int32                                  `protobuf:"bytes,301,rep,name=map_int32,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapInt64   map[string]int64                                  `protobuf:"bytes,302,rep,name=map_int64,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapUint32  map[string]uint32                                 `protobuf:"bytes,303,rep,name=map_uint32,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapUint64  map[string]uint64                                 `protobuf:"bytes,304,rep,name=map_uint64,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapFloat   map[string]float32                                `protobuf:"bytes,305,rep,name=map_float,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"fixed32,2,opt,name=value,proto3"`
	MapDouble  map[string]float64                                `protobuf:"bytes,306,rep,name=map_double,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"fixed64,2,opt,name=value,proto3"`
	MapString  map[string]string                                 `protobuf:"bytes,307,rep,name=map_string,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	MapBytes   map[string][]byte                                 `protobuf:"bytes,308,rep,name=map_bytes,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
	MapEnum    map[string]descriptorpb.FieldDescriptorProto_Type `protobuf:"bytes,309,rep,name=map_enum,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"varint,2,opt,name=value,proto3"`
	MapMessage map[string]descriptorpb.FieldOptions              `protobuf:"bytes,310,rep,name=map_message,proto3" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`

	OneofUnion isProto3_OneofUnion `protobuf_oneof:"oneof_union"`
}

func (x *Proto3) ProtoMessage()  {}
func (x *Proto3) Reset()         { *x = Proto3{} }
func (x *Proto3) String() string { return prototext.Format(protoimpl.X.ProtoMessageV2Of(x)) }
func (x *Proto3) XXX_OneofWrappers() []interface{} {
	return []interface{}{
		(*Proto3_OneofBool)(nil),
		(*Proto3_OneofInt32)(nil),
		(*Proto3_OneofInt64)(nil),
		(*Proto3_OneofUint32)(nil),
		(*Proto3_OneofUint64)(nil),
		(*Proto3_OneofFloat)(nil),
		(*Proto3_OneofDouble)(nil),
		(*Proto3_OneofString)(nil),
		(*Proto3_OneofBytes)(nil),
		(*Proto3_OneofEnum)(nil),
		(*Proto3_OneofMessage)(nil),
	}
}

type isProto3_OneofUnion interface{ isProto3_OneofUnion() }

type Proto3_OneofBool struct {
	OneofBool bool `protobuf:"varint,400,opt,name=oneof_bool,proto3,oneof"`
}
type Proto3_OneofInt32 struct {
	OneofInt32 int32 `protobuf:"varint,401,opt,name=oneof_int32,proto3,oneof"`
}
type Proto3_OneofInt64 struct {
	OneofInt64 int64 `protobuf:"varint,402,opt,name=oneof_int64,proto3,oneof"`
}
type Proto3_OneofUint32 struct {
	OneofUint32 uint32 `protobuf:"varint,403,opt,name=oneof_uint32,proto3,oneof"`
}
type Proto3_OneofUint64 struct {
	OneofUint64 uint64 `protobuf:"varint,404,opt,name=oneof_uint64,proto3,oneof"`
}
type Proto3_OneofFloat struct {
	OneofFloat float32 `protobuf:"fixed32,405,opt,name=oneof_float,proto3,oneof"`
}
type Proto3_OneofDouble struct {
	OneofDouble float64 `protobuf:"fixed64,406,opt,name=oneof_double,proto3,oneof"`
}
type Proto3_OneofString struct {
	OneofString string `protobuf:"bytes,407,opt,name=oneof_string,proto3,oneof"`
}
type Proto3_OneofBytes struct {
	OneofBytes []byte `protobuf:"bytes,408,opt,name=oneof_bytes,proto3,oneof"`
}
type Proto3_OneofEnum struct {
	OneofEnum descriptorpb.FieldDescriptorProto_Type `protobuf:"varint,409,opt,name=oneof_enum,proto3,oneof"`
}
type Proto3_OneofMessage struct {
	OneofMessage descriptorpb.FieldOptions `protobuf:"bytes,410,opt,name=oneof_message,proto3,oneof"`
}

func (*Proto3_OneofBool) isProto3_OneofUnion()    {}
func (*Proto3_OneofInt32) isProto3_OneofUnion()   {}
func (*Proto3_OneofInt64) isProto3_OneofUnion()   {}
func (*Proto3_OneofUint32) isProto3_OneofUnion()  {}
func (*Proto3_OneofUint64) isProto3_OneofUnion()  {}
func (*Proto3_OneofFloat) isProto3_OneofUnion()   {}
func (*Proto3_OneofDouble) isProto3_OneofUnion()  {}
func (*Proto3_OneofString) isProto3_OneofUnion()  {}
func (*Proto3_OneofBytes) isProto3_OneofUnion()   {}
func (*Proto3_OneofEnum) isProto3_OneofUnion()    {}
func (*Proto3_OneofMessage) isProto3_OneofUnion() {}

type Methods struct {
	OptionalInt32 int32 `protobuf:"varint,101,opt,name=optional_int32"`
}

func (x *Methods) ProtoMessage()  {}
func (x *Methods) Reset()         { *x = Methods{} }
func (x *Methods) String() string { return prototext.Format(protoimpl.X.ProtoMessageV2Of(x)) }

func (x *Methods) Marshal() ([]byte, error) {
	var b []byte
	b = protowire.AppendTag(b, 101, protowire.VarintType)
	b = protowire.AppendVarint(b, uint64(x.OptionalInt32))
	return b, nil
}

func (x *Methods) Unmarshal(b []byte) error {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return protowire.ParseError(n)
		}
		b = b[n:]
		if num != 101 || typ != protowire.VarintType {
			n = protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return protowire.ParseError(n)
			}
			b = b[n:]
			continue
		}
		v, n := protowire.ConsumeVarint(b)
		if n < 0 {
			return protowire.ParseError(n)
		}
		b = b[n:]
		x.OptionalInt32 = int32(v)
	}
	return nil
}
