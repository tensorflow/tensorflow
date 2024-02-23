// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"text/template"
)

type WireType string

const (
	WireVarint  WireType = "Varint"
	WireFixed32 WireType = "Fixed32"
	WireFixed64 WireType = "Fixed64"
	WireBytes   WireType = "Bytes"
	WireGroup   WireType = "Group"
)

func (w WireType) Expr() Expr {
	if w == WireGroup {
		return "protowire.StartGroupType"
	}
	return "protowire." + Expr(w) + "Type"
}

func (w WireType) Packable() bool {
	return w == WireVarint || w == WireFixed32 || w == WireFixed64
}

func (w WireType) ConstSize() bool {
	return w == WireFixed32 || w == WireFixed64
}

type GoType string

var GoTypes = []GoType{
	GoBool,
	GoInt32,
	GoUint32,
	GoInt64,
	GoUint64,
	GoFloat32,
	GoFloat64,
	GoString,
	GoBytes,
}

const (
	GoBool    = "bool"
	GoInt32   = "int32"
	GoUint32  = "uint32"
	GoInt64   = "int64"
	GoUint64  = "uint64"
	GoFloat32 = "float32"
	GoFloat64 = "float64"
	GoString  = "string"
	GoBytes   = "[]byte"
)

func (g GoType) Zero() Expr {
	switch g {
	case GoBool:
		return "false"
	case GoString:
		return `""`
	case GoBytes:
		return "nil"
	}
	return "0"
}

// Kind is the reflect.Kind of the type.
func (g GoType) Kind() Expr {
	if g == "" || g == GoBytes {
		return ""
	}
	return "reflect." + Expr(strings.ToUpper(string(g[:1]))+string(g[1:]))
}

// PointerMethod is the "internal/impl".pointer method used to access a pointer to this type.
func (g GoType) PointerMethod() Expr {
	if g == GoBytes {
		return "Bytes"
	}
	return Expr(strings.ToUpper(string(g[:1])) + string(g[1:]))
}

type ProtoKind struct {
	Name     string
	WireType WireType

	// Conversions to/from protoreflect.Value.
	ToValue   Expr
	FromValue Expr

	// Conversions to/from generated structures.
	GoType         GoType
	ToGoType       Expr
	ToGoTypeNoZero Expr
	FromGoType     Expr
	NoPointer      bool
	NoValueCodec   bool
}

func (k ProtoKind) Expr() Expr {
	return "protoreflect." + Expr(k.Name) + "Kind"
}

var ProtoKinds = []ProtoKind{
	{
		Name:       "Bool",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfBool(protowire.DecodeBool(v))",
		FromValue:  "protowire.EncodeBool(v.Bool())",
		GoType:     GoBool,
		ToGoType:   "protowire.DecodeBool(v)",
		FromGoType: "protowire.EncodeBool(v)",
	},
	{
		Name:      "Enum",
		WireType:  WireVarint,
		ToValue:   "protoreflect.ValueOfEnum(protoreflect.EnumNumber(v))",
		FromValue: "uint64(v.Enum())",
	},
	{
		Name:       "Int32",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfInt32(int32(v))",
		FromValue:  "uint64(int32(v.Int()))",
		GoType:     GoInt32,
		ToGoType:   "int32(v)",
		FromGoType: "uint64(v)",
	},
	{
		Name:       "Sint32",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfInt32(int32(protowire.DecodeZigZag(v & math.MaxUint32)))",
		FromValue:  "protowire.EncodeZigZag(int64(int32(v.Int())))",
		GoType:     GoInt32,
		ToGoType:   "int32(protowire.DecodeZigZag(v & math.MaxUint32))",
		FromGoType: "protowire.EncodeZigZag(int64(v))",
	},
	{
		Name:       "Uint32",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfUint32(uint32(v))",
		FromValue:  "uint64(uint32(v.Uint()))",
		GoType:     GoUint32,
		ToGoType:   "uint32(v)",
		FromGoType: "uint64(v)",
	},
	{
		Name:       "Int64",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfInt64(int64(v))",
		FromValue:  "uint64(v.Int())",
		GoType:     GoInt64,
		ToGoType:   "int64(v)",
		FromGoType: "uint64(v)",
	},
	{
		Name:       "Sint64",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfInt64(protowire.DecodeZigZag(v))",
		FromValue:  "protowire.EncodeZigZag(v.Int())",
		GoType:     GoInt64,
		ToGoType:   "protowire.DecodeZigZag(v)",
		FromGoType: "protowire.EncodeZigZag(v)",
	},
	{
		Name:       "Uint64",
		WireType:   WireVarint,
		ToValue:    "protoreflect.ValueOfUint64(v)",
		FromValue:  "v.Uint()",
		GoType:     GoUint64,
		ToGoType:   "v",
		FromGoType: "v",
	},
	{
		Name:       "Sfixed32",
		WireType:   WireFixed32,
		ToValue:    "protoreflect.ValueOfInt32(int32(v))",
		FromValue:  "uint32(v.Int())",
		GoType:     GoInt32,
		ToGoType:   "int32(v)",
		FromGoType: "uint32(v)",
	},
	{
		Name:       "Fixed32",
		WireType:   WireFixed32,
		ToValue:    "protoreflect.ValueOfUint32(uint32(v))",
		FromValue:  "uint32(v.Uint())",
		GoType:     GoUint32,
		ToGoType:   "v",
		FromGoType: "v",
	},
	{
		Name:       "Float",
		WireType:   WireFixed32,
		ToValue:    "protoreflect.ValueOfFloat32(math.Float32frombits(uint32(v)))",
		FromValue:  "math.Float32bits(float32(v.Float()))",
		GoType:     GoFloat32,
		ToGoType:   "math.Float32frombits(v)",
		FromGoType: "math.Float32bits(v)",
	},
	{
		Name:       "Sfixed64",
		WireType:   WireFixed64,
		ToValue:    "protoreflect.ValueOfInt64(int64(v))",
		FromValue:  "uint64(v.Int())",
		GoType:     GoInt64,
		ToGoType:   "int64(v)",
		FromGoType: "uint64(v)",
	},
	{
		Name:       "Fixed64",
		WireType:   WireFixed64,
		ToValue:    "protoreflect.ValueOfUint64(v)",
		FromValue:  "v.Uint()",
		GoType:     GoUint64,
		ToGoType:   "v",
		FromGoType: "v",
	},
	{
		Name:       "Double",
		WireType:   WireFixed64,
		ToValue:    "protoreflect.ValueOfFloat64(math.Float64frombits(v))",
		FromValue:  "math.Float64bits(v.Float())",
		GoType:     GoFloat64,
		ToGoType:   "math.Float64frombits(v)",
		FromGoType: "math.Float64bits(v)",
	},
	{
		Name:       "String",
		WireType:   WireBytes,
		ToValue:    "protoreflect.ValueOfString(string(v))",
		FromValue:  "v.String()",
		GoType:     GoString,
		ToGoType:   "string(v)",
		FromGoType: "v",
	},
	{
		Name:           "Bytes",
		WireType:       WireBytes,
		ToValue:        "protoreflect.ValueOfBytes(append(emptyBuf[:], v...))",
		FromValue:      "v.Bytes()",
		GoType:         GoBytes,
		ToGoType:       "append(emptyBuf[:], v...)",
		ToGoTypeNoZero: "append(([]byte)(nil), v...)",
		FromGoType:     "v",
		NoPointer:      true,
	},
	{
		Name:         "Message",
		WireType:     WireBytes,
		ToValue:      "protoreflect.ValueOfBytes(v)",
		FromValue:    "v",
		NoValueCodec: true,
	},
	{
		Name:         "Group",
		WireType:     WireGroup,
		ToValue:      "protoreflect.ValueOfBytes(v)",
		FromValue:    "v",
		NoValueCodec: true,
	},
}

func generateProtoDecode() string {
	return mustExecute(protoDecodeTemplate, ProtoKinds)
}

var protoDecodeTemplate = template.Must(template.New("").Parse(`
// unmarshalScalar decodes a value of the given kind.
//
// Message values are decoded into a []byte which aliases the input data.
func (o UnmarshalOptions) unmarshalScalar(b []byte, wtyp protowire.Type, fd protoreflect.FieldDescriptor) (val protoreflect.Value, n int, err error) {
	switch fd.Kind() {
	{{- range .}}
	case {{.Expr}}:
		if wtyp != {{.WireType.Expr}} {
			return val, 0, errUnknown
		}
		{{if (eq .WireType "Group") -}}
		v, n := protowire.ConsumeGroup(fd.Number(), b)
		{{- else -}}
		v, n := protowire.Consume{{.WireType}}(b)
		{{- end}}
		if n < 0 {
			return val, 0, errDecode
		}
		{{if (eq .Name "String") -}}
		if strs.EnforceUTF8(fd) && !utf8.Valid(v) {
			return protoreflect.Value{}, 0, errors.InvalidUTF8(string(fd.FullName()))
		}
		{{end -}}
		return {{.ToValue}}, n, nil
	{{- end}}
	default:
		return val, 0, errUnknown
	}
}

func (o UnmarshalOptions) unmarshalList(b []byte, wtyp protowire.Type, list protoreflect.List, fd protoreflect.FieldDescriptor) (n int, err error) {
	switch fd.Kind() {
	{{- range .}}
	case {{.Expr}}:
		{{- if .WireType.Packable}}
		if wtyp == protowire.BytesType {
			buf, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return 0, errDecode
			}
			for len(buf) > 0 {
				v, n := protowire.Consume{{.WireType}}(buf)
				if n < 0 {
					return 0, errDecode
				}
				buf = buf[n:]
				list.Append({{.ToValue}})
			}
			return n, nil
		}
		{{- end}}
		if wtyp != {{.WireType.Expr}} {
			return 0, errUnknown
		}
		{{if (eq .WireType "Group") -}}
		v, n := protowire.ConsumeGroup(fd.Number(), b)
		{{- else -}}
		v, n := protowire.Consume{{.WireType}}(b)
		{{- end}}
		if n < 0 {
			return 0, errDecode
		}
		{{if (eq .Name "String") -}}
		if strs.EnforceUTF8(fd) && !utf8.Valid(v) {
			return 0, errors.InvalidUTF8(string(fd.FullName()))
		}
		{{end -}}
		{{if or (eq .Name "Message") (eq .Name "Group") -}}
		m := list.NewElement()
		if err := o.unmarshalMessage(v, m.Message()); err != nil {
			return 0, err
		}
		list.Append(m)
		{{- else -}}
		list.Append({{.ToValue}})
		{{- end}}
		return n, nil
	{{- end}}
	default:
		return 0, errUnknown
	}
}

// We append to an empty array rather than a nil []byte to get non-nil zero-length byte slices.
var emptyBuf [0]byte
`))

func generateProtoEncode() string {
	return mustExecute(protoEncodeTemplate, ProtoKinds)
}

var protoEncodeTemplate = template.Must(template.New("").Parse(`
var wireTypes = map[protoreflect.Kind]protowire.Type{
{{- range .}}
	{{.Expr}}: {{.WireType.Expr}},
{{- end}}
}

func (o MarshalOptions) marshalSingular(b []byte, fd protoreflect.FieldDescriptor, v protoreflect.Value) ([]byte, error) {
	switch fd.Kind() {
	{{- range .}}
	case {{.Expr}}:
		{{- if (eq .Name "String") }}
		if strs.EnforceUTF8(fd) && !utf8.ValidString(v.String()) {
			return b, errors.InvalidUTF8(string(fd.FullName()))
		}
		b = protowire.AppendString(b, {{.FromValue}})
		{{- else if (eq .Name "Message") -}}
		var pos int
		var err error
		b, pos = appendSpeculativeLength(b)
		b, err = o.marshalMessage(b, v.Message())
		if err != nil {
			return b, err
		}
		b = finishSpeculativeLength(b, pos)
		{{- else if (eq .Name "Group") -}}
		var err error
		b, err = o.marshalMessage(b, v.Message())
		if err != nil {
			return b, err
		}
		b = protowire.AppendVarint(b, protowire.EncodeTag(fd.Number(), protowire.EndGroupType))
		{{- else -}}
		b = protowire.Append{{.WireType}}(b, {{.FromValue}})
		{{- end}}
	{{- end}}
	default:
		return b, errors.New("invalid kind %v", fd.Kind())
	}
	return b, nil
}
`))

func generateProtoSize() string {
	return mustExecute(protoSizeTemplate, ProtoKinds)
}

var protoSizeTemplate = template.Must(template.New("").Parse(`
func (o MarshalOptions) sizeSingular(num protowire.Number, kind protoreflect.Kind, v protoreflect.Value) int {
	switch kind {
	{{- range .}}
	case {{.Expr}}:
		{{if (eq .Name "Message") -}}
		return protowire.SizeBytes(o.size(v.Message()))
		{{- else if or (eq .WireType "Fixed32") (eq .WireType "Fixed64") -}}
		return protowire.Size{{.WireType}}()
		{{- else if (eq .WireType "Bytes") -}}
		return protowire.Size{{.WireType}}(len({{.FromValue}}))
		{{- else if (eq .WireType "Group") -}}
		return protowire.Size{{.WireType}}(num, o.size(v.Message()))
		{{- else -}}
		return protowire.Size{{.WireType}}({{.FromValue}})
		{{- end}}
	{{- end}}
	default:
		return 0
	}
}
`))
