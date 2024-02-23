// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"text/template"
)

func generateImplCodec() string {
	return mustExecute(implCodecTemplate, ProtoKinds)
}

var implCodecTemplate = template.Must(template.New("").Parse(`
{{- /*
  IsZero is an expression testing if 'v' is the zero value.
*/ -}}
{{- define "IsZero" -}}
{{if eq .WireType "Bytes" -}}
len(v) == 0
{{- else if or (eq .Name "Double") (eq .Name "Float") -}}
v == 0 && !math.Signbit(float64(v))
{{- else -}}
v == {{.GoType.Zero}}
{{- end -}}
{{- end -}}

{{- /*
  Size is an expression computing the size of 'v'.
*/ -}}
{{- define "Size" -}}
{{- if .WireType.ConstSize -}}
protowire.Size{{.WireType}}()
{{- else if eq .WireType "Bytes" -}}
protowire.SizeBytes(len({{.FromGoType}}))
{{- else -}}
protowire.Size{{.WireType}}({{.FromGoType}})
{{- end -}}
{{- end -}}

{{- define "SizeValue" -}}
{{- if .WireType.ConstSize -}}
protowire.Size{{.WireType}}()
{{- else if eq .WireType "Bytes" -}}
protowire.SizeBytes(len({{.FromValue}}))
{{- else -}}
protowire.Size{{.WireType}}({{.FromValue}})
{{- end -}}
{{- end -}}

{{- /*
  Append is a set of statements appending 'v' to 'b'.
*/ -}}
{{- define "Append" -}}
{{- if eq .Name "String" -}}
b = protowire.AppendString(b, {{.FromGoType}})
{{- else -}}
b = protowire.Append{{.WireType}}(b, {{.FromGoType}})
{{- end -}}
{{- end -}}

{{- define "AppendValue" -}}
{{- if eq .Name "String" -}}
b = protowire.AppendString(b, {{.FromValue}})
{{- else -}}
b = protowire.Append{{.WireType}}(b, {{.FromValue}})
{{- end -}}
{{- end -}}

{{- define "Consume" -}}
{{- if eq .WireType "Varint" -}}
var v uint64
var n int
if len(b) >= 1 && b[0] < 0x80 {
	v = uint64(b[0])
	n = 1
} else if len(b) >= 2 && b[1] < 128 {
	v = uint64(b[0]&0x7f) + uint64(b[1])<<7
	n = 2
} else {
	v, n = protowire.ConsumeVarint(b)
}
{{- else -}}
v, n := protowire.Consume{{.WireType}}(b)
{{- end -}}
{{- end -}}

{{- range .}}

{{- if .FromGoType }}
// size{{.Name}} returns the size of wire encoding a {{.GoType}} pointer as a {{.Name}}.
func size{{.Name}}(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	{{if not .WireType.ConstSize -}}
	v := *p.{{.GoType.PointerMethod}}()
	{{- end}}
	return f.tagsize + {{template "Size" .}}
}

// append{{.Name}} wire encodes a {{.GoType}} pointer as a {{.Name}}.
func append{{.Name}}(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := *p.{{.GoType.PointerMethod}}()
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	return b, nil
}

// consume{{.Name}} wire decodes a {{.GoType}} pointer as a {{.Name}}.
func consume{{.Name}}(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	*p.{{.GoType.PointerMethod}}() = {{.ToGoType}}
	out.n = n
	return out, nil
}

var coder{{.Name}} = pointerCoderFuncs{
	size:      size{{.Name}},
	marshal:   append{{.Name}},
	unmarshal: consume{{.Name}},
	merge:     merge{{.GoType.PointerMethod}},
}

{{if or (eq .Name "Bytes") (eq .Name "String")}}
// append{{.Name}}ValidateUTF8 wire encodes a {{.GoType}} pointer as a {{.Name}}.
func append{{.Name}}ValidateUTF8(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := *p.{{.GoType.PointerMethod}}()
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	if !utf8.Valid{{if eq .Name "String"}}String{{end}}(v) {
		return b, errInvalidUTF8{}
	}
	return b, nil
}

// consume{{.Name}}ValidateUTF8 wire decodes a {{.GoType}} pointer as a {{.Name}}.
func consume{{.Name}}ValidateUTF8(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	if !utf8.Valid(v) {
		return out, errInvalidUTF8{}
	}
	*p.{{.GoType.PointerMethod}}() = {{.ToGoType}}
	out.n = n
	return out, nil
}

var coder{{.Name}}ValidateUTF8 = pointerCoderFuncs{
	size:      size{{.Name}},
	marshal:   append{{.Name}}ValidateUTF8,
	unmarshal: consume{{.Name}}ValidateUTF8,
	merge:     merge{{.GoType.PointerMethod}},
}
{{end}}

// size{{.Name}}NoZero returns the size of wire encoding a {{.GoType}} pointer as a {{.Name}}.
// The zero value is not encoded.
func size{{.Name}}NoZero(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	v := *p.{{.GoType.PointerMethod}}()
	if {{template "IsZero" .}} {
		return 0
	}
	return f.tagsize + {{template "Size" .}}
}

// append{{.Name}}NoZero wire encodes a {{.GoType}} pointer as a {{.Name}}.
// The zero value is not encoded.
func append{{.Name}}NoZero(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := *p.{{.GoType.PointerMethod}}()
	if {{template "IsZero" .}} {
		return b, nil
	}
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	return b, nil
}

{{if .ToGoTypeNoZero}}
// consume{{.Name}}NoZero wire decodes a {{.GoType}} pointer as a {{.Name}}.
// The zero value is not decoded.
func consume{{.Name}}NoZero(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	*p.{{.GoType.PointerMethod}}() = {{.ToGoTypeNoZero}}
	out.n = n
	return out, nil
}
{{end}}

var coder{{.Name}}NoZero = pointerCoderFuncs{
	size:      size{{.Name}}NoZero,
	marshal:   append{{.Name}}NoZero,
	unmarshal: consume{{.Name}}{{if .ToGoTypeNoZero}}NoZero{{end}},
	merge:     merge{{.GoType.PointerMethod}}NoZero,
}

{{if or (eq .Name "Bytes") (eq .Name "String")}}
// append{{.Name}}NoZeroValidateUTF8 wire encodes a {{.GoType}} pointer as a {{.Name}}.
// The zero value is not encoded.
func append{{.Name}}NoZeroValidateUTF8(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := *p.{{.GoType.PointerMethod}}()
	if {{template "IsZero" .}} {
		return b, nil
	}
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	if !utf8.Valid{{if eq .Name "String"}}String{{end}}(v) {
		return b, errInvalidUTF8{}
	}
	return b, nil
}

{{if .ToGoTypeNoZero}}
// consume{{.Name}}NoZeroValidateUTF8 wire decodes a {{.GoType}} pointer as a {{.Name}}.
func consume{{.Name}}NoZeroValidateUTF8(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	if !utf8.Valid(v) {
		return out, errInvalidUTF8{}
	}
	*p.{{.GoType.PointerMethod}}() = {{.ToGoTypeNoZero}}
	out.n = n
	return out, nil
}
{{end}}

var coder{{.Name}}NoZeroValidateUTF8 = pointerCoderFuncs{
	size:      size{{.Name}}NoZero,
	marshal:   append{{.Name}}NoZeroValidateUTF8,
	unmarshal: consume{{.Name}}{{if .ToGoTypeNoZero}}NoZero{{end}}ValidateUTF8,
	merge:     merge{{.GoType.PointerMethod}}NoZero,
}
{{end}}

{{- if not .NoPointer}}
// size{{.Name}}Ptr returns the size of wire encoding a *{{.GoType}} pointer as a {{.Name}}.
// It panics if the pointer is nil.
func size{{.Name}}Ptr(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	{{if not .WireType.ConstSize -}}
	v := **p.{{.GoType.PointerMethod}}Ptr()
	{{end -}}
	return f.tagsize + {{template "Size" .}}
}

// append{{.Name}}Ptr wire encodes a *{{.GoType}} pointer as a {{.Name}}.
// It panics if the pointer is nil.
func append{{.Name}}Ptr(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := **p.{{.GoType.PointerMethod}}Ptr()
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	return b, nil
}

// consume{{.Name}}Ptr wire decodes a *{{.GoType}} pointer as a {{.Name}}.
func consume{{.Name}}Ptr(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	vp := p.{{.GoType.PointerMethod}}Ptr()
	if *vp == nil {
		*vp = new({{.GoType}})
	}
	**vp = {{.ToGoType}}
	out.n = n
	return out, nil
}

var coder{{.Name}}Ptr = pointerCoderFuncs{
	size:      size{{.Name}}Ptr,
	marshal:   append{{.Name}}Ptr,
	unmarshal: consume{{.Name}}Ptr,
	merge:     merge{{.GoType.PointerMethod}}Ptr,
}
{{end}}

{{if (eq .Name "String")}}
// append{{.Name}}PtrValidateUTF8 wire encodes a *{{.GoType}} pointer as a {{.Name}}.
// It panics if the pointer is nil.
func append{{.Name}}PtrValidateUTF8(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := **p.{{.GoType.PointerMethod}}Ptr()
	b = protowire.AppendVarint(b, f.wiretag)
	{{template "Append" .}}
	if !utf8.Valid{{if eq .Name "String"}}String{{end}}(v) {
		return b, errInvalidUTF8{}
	}
	return b, nil
}

// consume{{.Name}}PtrValidateUTF8 wire decodes a *{{.GoType}} pointer as a {{.Name}}.
func consume{{.Name}}PtrValidateUTF8(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	if !utf8.Valid(v) {
		return out, errInvalidUTF8{}
	}
	vp := p.{{.GoType.PointerMethod}}Ptr()
	if *vp == nil {
		*vp = new({{.GoType}})
	}
	**vp = {{.ToGoType}}
	out.n = n
	return out, nil
}

var coder{{.Name}}PtrValidateUTF8 = pointerCoderFuncs{
	size:      size{{.Name}}Ptr,
	marshal:   append{{.Name}}PtrValidateUTF8,
	unmarshal: consume{{.Name}}PtrValidateUTF8,
	merge:     merge{{.GoType.PointerMethod}}Ptr,
}
{{end}}

// size{{.Name}}Slice returns the size of wire encoding a []{{.GoType}} pointer as a repeated {{.Name}}.
func size{{.Name}}Slice(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	s := *p.{{.GoType.PointerMethod}}Slice()
	{{if .WireType.ConstSize -}}
	size = len(s) * (f.tagsize + {{template "Size" .}})
	{{- else -}}
	for _, v := range s {
		size += f.tagsize + {{template "Size" .}}
	}
	{{- end}}
	return size
}

// append{{.Name}}Slice encodes a []{{.GoType}} pointer as a repeated {{.Name}}.
func append{{.Name}}Slice(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := *p.{{.GoType.PointerMethod}}Slice()
	for _, v := range s {
		b = protowire.AppendVarint(b, f.wiretag)
		{{template "Append" .}}
	}
	return b, nil
}

// consume{{.Name}}Slice wire decodes a []{{.GoType}} pointer as a repeated {{.Name}}.
func consume{{.Name}}Slice(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	sp := p.{{.GoType.PointerMethod}}Slice()
	{{- if .WireType.Packable}}
	if wtyp == protowire.BytesType {
		b, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return out, errDecode
		}
		{{if .WireType.ConstSize -}}
		count := len(b) / {{template "Size" .}}
		{{- else -}}
		count := 0
		for _, v := range b {
			if v < 0x80 {
				count++
			}
		}
		{{- end}}
		if count > 0 {
			p.grow{{.GoType.PointerMethod}}Slice(count)
		}
		s := *sp
		for len(b) > 0 {
			{{template "Consume" .}}
			if n < 0 {
				return out, errDecode
			}
			s = append(s, {{.ToGoType}})
			b = b[n:]
		}
		*sp = s
		out.n = n
		return out, nil
	}
	{{- end}}
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	*sp = append(*sp, {{.ToGoType}})
	out.n = n
	return out, nil
}

var coder{{.Name}}Slice = pointerCoderFuncs{
	size:      size{{.Name}}Slice,
	marshal:   append{{.Name}}Slice,
	unmarshal: consume{{.Name}}Slice,
	merge:     merge{{.GoType.PointerMethod}}Slice,
}

{{if or (eq .Name "Bytes") (eq .Name "String")}}
// append{{.Name}}SliceValidateUTF8 encodes a []{{.GoType}} pointer as a repeated {{.Name}}.
func append{{.Name}}SliceValidateUTF8(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := *p.{{.GoType.PointerMethod}}Slice()
	for _, v := range s {
		b = protowire.AppendVarint(b, f.wiretag)
		{{template "Append" .}}
		if !utf8.Valid{{if eq .Name "String"}}String{{end}}(v) {
			return b, errInvalidUTF8{}
		}
	}
	return b, nil
}

// consume{{.Name}}SliceValidateUTF8 wire decodes a []{{.GoType}} pointer as a repeated {{.Name}}.
func consume{{.Name}}SliceValidateUTF8(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return out, errDecode
	}
	if !utf8.Valid(v) {
		return out, errInvalidUTF8{}
	}
	sp := p.{{.GoType.PointerMethod}}Slice()
	*sp = append(*sp, {{.ToGoType}})
	out.n = n
	return out, nil
}

var coder{{.Name}}SliceValidateUTF8 = pointerCoderFuncs{
	size:      size{{.Name}}Slice,
	marshal:   append{{.Name}}SliceValidateUTF8,
	unmarshal: consume{{.Name}}SliceValidateUTF8,
	merge:     merge{{.GoType.PointerMethod}}Slice,
}
{{end}}

{{if or (eq .WireType "Varint") (eq .WireType "Fixed32") (eq .WireType "Fixed64")}}
// size{{.Name}}PackedSlice returns the size of wire encoding a []{{.GoType}} pointer as a packed repeated {{.Name}}.
func size{{.Name}}PackedSlice(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	s := *p.{{.GoType.PointerMethod}}Slice()
	if len(s) == 0 {
		return 0
	}
	{{if .WireType.ConstSize -}}
	n := len(s) * {{template "Size" .}}
	{{- else -}}
	n := 0
	for _, v := range s {
		n += {{template "Size" .}}
	}
	{{- end}}
	return f.tagsize + protowire.SizeBytes(n)
}

// append{{.Name}}PackedSlice encodes a []{{.GoType}} pointer as a packed repeated {{.Name}}.
func append{{.Name}}PackedSlice(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := *p.{{.GoType.PointerMethod}}Slice()
	if len(s) == 0 {
		return b, nil
	}
	b = protowire.AppendVarint(b, f.wiretag)
	{{if .WireType.ConstSize -}}
	n := len(s) * {{template "Size" .}}
	{{- else -}}
	n := 0
	for _, v := range s {
		n += {{template "Size" .}}
	}
	{{- end}}
	b = protowire.AppendVarint(b, uint64(n))
	for _, v := range s {
		{{template "Append" .}}
	}
	return b, nil
}

var coder{{.Name}}PackedSlice = pointerCoderFuncs{
	size:      size{{.Name}}PackedSlice,
	marshal:   append{{.Name}}PackedSlice,
	unmarshal: consume{{.Name}}Slice,
	merge:     merge{{.GoType.PointerMethod}}Slice,
}
{{end}}

{{end -}}

{{- if not .NoValueCodec}}
// size{{.Name}}Value returns the size of wire encoding a {{.GoType}} value as a {{.Name}}.
func size{{.Name}}Value(v protoreflect.Value, tagsize int, opts marshalOptions) int {
	return tagsize + {{template "SizeValue" .}}
}

// append{{.Name}}Value encodes a {{.GoType}} value as a {{.Name}}.
func append{{.Name}}Value(b []byte, v protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, wiretag)
	{{template "AppendValue" .}}
	return b, nil
}

// consume{{.Name}}Value decodes a {{.GoType}} value as a {{.Name}}.
func consume{{.Name}}Value(b []byte, _ protoreflect.Value, _ protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (_ protoreflect.Value, out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return protoreflect.Value{}, out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return protoreflect.Value{}, out, errDecode
	}
	out.n = n
	return {{.ToValue}}, out, nil
}

var coder{{.Name}}Value = valueCoderFuncs{
	size:      size{{.Name}}Value,
	marshal:   append{{.Name}}Value,
	unmarshal: consume{{.Name}}Value,
{{- if (eq .Name "Bytes")}}
	merge:     mergeBytesValue,
{{- else}}
	merge:     mergeScalarValue,
{{- end}}
}

{{if (eq .Name "String")}}
// append{{.Name}}ValueValidateUTF8 encodes a {{.GoType}} value as a {{.Name}}.
func append{{.Name}}ValueValidateUTF8(b []byte, v protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, wiretag)
	{{template "AppendValue" .}}
	if !utf8.ValidString({{.FromValue}}) {
		return b, errInvalidUTF8{}
	}
	return b, nil
}

// consume{{.Name}}ValueValidateUTF8 decodes a {{.GoType}} value as a {{.Name}}.
func consume{{.Name}}ValueValidateUTF8(b []byte, _ protoreflect.Value, _ protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (_ protoreflect.Value, out unmarshalOutput, err error) {
	if wtyp != {{.WireType.Expr}} {
		return protoreflect.Value{}, out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return protoreflect.Value{}, out, errDecode
	}
	if !utf8.Valid(v) {
		return protoreflect.Value{}, out, errInvalidUTF8{}
	}
	out.n = n
	return {{.ToValue}}, out, nil
}

var coder{{.Name}}ValueValidateUTF8 = valueCoderFuncs{
	size:      size{{.Name}}Value,
	marshal:   append{{.Name}}ValueValidateUTF8,
	unmarshal: consume{{.Name}}ValueValidateUTF8,
	merge:     mergeScalarValue,
}
{{end}}

// size{{.Name}}SliceValue returns the size of wire encoding a []{{.GoType}} value as a repeated {{.Name}}.
func size{{.Name}}SliceValue(listv protoreflect.Value, tagsize int, opts marshalOptions) (size int) {
	list := listv.List()
	{{if .WireType.ConstSize -}}
	size = list.Len() * (tagsize + {{template "SizeValue" .}})
	{{- else -}}
	for i, llen := 0, list.Len(); i < llen; i++ {
		v := list.Get(i)
		size += tagsize + {{template "SizeValue" .}}
	}
	{{- end}}
	return size
}

// append{{.Name}}SliceValue encodes a []{{.GoType}} value as a repeated {{.Name}}.
func append{{.Name}}SliceValue(b []byte, listv protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	list := listv.List()
	for i, llen := 0, list.Len(); i < llen; i++ {
		v := list.Get(i)
		b = protowire.AppendVarint(b, wiretag)
		{{template "AppendValue" .}}
	}
	return b, nil
}

// consume{{.Name}}SliceValue wire decodes a []{{.GoType}} value as a repeated {{.Name}}.
func consume{{.Name}}SliceValue(b []byte, listv protoreflect.Value, _ protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (_ protoreflect.Value, out unmarshalOutput, err error) {
	list := listv.List()
	{{- if .WireType.Packable}}
	if wtyp == protowire.BytesType {
		b, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return protoreflect.Value{}, out, errDecode
		}
		for len(b) > 0 {
			{{template "Consume" .}}
			if n < 0 {
				return protoreflect.Value{}, out, errDecode
			}
			list.Append({{.ToValue}})
			b = b[n:]
		}
		out.n = n
		return listv, out, nil
	}
	{{- end}}
	if wtyp != {{.WireType.Expr}} {
		return protoreflect.Value{}, out, errUnknown
	}
	{{template "Consume" .}}
	if n < 0 {
		return protoreflect.Value{}, out, errDecode
	}
	list.Append({{.ToValue}})
	out.n = n
	return listv, out, nil
}

var coder{{.Name}}SliceValue = valueCoderFuncs{
	size:      size{{.Name}}SliceValue,
	marshal:   append{{.Name}}SliceValue,
	unmarshal: consume{{.Name}}SliceValue,
{{- if (eq .Name "Bytes")}}
	merge:     mergeBytesListValue,
{{- else}}
	merge:     mergeListValue,
{{- end}}
}

{{if or (eq .WireType "Varint") (eq .WireType "Fixed32") (eq .WireType "Fixed64")}}
// size{{.Name}}PackedSliceValue returns the size of wire encoding a []{{.GoType}} value as a packed repeated {{.Name}}.
func size{{.Name}}PackedSliceValue(listv protoreflect.Value, tagsize int, opts marshalOptions) (size int) {
	list := listv.List()
	llen := list.Len()
	if llen == 0 {
		return 0
	}
	{{if .WireType.ConstSize -}}
	n := llen * {{template "SizeValue" .}}
	{{- else -}}
	n := 0
	for i, llen := 0, llen; i < llen; i++ {
		v := list.Get(i)
		n += {{template "SizeValue" .}}
	}
	{{- end}}
	return tagsize + protowire.SizeBytes(n)
}

// append{{.Name}}PackedSliceValue encodes a []{{.GoType}} value as a packed repeated {{.Name}}.
func append{{.Name}}PackedSliceValue(b []byte, listv protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	list := listv.List()
	llen := list.Len()
	if llen == 0 {
		return b, nil
	}
	b = protowire.AppendVarint(b, wiretag)
	{{if .WireType.ConstSize -}}
	n := llen * {{template "SizeValue" .}}
	{{- else -}}
	n := 0
	for i := 0; i < llen; i++ {
		v := list.Get(i)
		n += {{template "SizeValue" .}}
	}
	{{- end}}
	b = protowire.AppendVarint(b, uint64(n))
	for i := 0; i < llen; i++ {
		v := list.Get(i)
		{{template "AppendValue" .}}
	}
	return b, nil
}

var coder{{.Name}}PackedSliceValue = valueCoderFuncs{
	size:      size{{.Name}}PackedSliceValue,
	marshal:   append{{.Name}}PackedSliceValue,
	unmarshal: consume{{.Name}}SliceValue,
	merge:     mergeListValue,
}
{{end}}

{{- end}}{{/* if not .NoValueCodec */}}

{{end -}}

// We append to an empty array rather than a nil []byte to get non-nil zero-length byte slices.
var emptyBuf [0]byte

var wireTypes = map[protoreflect.Kind]protowire.Type{
{{range . -}}
	protoreflect.{{.Name}}Kind: {{.WireType.Expr}},
{{end}}
}
`))

func generateImplMessage() string {
	return mustExecute(implMessageTemplate, []string{"messageState", "messageReflectWrapper"})
}

var implMessageTemplate = template.Must(template.New("").Parse(`
{{range . -}}
func (m *{{.}}) Descriptor() protoreflect.MessageDescriptor {
	return m.messageInfo().Desc
}
func (m *{{.}}) Type() protoreflect.MessageType {
	return m.messageInfo()
}
func (m *{{.}}) New() protoreflect.Message {
	return m.messageInfo().New()
}
func (m *{{.}}) Interface() protoreflect.ProtoMessage {
	{{if eq . "messageState" -}}
	return m.protoUnwrap().(protoreflect.ProtoMessage)
	{{- else -}}
	if m, ok := m.protoUnwrap().(protoreflect.ProtoMessage); ok {
		return m
	}
	return (*messageIfaceWrapper)(m)
	{{- end -}}
}
func (m *{{.}}) protoUnwrap() interface{} {
	return m.pointer().AsIfaceOf(m.messageInfo().GoReflectType.Elem())
}
func (m *{{.}}) ProtoMethods() *protoiface.Methods {
	m.messageInfo().init()
	return &m.messageInfo().methods
}

// ProtoMessageInfo is a pseudo-internal API for allowing the v1 code
// to be able to retrieve a v2 MessageInfo struct.
//
// WARNING: This method is exempt from the compatibility promise and
// may be removed in the future without warning.
func (m *{{.}}) ProtoMessageInfo() *MessageInfo {
	return m.messageInfo()
}

func (m *{{.}}) Range(f func(protoreflect.FieldDescriptor, protoreflect.Value) bool) {
	m.messageInfo().init()
	for _, ri := range m.messageInfo().rangeInfos {
		switch ri := ri.(type) {
		case *fieldInfo:
			if ri.has(m.pointer()) {
				if !f(ri.fieldDesc, ri.get(m.pointer())) {
					return
				}
			}
		case *oneofInfo:
			if n := ri.which(m.pointer()); n > 0 {
				fi := m.messageInfo().fields[n]
				if !f(fi.fieldDesc, fi.get(m.pointer())) {
					return
				}
			}
		}
	}
	m.messageInfo().extensionMap(m.pointer()).Range(f)
}
func (m *{{.}}) Has(fd protoreflect.FieldDescriptor) bool {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		return fi.has(m.pointer())
	} else {
		return m.messageInfo().extensionMap(m.pointer()).Has(xt)
	}
}
func (m *{{.}}) Clear(fd protoreflect.FieldDescriptor) {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		fi.clear(m.pointer())
	} else {
		m.messageInfo().extensionMap(m.pointer()).Clear(xt)
	}
}
func (m *{{.}}) Get(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		return fi.get(m.pointer())
	} else {
		return m.messageInfo().extensionMap(m.pointer()).Get(xt)
	}
}
func (m *{{.}}) Set(fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		fi.set(m.pointer(), v)
	} else {
		m.messageInfo().extensionMap(m.pointer()).Set(xt, v)
	}
}
func (m *{{.}}) Mutable(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		return fi.mutable(m.pointer())
	} else {
		return m.messageInfo().extensionMap(m.pointer()).Mutable(xt)
	}
}
func (m *{{.}}) NewField(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.messageInfo().init()
	if fi, xt := m.messageInfo().checkField(fd); fi != nil {
		return fi.newField()
	} else {
		return xt.New()
	}
}
func (m *{{.}}) WhichOneof(od protoreflect.OneofDescriptor) protoreflect.FieldDescriptor {
	m.messageInfo().init()
	if oi := m.messageInfo().oneofs[od.Name()]; oi != nil && oi.oneofDesc == od {
		return od.Fields().ByNumber(oi.which(m.pointer()))
	}
	panic("invalid oneof descriptor " + string(od.FullName()) + " for message " + string(m.Descriptor().FullName()))
}
func (m *{{.}}) GetUnknown() protoreflect.RawFields {
	m.messageInfo().init()
	return m.messageInfo().getUnknown(m.pointer())
}
func (m *{{.}}) SetUnknown(b protoreflect.RawFields) {
	m.messageInfo().init()
	m.messageInfo().setUnknown(m.pointer(), b)
}
func (m *{{.}}) IsValid() bool {
	return !m.pointer().IsNil()
}

{{end}}
`))

func generateImplMerge() string {
	return mustExecute(implMergeTemplate, GoTypes)
}

var implMergeTemplate = template.Must(template.New("").Parse(`
{{range .}}
{{if ne . "[]byte"}}
func merge{{.PointerMethod}}(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	*dst.{{.PointerMethod}}() = *src.{{.PointerMethod}}()
}

func merge{{.PointerMethod}}NoZero(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	v := *src.{{.PointerMethod}}()
	if v != {{.Zero}} {
		*dst.{{.PointerMethod}}() = v
	}
}

func merge{{.PointerMethod}}Ptr(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	p := *src.{{.PointerMethod}}Ptr()
	if p != nil {
		v := *p
		*dst.{{.PointerMethod}}Ptr() = &v
	}
}

func merge{{.PointerMethod}}Slice(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	ds := dst.{{.PointerMethod}}Slice()
	ss := src.{{.PointerMethod}}Slice()
	*ds = append(*ds, *ss...)
}

{{end}}
{{end}}
`))
