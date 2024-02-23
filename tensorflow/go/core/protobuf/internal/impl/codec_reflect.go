// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego || appengine
// +build purego appengine

package impl

import (
	"reflect"

	"google.golang.org/protobuf/encoding/protowire"
)

func sizeEnum(p pointer, f *coderFieldInfo, _ marshalOptions) (size int) {
	v := p.v.Elem().Int()
	return f.tagsize + protowire.SizeVarint(uint64(v))
}

func appendEnum(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	v := p.v.Elem().Int()
	b = protowire.AppendVarint(b, f.wiretag)
	b = protowire.AppendVarint(b, uint64(v))
	return b, nil
}

func consumeEnum(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, _ unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.VarintType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeVarint(b)
	if n < 0 {
		return out, errDecode
	}
	p.v.Elem().SetInt(int64(v))
	out.n = n
	return out, nil
}

func mergeEnum(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	dst.v.Elem().Set(src.v.Elem())
}

var coderEnum = pointerCoderFuncs{
	size:      sizeEnum,
	marshal:   appendEnum,
	unmarshal: consumeEnum,
	merge:     mergeEnum,
}

func sizeEnumNoZero(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	if p.v.Elem().Int() == 0 {
		return 0
	}
	return sizeEnum(p, f, opts)
}

func appendEnumNoZero(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	if p.v.Elem().Int() == 0 {
		return b, nil
	}
	return appendEnum(b, p, f, opts)
}

func mergeEnumNoZero(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	if src.v.Elem().Int() != 0 {
		dst.v.Elem().Set(src.v.Elem())
	}
}

var coderEnumNoZero = pointerCoderFuncs{
	size:      sizeEnumNoZero,
	marshal:   appendEnumNoZero,
	unmarshal: consumeEnum,
	merge:     mergeEnumNoZero,
}

func sizeEnumPtr(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	return sizeEnum(pointer{p.v.Elem()}, f, opts)
}

func appendEnumPtr(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	return appendEnum(b, pointer{p.v.Elem()}, f, opts)
}

func consumeEnumPtr(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.VarintType {
		return out, errUnknown
	}
	if p.v.Elem().IsNil() {
		p.v.Elem().Set(reflect.New(p.v.Elem().Type().Elem()))
	}
	return consumeEnum(b, pointer{p.v.Elem()}, wtyp, f, opts)
}

func mergeEnumPtr(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	if !src.v.Elem().IsNil() {
		v := reflect.New(dst.v.Type().Elem().Elem())
		v.Elem().Set(src.v.Elem().Elem())
		dst.v.Elem().Set(v)
	}
}

var coderEnumPtr = pointerCoderFuncs{
	size:      sizeEnumPtr,
	marshal:   appendEnumPtr,
	unmarshal: consumeEnumPtr,
	merge:     mergeEnumPtr,
}

func sizeEnumSlice(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	s := p.v.Elem()
	for i, llen := 0, s.Len(); i < llen; i++ {
		size += protowire.SizeVarint(uint64(s.Index(i).Int())) + f.tagsize
	}
	return size
}

func appendEnumSlice(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := p.v.Elem()
	for i, llen := 0, s.Len(); i < llen; i++ {
		b = protowire.AppendVarint(b, f.wiretag)
		b = protowire.AppendVarint(b, uint64(s.Index(i).Int()))
	}
	return b, nil
}

func consumeEnumSlice(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	s := p.v.Elem()
	if wtyp == protowire.BytesType {
		b, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return out, errDecode
		}
		for len(b) > 0 {
			v, n := protowire.ConsumeVarint(b)
			if n < 0 {
				return out, errDecode
			}
			rv := reflect.New(s.Type().Elem()).Elem()
			rv.SetInt(int64(v))
			s.Set(reflect.Append(s, rv))
			b = b[n:]
		}
		out.n = n
		return out, nil
	}
	if wtyp != protowire.VarintType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeVarint(b)
	if n < 0 {
		return out, errDecode
	}
	rv := reflect.New(s.Type().Elem()).Elem()
	rv.SetInt(int64(v))
	s.Set(reflect.Append(s, rv))
	out.n = n
	return out, nil
}

func mergeEnumSlice(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	dst.v.Elem().Set(reflect.AppendSlice(dst.v.Elem(), src.v.Elem()))
}

var coderEnumSlice = pointerCoderFuncs{
	size:      sizeEnumSlice,
	marshal:   appendEnumSlice,
	unmarshal: consumeEnumSlice,
	merge:     mergeEnumSlice,
}

func sizeEnumPackedSlice(p pointer, f *coderFieldInfo, opts marshalOptions) (size int) {
	s := p.v.Elem()
	llen := s.Len()
	if llen == 0 {
		return 0
	}
	n := 0
	for i := 0; i < llen; i++ {
		n += protowire.SizeVarint(uint64(s.Index(i).Int()))
	}
	return f.tagsize + protowire.SizeBytes(n)
}

func appendEnumPackedSlice(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := p.v.Elem()
	llen := s.Len()
	if llen == 0 {
		return b, nil
	}
	b = protowire.AppendVarint(b, f.wiretag)
	n := 0
	for i := 0; i < llen; i++ {
		n += protowire.SizeVarint(uint64(s.Index(i).Int()))
	}
	b = protowire.AppendVarint(b, uint64(n))
	for i := 0; i < llen; i++ {
		b = protowire.AppendVarint(b, uint64(s.Index(i).Int()))
	}
	return b, nil
}

var coderEnumPackedSlice = pointerCoderFuncs{
	size:      sizeEnumPackedSlice,
	marshal:   appendEnumPackedSlice,
	unmarshal: consumeEnumSlice,
	merge:     mergeEnumSlice,
}
