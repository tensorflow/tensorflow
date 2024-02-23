// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"reflect"
	"sort"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type mapInfo struct {
	goType     reflect.Type
	keyWiretag uint64
	valWiretag uint64
	keyFuncs   valueCoderFuncs
	valFuncs   valueCoderFuncs
	keyZero    protoreflect.Value
	keyKind    protoreflect.Kind
	conv       *mapConverter
}

func encoderFuncsForMap(fd protoreflect.FieldDescriptor, ft reflect.Type) (valueMessage *MessageInfo, funcs pointerCoderFuncs) {
	// TODO: Consider generating specialized map coders.
	keyField := fd.MapKey()
	valField := fd.MapValue()
	keyWiretag := protowire.EncodeTag(1, wireTypes[keyField.Kind()])
	valWiretag := protowire.EncodeTag(2, wireTypes[valField.Kind()])
	keyFuncs := encoderFuncsForValue(keyField)
	valFuncs := encoderFuncsForValue(valField)
	conv := newMapConverter(ft, fd)

	mapi := &mapInfo{
		goType:     ft,
		keyWiretag: keyWiretag,
		valWiretag: valWiretag,
		keyFuncs:   keyFuncs,
		valFuncs:   valFuncs,
		keyZero:    keyField.Default(),
		keyKind:    keyField.Kind(),
		conv:       conv,
	}
	if valField.Kind() == protoreflect.MessageKind {
		valueMessage = getMessageInfo(ft.Elem())
	}

	funcs = pointerCoderFuncs{
		size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
			return sizeMap(p.AsValueOf(ft).Elem(), mapi, f, opts)
		},
		marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
			return appendMap(b, p.AsValueOf(ft).Elem(), mapi, f, opts)
		},
		unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
			mp := p.AsValueOf(ft)
			if mp.Elem().IsNil() {
				mp.Elem().Set(reflect.MakeMap(mapi.goType))
			}
			if f.mi == nil {
				return consumeMap(b, mp.Elem(), wtyp, mapi, f, opts)
			} else {
				return consumeMapOfMessage(b, mp.Elem(), wtyp, mapi, f, opts)
			}
		},
	}
	switch valField.Kind() {
	case protoreflect.MessageKind:
		funcs.merge = mergeMapOfMessage
	case protoreflect.BytesKind:
		funcs.merge = mergeMapOfBytes
	default:
		funcs.merge = mergeMap
	}
	if valFuncs.isInit != nil {
		funcs.isInit = func(p pointer, f *coderFieldInfo) error {
			return isInitMap(p.AsValueOf(ft).Elem(), mapi, f)
		}
	}
	return valueMessage, funcs
}

const (
	mapKeyTagSize = 1 // field 1, tag size 1.
	mapValTagSize = 1 // field 2, tag size 2.
)

func sizeMap(mapv reflect.Value, mapi *mapInfo, f *coderFieldInfo, opts marshalOptions) int {
	if mapv.Len() == 0 {
		return 0
	}
	n := 0
	iter := mapRange(mapv)
	for iter.Next() {
		key := mapi.conv.keyConv.PBValueOf(iter.Key()).MapKey()
		keySize := mapi.keyFuncs.size(key.Value(), mapKeyTagSize, opts)
		var valSize int
		value := mapi.conv.valConv.PBValueOf(iter.Value())
		if f.mi == nil {
			valSize = mapi.valFuncs.size(value, mapValTagSize, opts)
		} else {
			p := pointerOfValue(iter.Value())
			valSize += mapValTagSize
			valSize += protowire.SizeBytes(f.mi.sizePointer(p, opts))
		}
		n += f.tagsize + protowire.SizeBytes(keySize+valSize)
	}
	return n
}

func consumeMap(b []byte, mapv reflect.Value, wtyp protowire.Type, mapi *mapInfo, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	b, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	var (
		key = mapi.keyZero
		val = mapi.conv.valConv.New()
	)
	for len(b) > 0 {
		num, wtyp, n := protowire.ConsumeTag(b)
		if n < 0 {
			return out, errDecode
		}
		if num > protowire.MaxValidNumber {
			return out, errDecode
		}
		b = b[n:]
		err := errUnknown
		switch num {
		case genid.MapEntry_Key_field_number:
			var v protoreflect.Value
			var o unmarshalOutput
			v, o, err = mapi.keyFuncs.unmarshal(b, key, num, wtyp, opts)
			if err != nil {
				break
			}
			key = v
			n = o.n
		case genid.MapEntry_Value_field_number:
			var v protoreflect.Value
			var o unmarshalOutput
			v, o, err = mapi.valFuncs.unmarshal(b, val, num, wtyp, opts)
			if err != nil {
				break
			}
			val = v
			n = o.n
		}
		if err == errUnknown {
			n = protowire.ConsumeFieldValue(num, wtyp, b)
			if n < 0 {
				return out, errDecode
			}
		} else if err != nil {
			return out, err
		}
		b = b[n:]
	}
	mapv.SetMapIndex(mapi.conv.keyConv.GoValueOf(key), mapi.conv.valConv.GoValueOf(val))
	out.n = n
	return out, nil
}

func consumeMapOfMessage(b []byte, mapv reflect.Value, wtyp protowire.Type, mapi *mapInfo, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	b, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	var (
		key = mapi.keyZero
		val = reflect.New(f.mi.GoReflectType.Elem())
	)
	for len(b) > 0 {
		num, wtyp, n := protowire.ConsumeTag(b)
		if n < 0 {
			return out, errDecode
		}
		if num > protowire.MaxValidNumber {
			return out, errDecode
		}
		b = b[n:]
		err := errUnknown
		switch num {
		case 1:
			var v protoreflect.Value
			var o unmarshalOutput
			v, o, err = mapi.keyFuncs.unmarshal(b, key, num, wtyp, opts)
			if err != nil {
				break
			}
			key = v
			n = o.n
		case 2:
			if wtyp != protowire.BytesType {
				break
			}
			var v []byte
			v, n = protowire.ConsumeBytes(b)
			if n < 0 {
				return out, errDecode
			}
			var o unmarshalOutput
			o, err = f.mi.unmarshalPointer(v, pointerOfValue(val), 0, opts)
			if o.initialized {
				// Consider this map item initialized so long as we see
				// an initialized value.
				out.initialized = true
			}
		}
		if err == errUnknown {
			n = protowire.ConsumeFieldValue(num, wtyp, b)
			if n < 0 {
				return out, errDecode
			}
		} else if err != nil {
			return out, err
		}
		b = b[n:]
	}
	mapv.SetMapIndex(mapi.conv.keyConv.GoValueOf(key), val)
	out.n = n
	return out, nil
}

func appendMapItem(b []byte, keyrv, valrv reflect.Value, mapi *mapInfo, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	if f.mi == nil {
		key := mapi.conv.keyConv.PBValueOf(keyrv).MapKey()
		val := mapi.conv.valConv.PBValueOf(valrv)
		size := 0
		size += mapi.keyFuncs.size(key.Value(), mapKeyTagSize, opts)
		size += mapi.valFuncs.size(val, mapValTagSize, opts)
		b = protowire.AppendVarint(b, uint64(size))
		b, err := mapi.keyFuncs.marshal(b, key.Value(), mapi.keyWiretag, opts)
		if err != nil {
			return nil, err
		}
		return mapi.valFuncs.marshal(b, val, mapi.valWiretag, opts)
	} else {
		key := mapi.conv.keyConv.PBValueOf(keyrv).MapKey()
		val := pointerOfValue(valrv)
		valSize := f.mi.sizePointer(val, opts)
		size := 0
		size += mapi.keyFuncs.size(key.Value(), mapKeyTagSize, opts)
		size += mapValTagSize + protowire.SizeBytes(valSize)
		b = protowire.AppendVarint(b, uint64(size))
		b, err := mapi.keyFuncs.marshal(b, key.Value(), mapi.keyWiretag, opts)
		if err != nil {
			return nil, err
		}
		b = protowire.AppendVarint(b, mapi.valWiretag)
		b = protowire.AppendVarint(b, uint64(valSize))
		return f.mi.marshalAppendPointer(b, val, opts)
	}
}

func appendMap(b []byte, mapv reflect.Value, mapi *mapInfo, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	if mapv.Len() == 0 {
		return b, nil
	}
	if opts.Deterministic() {
		return appendMapDeterministic(b, mapv, mapi, f, opts)
	}
	iter := mapRange(mapv)
	for iter.Next() {
		var err error
		b = protowire.AppendVarint(b, f.wiretag)
		b, err = appendMapItem(b, iter.Key(), iter.Value(), mapi, f, opts)
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func appendMapDeterministic(b []byte, mapv reflect.Value, mapi *mapInfo, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	keys := mapv.MapKeys()
	sort.Slice(keys, func(i, j int) bool {
		switch keys[i].Kind() {
		case reflect.Bool:
			return !keys[i].Bool() && keys[j].Bool()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return keys[i].Int() < keys[j].Int()
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
			return keys[i].Uint() < keys[j].Uint()
		case reflect.Float32, reflect.Float64:
			return keys[i].Float() < keys[j].Float()
		case reflect.String:
			return keys[i].String() < keys[j].String()
		default:
			panic("invalid kind: " + keys[i].Kind().String())
		}
	})
	for _, key := range keys {
		var err error
		b = protowire.AppendVarint(b, f.wiretag)
		b, err = appendMapItem(b, key, mapv.MapIndex(key), mapi, f, opts)
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func isInitMap(mapv reflect.Value, mapi *mapInfo, f *coderFieldInfo) error {
	if mi := f.mi; mi != nil {
		mi.init()
		if !mi.needsInitCheck {
			return nil
		}
		iter := mapRange(mapv)
		for iter.Next() {
			val := pointerOfValue(iter.Value())
			if err := mi.checkInitializedPointer(val); err != nil {
				return err
			}
		}
	} else {
		iter := mapRange(mapv)
		for iter.Next() {
			val := mapi.conv.valConv.PBValueOf(iter.Value())
			if err := mapi.valFuncs.isInit(val); err != nil {
				return err
			}
		}
	}
	return nil
}

func mergeMap(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
	dstm := dst.AsValueOf(f.ft).Elem()
	srcm := src.AsValueOf(f.ft).Elem()
	if srcm.Len() == 0 {
		return
	}
	if dstm.IsNil() {
		dstm.Set(reflect.MakeMap(f.ft))
	}
	iter := mapRange(srcm)
	for iter.Next() {
		dstm.SetMapIndex(iter.Key(), iter.Value())
	}
}

func mergeMapOfBytes(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
	dstm := dst.AsValueOf(f.ft).Elem()
	srcm := src.AsValueOf(f.ft).Elem()
	if srcm.Len() == 0 {
		return
	}
	if dstm.IsNil() {
		dstm.Set(reflect.MakeMap(f.ft))
	}
	iter := mapRange(srcm)
	for iter.Next() {
		dstm.SetMapIndex(iter.Key(), reflect.ValueOf(append(emptyBuf[:], iter.Value().Bytes()...)))
	}
}

func mergeMapOfMessage(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
	dstm := dst.AsValueOf(f.ft).Elem()
	srcm := src.AsValueOf(f.ft).Elem()
	if srcm.Len() == 0 {
		return
	}
	if dstm.IsNil() {
		dstm.Set(reflect.MakeMap(f.ft))
	}
	iter := mapRange(srcm)
	for iter.Next() {
		val := reflect.New(f.ft.Elem().Elem())
		if f.mi != nil {
			f.mi.mergePointer(pointerOfValue(val), pointerOfValue(iter.Value()), opts)
		} else {
			opts.Merge(asMessage(val), asMessage(iter.Value()))
		}
		dstm.SetMapIndex(iter.Key(), val)
	}
}
