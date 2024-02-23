// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

type mergeOptions struct{}

func (o mergeOptions) Merge(dst, src proto.Message) {
	proto.Merge(dst, src)
}

// merge is protoreflect.Methods.Merge.
func (mi *MessageInfo) merge(in protoiface.MergeInput) protoiface.MergeOutput {
	dp, ok := mi.getPointer(in.Destination)
	if !ok {
		return protoiface.MergeOutput{}
	}
	sp, ok := mi.getPointer(in.Source)
	if !ok {
		return protoiface.MergeOutput{}
	}
	mi.mergePointer(dp, sp, mergeOptions{})
	return protoiface.MergeOutput{Flags: protoiface.MergeComplete}
}

func (mi *MessageInfo) mergePointer(dst, src pointer, opts mergeOptions) {
	mi.init()
	if dst.IsNil() {
		panic(fmt.Sprintf("invalid value: merging into nil message"))
	}
	if src.IsNil() {
		return
	}
	for _, f := range mi.orderedCoderFields {
		if f.funcs.merge == nil {
			continue
		}
		sfptr := src.Apply(f.offset)
		if f.isPointer && sfptr.Elem().IsNil() {
			continue
		}
		f.funcs.merge(dst.Apply(f.offset), sfptr, f, opts)
	}
	if mi.extensionOffset.IsValid() {
		sext := src.Apply(mi.extensionOffset).Extensions()
		dext := dst.Apply(mi.extensionOffset).Extensions()
		if *dext == nil {
			*dext = make(map[int32]ExtensionField)
		}
		for num, sx := range *sext {
			xt := sx.Type()
			xi := getExtensionFieldInfo(xt)
			if xi.funcs.merge == nil {
				continue
			}
			dx := (*dext)[num]
			var dv protoreflect.Value
			if dx.Type() == sx.Type() {
				dv = dx.Value()
			}
			if !dv.IsValid() && xi.unmarshalNeedsValue {
				dv = xt.New()
			}
			dv = xi.funcs.merge(dv, sx.Value(), opts)
			dx.Set(sx.Type(), dv)
			(*dext)[num] = dx
		}
	}
	if mi.unknownOffset.IsValid() {
		su := mi.getUnknownBytes(src)
		if su != nil && len(*su) > 0 {
			du := mi.mutableUnknownBytes(dst)
			*du = append(*du, *su...)
		}
	}
}

func mergeScalarValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	return src
}

func mergeBytesValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	return protoreflect.ValueOfBytes(append(emptyBuf[:], src.Bytes()...))
}

func mergeListValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	dstl := dst.List()
	srcl := src.List()
	for i, llen := 0, srcl.Len(); i < llen; i++ {
		dstl.Append(srcl.Get(i))
	}
	return dst
}

func mergeBytesListValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	dstl := dst.List()
	srcl := src.List()
	for i, llen := 0, srcl.Len(); i < llen; i++ {
		sb := srcl.Get(i).Bytes()
		db := append(emptyBuf[:], sb...)
		dstl.Append(protoreflect.ValueOfBytes(db))
	}
	return dst
}

func mergeMessageListValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	dstl := dst.List()
	srcl := src.List()
	for i, llen := 0, srcl.Len(); i < llen; i++ {
		sm := srcl.Get(i).Message()
		dm := proto.Clone(sm.Interface()).ProtoReflect()
		dstl.Append(protoreflect.ValueOfMessage(dm))
	}
	return dst
}

func mergeMessageValue(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value {
	opts.Merge(dst.Message().Interface(), src.Message().Interface())
	return dst
}

func mergeMessage(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
	if f.mi != nil {
		if dst.Elem().IsNil() {
			dst.SetPointer(pointerOfValue(reflect.New(f.mi.GoReflectType.Elem())))
		}
		f.mi.mergePointer(dst.Elem(), src.Elem(), opts)
	} else {
		dm := dst.AsValueOf(f.ft).Elem()
		sm := src.AsValueOf(f.ft).Elem()
		if dm.IsNil() {
			dm.Set(reflect.New(f.ft.Elem()))
		}
		opts.Merge(asMessage(dm), asMessage(sm))
	}
}

func mergeMessageSlice(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
	for _, sp := range src.PointerSlice() {
		dm := reflect.New(f.ft.Elem().Elem())
		if f.mi != nil {
			f.mi.mergePointer(pointerOfValue(dm), sp, opts)
		} else {
			opts.Merge(asMessage(dm), asMessage(sp.AsValueOf(f.ft.Elem().Elem())))
		}
		dst.AppendPointerSlice(pointerOfValue(dm))
	}
}

func mergeBytes(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	*dst.Bytes() = append(emptyBuf[:], *src.Bytes()...)
}

func mergeBytesNoZero(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	v := *src.Bytes()
	if len(v) > 0 {
		*dst.Bytes() = append(emptyBuf[:], v...)
	}
}

func mergeBytesSlice(dst, src pointer, _ *coderFieldInfo, _ mergeOptions) {
	ds := dst.BytesSlice()
	for _, v := range *src.BytesSlice() {
		*ds = append(*ds, append(emptyBuf[:], v...))
	}
}
