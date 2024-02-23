// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
)

type errInvalidUTF8 struct{}

func (errInvalidUTF8) Error() string     { return "string field contains invalid UTF-8" }
func (errInvalidUTF8) InvalidUTF8() bool { return true }
func (errInvalidUTF8) Unwrap() error     { return errors.Error }

// initOneofFieldCoders initializes the fast-path functions for the fields in a oneof.
//
// For size, marshal, and isInit operations, functions are set only on the first field
// in the oneof. The functions are called when the oneof is non-nil, and will dispatch
// to the appropriate field-specific function as necessary.
//
// The unmarshal function is set on each field individually as usual.
func (mi *MessageInfo) initOneofFieldCoders(od protoreflect.OneofDescriptor, si structInfo) {
	fs := si.oneofsByName[od.Name()]
	ft := fs.Type
	oneofFields := make(map[reflect.Type]*coderFieldInfo)
	needIsInit := false
	fields := od.Fields()
	for i, lim := 0, fields.Len(); i < lim; i++ {
		fd := od.Fields().Get(i)
		num := fd.Number()
		// Make a copy of the original coderFieldInfo for use in unmarshaling.
		//
		// oneofFields[oneofType].funcs.marshal is the field-specific marshal function.
		//
		// mi.coderFields[num].marshal is set on only the first field in the oneof,
		// and dispatches to the field-specific marshaler in oneofFields.
		cf := *mi.coderFields[num]
		ot := si.oneofWrappersByNumber[num]
		cf.ft = ot.Field(0).Type
		cf.mi, cf.funcs = fieldCoder(fd, cf.ft)
		oneofFields[ot] = &cf
		if cf.funcs.isInit != nil {
			needIsInit = true
		}
		mi.coderFields[num].funcs.unmarshal = func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
			var vw reflect.Value         // pointer to wrapper type
			vi := p.AsValueOf(ft).Elem() // oneof field value of interface kind
			if !vi.IsNil() && !vi.Elem().IsNil() && vi.Elem().Elem().Type() == ot {
				vw = vi.Elem()
			} else {
				vw = reflect.New(ot)
			}
			out, err := cf.funcs.unmarshal(b, pointerOfValue(vw).Apply(zeroOffset), wtyp, &cf, opts)
			if err != nil {
				return out, err
			}
			vi.Set(vw)
			return out, nil
		}
	}
	getInfo := func(p pointer) (pointer, *coderFieldInfo) {
		v := p.AsValueOf(ft).Elem()
		if v.IsNil() {
			return pointer{}, nil
		}
		v = v.Elem() // interface -> *struct
		if v.IsNil() {
			return pointer{}, nil
		}
		return pointerOfValue(v).Apply(zeroOffset), oneofFields[v.Elem().Type()]
	}
	first := mi.coderFields[od.Fields().Get(0).Number()]
	first.funcs.size = func(p pointer, _ *coderFieldInfo, opts marshalOptions) int {
		p, info := getInfo(p)
		if info == nil || info.funcs.size == nil {
			return 0
		}
		return info.funcs.size(p, info, opts)
	}
	first.funcs.marshal = func(b []byte, p pointer, _ *coderFieldInfo, opts marshalOptions) ([]byte, error) {
		p, info := getInfo(p)
		if info == nil || info.funcs.marshal == nil {
			return b, nil
		}
		return info.funcs.marshal(b, p, info, opts)
	}
	first.funcs.merge = func(dst, src pointer, _ *coderFieldInfo, opts mergeOptions) {
		srcp, srcinfo := getInfo(src)
		if srcinfo == nil || srcinfo.funcs.merge == nil {
			return
		}
		dstp, dstinfo := getInfo(dst)
		if dstinfo != srcinfo {
			dst.AsValueOf(ft).Elem().Set(reflect.New(src.AsValueOf(ft).Elem().Elem().Elem().Type()))
			dstp = pointerOfValue(dst.AsValueOf(ft).Elem().Elem()).Apply(zeroOffset)
		}
		srcinfo.funcs.merge(dstp, srcp, srcinfo, opts)
	}
	if needIsInit {
		first.funcs.isInit = func(p pointer, _ *coderFieldInfo) error {
			p, info := getInfo(p)
			if info == nil || info.funcs.isInit == nil {
				return nil
			}
			return info.funcs.isInit(p, info)
		}
	}
}

func makeWeakMessageFieldCoder(fd protoreflect.FieldDescriptor) pointerCoderFuncs {
	var once sync.Once
	var messageType protoreflect.MessageType
	lazyInit := func() {
		once.Do(func() {
			messageName := fd.Message().FullName()
			messageType, _ = protoregistry.GlobalTypes.FindMessageByName(messageName)
		})
	}

	return pointerCoderFuncs{
		size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
			m, ok := p.WeakFields().get(f.num)
			if !ok {
				return 0
			}
			lazyInit()
			if messageType == nil {
				panic(fmt.Sprintf("weak message %v is not linked in", fd.Message().FullName()))
			}
			return sizeMessage(m, f.tagsize, opts)
		},
		marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
			m, ok := p.WeakFields().get(f.num)
			if !ok {
				return b, nil
			}
			lazyInit()
			if messageType == nil {
				panic(fmt.Sprintf("weak message %v is not linked in", fd.Message().FullName()))
			}
			return appendMessage(b, m, f.wiretag, opts)
		},
		unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
			fs := p.WeakFields()
			m, ok := fs.get(f.num)
			if !ok {
				lazyInit()
				if messageType == nil {
					return unmarshalOutput{}, errUnknown
				}
				m = messageType.New().Interface()
				fs.set(f.num, m)
			}
			return consumeMessage(b, m, wtyp, opts)
		},
		isInit: func(p pointer, f *coderFieldInfo) error {
			m, ok := p.WeakFields().get(f.num)
			if !ok {
				return nil
			}
			return proto.CheckInitialized(m)
		},
		merge: func(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
			sm, ok := src.WeakFields().get(f.num)
			if !ok {
				return
			}
			dm, ok := dst.WeakFields().get(f.num)
			if !ok {
				lazyInit()
				if messageType == nil {
					panic(fmt.Sprintf("weak message %v is not linked in", fd.Message().FullName()))
				}
				dm = messageType.New().Interface()
				dst.WeakFields().set(f.num, dm)
			}
			opts.Merge(dm, sm)
		},
	}
}

func makeMessageFieldCoder(fd protoreflect.FieldDescriptor, ft reflect.Type) pointerCoderFuncs {
	if mi := getMessageInfo(ft); mi != nil {
		funcs := pointerCoderFuncs{
			size:      sizeMessageInfo,
			marshal:   appendMessageInfo,
			unmarshal: consumeMessageInfo,
			merge:     mergeMessage,
		}
		if needsInitCheck(mi.Desc) {
			funcs.isInit = isInitMessageInfo
		}
		return funcs
	} else {
		return pointerCoderFuncs{
			size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
				m := asMessage(p.AsValueOf(ft).Elem())
				return sizeMessage(m, f.tagsize, opts)
			},
			marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
				m := asMessage(p.AsValueOf(ft).Elem())
				return appendMessage(b, m, f.wiretag, opts)
			},
			unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
				mp := p.AsValueOf(ft).Elem()
				if mp.IsNil() {
					mp.Set(reflect.New(ft.Elem()))
				}
				return consumeMessage(b, asMessage(mp), wtyp, opts)
			},
			isInit: func(p pointer, f *coderFieldInfo) error {
				m := asMessage(p.AsValueOf(ft).Elem())
				return proto.CheckInitialized(m)
			},
			merge: mergeMessage,
		}
	}
}

func sizeMessageInfo(p pointer, f *coderFieldInfo, opts marshalOptions) int {
	return protowire.SizeBytes(f.mi.sizePointer(p.Elem(), opts)) + f.tagsize
}

func appendMessageInfo(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, f.wiretag)
	b = protowire.AppendVarint(b, uint64(f.mi.sizePointer(p.Elem(), opts)))
	return f.mi.marshalAppendPointer(b, p.Elem(), opts)
}

func consumeMessageInfo(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	if p.Elem().IsNil() {
		p.SetPointer(pointerOfValue(reflect.New(f.mi.GoReflectType.Elem())))
	}
	o, err := f.mi.unmarshalPointer(v, p.Elem(), 0, opts)
	if err != nil {
		return out, err
	}
	out.n = n
	out.initialized = o.initialized
	return out, nil
}

func isInitMessageInfo(p pointer, f *coderFieldInfo) error {
	return f.mi.checkInitializedPointer(p.Elem())
}

func sizeMessage(m proto.Message, tagsize int, _ marshalOptions) int {
	return protowire.SizeBytes(proto.Size(m)) + tagsize
}

func appendMessage(b []byte, m proto.Message, wiretag uint64, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, wiretag)
	b = protowire.AppendVarint(b, uint64(proto.Size(m)))
	return opts.Options().MarshalAppend(b, m)
}

func consumeMessage(b []byte, m proto.Message, wtyp protowire.Type, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     v,
		Message: m.ProtoReflect(),
	})
	if err != nil {
		return out, err
	}
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return out, nil
}

func sizeMessageValue(v protoreflect.Value, tagsize int, opts marshalOptions) int {
	m := v.Message().Interface()
	return sizeMessage(m, tagsize, opts)
}

func appendMessageValue(b []byte, v protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	m := v.Message().Interface()
	return appendMessage(b, m, wiretag, opts)
}

func consumeMessageValue(b []byte, v protoreflect.Value, _ protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (protoreflect.Value, unmarshalOutput, error) {
	m := v.Message().Interface()
	out, err := consumeMessage(b, m, wtyp, opts)
	return v, out, err
}

func isInitMessageValue(v protoreflect.Value) error {
	m := v.Message().Interface()
	return proto.CheckInitialized(m)
}

var coderMessageValue = valueCoderFuncs{
	size:      sizeMessageValue,
	marshal:   appendMessageValue,
	unmarshal: consumeMessageValue,
	isInit:    isInitMessageValue,
	merge:     mergeMessageValue,
}

func sizeGroupValue(v protoreflect.Value, tagsize int, opts marshalOptions) int {
	m := v.Message().Interface()
	return sizeGroup(m, tagsize, opts)
}

func appendGroupValue(b []byte, v protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	m := v.Message().Interface()
	return appendGroup(b, m, wiretag, opts)
}

func consumeGroupValue(b []byte, v protoreflect.Value, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (protoreflect.Value, unmarshalOutput, error) {
	m := v.Message().Interface()
	out, err := consumeGroup(b, m, num, wtyp, opts)
	return v, out, err
}

var coderGroupValue = valueCoderFuncs{
	size:      sizeGroupValue,
	marshal:   appendGroupValue,
	unmarshal: consumeGroupValue,
	isInit:    isInitMessageValue,
	merge:     mergeMessageValue,
}

func makeGroupFieldCoder(fd protoreflect.FieldDescriptor, ft reflect.Type) pointerCoderFuncs {
	num := fd.Number()
	if mi := getMessageInfo(ft); mi != nil {
		funcs := pointerCoderFuncs{
			size:      sizeGroupType,
			marshal:   appendGroupType,
			unmarshal: consumeGroupType,
			merge:     mergeMessage,
		}
		if needsInitCheck(mi.Desc) {
			funcs.isInit = isInitMessageInfo
		}
		return funcs
	} else {
		return pointerCoderFuncs{
			size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
				m := asMessage(p.AsValueOf(ft).Elem())
				return sizeGroup(m, f.tagsize, opts)
			},
			marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
				m := asMessage(p.AsValueOf(ft).Elem())
				return appendGroup(b, m, f.wiretag, opts)
			},
			unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
				mp := p.AsValueOf(ft).Elem()
				if mp.IsNil() {
					mp.Set(reflect.New(ft.Elem()))
				}
				return consumeGroup(b, asMessage(mp), num, wtyp, opts)
			},
			isInit: func(p pointer, f *coderFieldInfo) error {
				m := asMessage(p.AsValueOf(ft).Elem())
				return proto.CheckInitialized(m)
			},
			merge: mergeMessage,
		}
	}
}

func sizeGroupType(p pointer, f *coderFieldInfo, opts marshalOptions) int {
	return 2*f.tagsize + f.mi.sizePointer(p.Elem(), opts)
}

func appendGroupType(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, f.wiretag) // start group
	b, err := f.mi.marshalAppendPointer(b, p.Elem(), opts)
	b = protowire.AppendVarint(b, f.wiretag+1) // end group
	return b, err
}

func consumeGroupType(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.StartGroupType {
		return out, errUnknown
	}
	if p.Elem().IsNil() {
		p.SetPointer(pointerOfValue(reflect.New(f.mi.GoReflectType.Elem())))
	}
	return f.mi.unmarshalPointer(b, p.Elem(), f.num, opts)
}

func sizeGroup(m proto.Message, tagsize int, _ marshalOptions) int {
	return 2*tagsize + proto.Size(m)
}

func appendGroup(b []byte, m proto.Message, wiretag uint64, opts marshalOptions) ([]byte, error) {
	b = protowire.AppendVarint(b, wiretag) // start group
	b, err := opts.Options().MarshalAppend(b, m)
	b = protowire.AppendVarint(b, wiretag+1) // end group
	return b, err
}

func consumeGroup(b []byte, m proto.Message, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.StartGroupType {
		return out, errUnknown
	}
	b, n := protowire.ConsumeGroup(num, b)
	if n < 0 {
		return out, errDecode
	}
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     b,
		Message: m.ProtoReflect(),
	})
	if err != nil {
		return out, err
	}
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return out, nil
}

func makeMessageSliceFieldCoder(fd protoreflect.FieldDescriptor, ft reflect.Type) pointerCoderFuncs {
	if mi := getMessageInfo(ft); mi != nil {
		funcs := pointerCoderFuncs{
			size:      sizeMessageSliceInfo,
			marshal:   appendMessageSliceInfo,
			unmarshal: consumeMessageSliceInfo,
			merge:     mergeMessageSlice,
		}
		if needsInitCheck(mi.Desc) {
			funcs.isInit = isInitMessageSliceInfo
		}
		return funcs
	}
	return pointerCoderFuncs{
		size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
			return sizeMessageSlice(p, ft, f.tagsize, opts)
		},
		marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
			return appendMessageSlice(b, p, f.wiretag, ft, opts)
		},
		unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
			return consumeMessageSlice(b, p, ft, wtyp, opts)
		},
		isInit: func(p pointer, f *coderFieldInfo) error {
			return isInitMessageSlice(p, ft)
		},
		merge: mergeMessageSlice,
	}
}

func sizeMessageSliceInfo(p pointer, f *coderFieldInfo, opts marshalOptions) int {
	s := p.PointerSlice()
	n := 0
	for _, v := range s {
		n += protowire.SizeBytes(f.mi.sizePointer(v, opts)) + f.tagsize
	}
	return n
}

func appendMessageSliceInfo(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := p.PointerSlice()
	var err error
	for _, v := range s {
		b = protowire.AppendVarint(b, f.wiretag)
		siz := f.mi.sizePointer(v, opts)
		b = protowire.AppendVarint(b, uint64(siz))
		b, err = f.mi.marshalAppendPointer(b, v, opts)
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func consumeMessageSliceInfo(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	m := reflect.New(f.mi.GoReflectType.Elem()).Interface()
	mp := pointerOfIface(m)
	o, err := f.mi.unmarshalPointer(v, mp, 0, opts)
	if err != nil {
		return out, err
	}
	p.AppendPointerSlice(mp)
	out.n = n
	out.initialized = o.initialized
	return out, nil
}

func isInitMessageSliceInfo(p pointer, f *coderFieldInfo) error {
	s := p.PointerSlice()
	for _, v := range s {
		if err := f.mi.checkInitializedPointer(v); err != nil {
			return err
		}
	}
	return nil
}

func sizeMessageSlice(p pointer, goType reflect.Type, tagsize int, _ marshalOptions) int {
	s := p.PointerSlice()
	n := 0
	for _, v := range s {
		m := asMessage(v.AsValueOf(goType.Elem()))
		n += protowire.SizeBytes(proto.Size(m)) + tagsize
	}
	return n
}

func appendMessageSlice(b []byte, p pointer, wiretag uint64, goType reflect.Type, opts marshalOptions) ([]byte, error) {
	s := p.PointerSlice()
	var err error
	for _, v := range s {
		m := asMessage(v.AsValueOf(goType.Elem()))
		b = protowire.AppendVarint(b, wiretag)
		siz := proto.Size(m)
		b = protowire.AppendVarint(b, uint64(siz))
		b, err = opts.Options().MarshalAppend(b, m)
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func consumeMessageSlice(b []byte, p pointer, goType reflect.Type, wtyp protowire.Type, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.BytesType {
		return out, errUnknown
	}
	v, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return out, errDecode
	}
	mp := reflect.New(goType.Elem())
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     v,
		Message: asMessage(mp).ProtoReflect(),
	})
	if err != nil {
		return out, err
	}
	p.AppendPointerSlice(pointerOfValue(mp))
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return out, nil
}

func isInitMessageSlice(p pointer, goType reflect.Type) error {
	s := p.PointerSlice()
	for _, v := range s {
		m := asMessage(v.AsValueOf(goType.Elem()))
		if err := proto.CheckInitialized(m); err != nil {
			return err
		}
	}
	return nil
}

// Slices of messages

func sizeMessageSliceValue(listv protoreflect.Value, tagsize int, opts marshalOptions) int {
	list := listv.List()
	n := 0
	for i, llen := 0, list.Len(); i < llen; i++ {
		m := list.Get(i).Message().Interface()
		n += protowire.SizeBytes(proto.Size(m)) + tagsize
	}
	return n
}

func appendMessageSliceValue(b []byte, listv protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	list := listv.List()
	mopts := opts.Options()
	for i, llen := 0, list.Len(); i < llen; i++ {
		m := list.Get(i).Message().Interface()
		b = protowire.AppendVarint(b, wiretag)
		siz := proto.Size(m)
		b = protowire.AppendVarint(b, uint64(siz))
		var err error
		b, err = mopts.MarshalAppend(b, m)
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func consumeMessageSliceValue(b []byte, listv protoreflect.Value, _ protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (_ protoreflect.Value, out unmarshalOutput, err error) {
	list := listv.List()
	if wtyp != protowire.BytesType {
		return protoreflect.Value{}, out, errUnknown
	}
	v, n := protowire.ConsumeBytes(b)
	if n < 0 {
		return protoreflect.Value{}, out, errDecode
	}
	m := list.NewElement()
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     v,
		Message: m.Message(),
	})
	if err != nil {
		return protoreflect.Value{}, out, err
	}
	list.Append(m)
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return listv, out, nil
}

func isInitMessageSliceValue(listv protoreflect.Value) error {
	list := listv.List()
	for i, llen := 0, list.Len(); i < llen; i++ {
		m := list.Get(i).Message().Interface()
		if err := proto.CheckInitialized(m); err != nil {
			return err
		}
	}
	return nil
}

var coderMessageSliceValue = valueCoderFuncs{
	size:      sizeMessageSliceValue,
	marshal:   appendMessageSliceValue,
	unmarshal: consumeMessageSliceValue,
	isInit:    isInitMessageSliceValue,
	merge:     mergeMessageListValue,
}

func sizeGroupSliceValue(listv protoreflect.Value, tagsize int, opts marshalOptions) int {
	list := listv.List()
	n := 0
	for i, llen := 0, list.Len(); i < llen; i++ {
		m := list.Get(i).Message().Interface()
		n += 2*tagsize + proto.Size(m)
	}
	return n
}

func appendGroupSliceValue(b []byte, listv protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error) {
	list := listv.List()
	mopts := opts.Options()
	for i, llen := 0, list.Len(); i < llen; i++ {
		m := list.Get(i).Message().Interface()
		b = protowire.AppendVarint(b, wiretag) // start group
		var err error
		b, err = mopts.MarshalAppend(b, m)
		if err != nil {
			return b, err
		}
		b = protowire.AppendVarint(b, wiretag+1) // end group
	}
	return b, nil
}

func consumeGroupSliceValue(b []byte, listv protoreflect.Value, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (_ protoreflect.Value, out unmarshalOutput, err error) {
	list := listv.List()
	if wtyp != protowire.StartGroupType {
		return protoreflect.Value{}, out, errUnknown
	}
	b, n := protowire.ConsumeGroup(num, b)
	if n < 0 {
		return protoreflect.Value{}, out, errDecode
	}
	m := list.NewElement()
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     b,
		Message: m.Message(),
	})
	if err != nil {
		return protoreflect.Value{}, out, err
	}
	list.Append(m)
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return listv, out, nil
}

var coderGroupSliceValue = valueCoderFuncs{
	size:      sizeGroupSliceValue,
	marshal:   appendGroupSliceValue,
	unmarshal: consumeGroupSliceValue,
	isInit:    isInitMessageSliceValue,
	merge:     mergeMessageListValue,
}

func makeGroupSliceFieldCoder(fd protoreflect.FieldDescriptor, ft reflect.Type) pointerCoderFuncs {
	num := fd.Number()
	if mi := getMessageInfo(ft); mi != nil {
		funcs := pointerCoderFuncs{
			size:      sizeGroupSliceInfo,
			marshal:   appendGroupSliceInfo,
			unmarshal: consumeGroupSliceInfo,
			merge:     mergeMessageSlice,
		}
		if needsInitCheck(mi.Desc) {
			funcs.isInit = isInitMessageSliceInfo
		}
		return funcs
	}
	return pointerCoderFuncs{
		size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
			return sizeGroupSlice(p, ft, f.tagsize, opts)
		},
		marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
			return appendGroupSlice(b, p, f.wiretag, ft, opts)
		},
		unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
			return consumeGroupSlice(b, p, num, wtyp, ft, opts)
		},
		isInit: func(p pointer, f *coderFieldInfo) error {
			return isInitMessageSlice(p, ft)
		},
		merge: mergeMessageSlice,
	}
}

func sizeGroupSlice(p pointer, messageType reflect.Type, tagsize int, _ marshalOptions) int {
	s := p.PointerSlice()
	n := 0
	for _, v := range s {
		m := asMessage(v.AsValueOf(messageType.Elem()))
		n += 2*tagsize + proto.Size(m)
	}
	return n
}

func appendGroupSlice(b []byte, p pointer, wiretag uint64, messageType reflect.Type, opts marshalOptions) ([]byte, error) {
	s := p.PointerSlice()
	var err error
	for _, v := range s {
		m := asMessage(v.AsValueOf(messageType.Elem()))
		b = protowire.AppendVarint(b, wiretag) // start group
		b, err = opts.Options().MarshalAppend(b, m)
		if err != nil {
			return b, err
		}
		b = protowire.AppendVarint(b, wiretag+1) // end group
	}
	return b, nil
}

func consumeGroupSlice(b []byte, p pointer, num protowire.Number, wtyp protowire.Type, goType reflect.Type, opts unmarshalOptions) (out unmarshalOutput, err error) {
	if wtyp != protowire.StartGroupType {
		return out, errUnknown
	}
	b, n := protowire.ConsumeGroup(num, b)
	if n < 0 {
		return out, errDecode
	}
	mp := reflect.New(goType.Elem())
	o, err := opts.Options().UnmarshalState(protoiface.UnmarshalInput{
		Buf:     b,
		Message: asMessage(mp).ProtoReflect(),
	})
	if err != nil {
		return out, err
	}
	p.AppendPointerSlice(pointerOfValue(mp))
	out.n = n
	out.initialized = o.Flags&protoiface.UnmarshalInitialized != 0
	return out, nil
}

func sizeGroupSliceInfo(p pointer, f *coderFieldInfo, opts marshalOptions) int {
	s := p.PointerSlice()
	n := 0
	for _, v := range s {
		n += 2*f.tagsize + f.mi.sizePointer(v, opts)
	}
	return n
}

func appendGroupSliceInfo(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
	s := p.PointerSlice()
	var err error
	for _, v := range s {
		b = protowire.AppendVarint(b, f.wiretag) // start group
		b, err = f.mi.marshalAppendPointer(b, v, opts)
		if err != nil {
			return b, err
		}
		b = protowire.AppendVarint(b, f.wiretag+1) // end group
	}
	return b, nil
}

func consumeGroupSliceInfo(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
	if wtyp != protowire.StartGroupType {
		return unmarshalOutput{}, errUnknown
	}
	m := reflect.New(f.mi.GoReflectType.Elem()).Interface()
	mp := pointerOfIface(m)
	out, err := f.mi.unmarshalPointer(b, mp, f.num, opts)
	if err != nil {
		return out, err
	}
	p.AppendPointerSlice(mp)
	return out, nil
}

func asMessage(v reflect.Value) protoreflect.ProtoMessage {
	if m, ok := v.Interface().(protoreflect.ProtoMessage); ok {
		return m
	}
	return legacyWrapMessage(v).Interface()
}
