// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"fmt"

	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

// Merge merges src into dst, which must be a message with the same descriptor.
//
// Populated scalar fields in src are copied to dst, while populated
// singular messages in src are merged into dst by recursively calling Merge.
// The elements of every list field in src is appended to the corresponded
// list fields in dst. The entries of every map field in src is copied into
// the corresponding map field in dst, possibly replacing existing entries.
// The unknown fields of src are appended to the unknown fields of dst.
//
// It is semantically equivalent to unmarshaling the encoded form of src
// into dst with the [UnmarshalOptions.Merge] option specified.
func Merge(dst, src Message) {
	// TODO: Should nil src be treated as semantically equivalent to a
	// untyped, read-only, empty message? What about a nil dst?

	dstMsg, srcMsg := dst.ProtoReflect(), src.ProtoReflect()
	if dstMsg.Descriptor() != srcMsg.Descriptor() {
		if got, want := dstMsg.Descriptor().FullName(), srcMsg.Descriptor().FullName(); got != want {
			panic(fmt.Sprintf("descriptor mismatch: %v != %v", got, want))
		}
		panic("descriptor mismatch")
	}
	mergeOptions{}.mergeMessage(dstMsg, srcMsg)
}

// Clone returns a deep copy of m.
// If the top-level message is invalid, it returns an invalid message as well.
func Clone(m Message) Message {
	// NOTE: Most usages of Clone assume the following properties:
	//	t := reflect.TypeOf(m)
	//	t == reflect.TypeOf(m.ProtoReflect().New().Interface())
	//	t == reflect.TypeOf(m.ProtoReflect().Type().Zero().Interface())
	//
	// Embedding protobuf messages breaks this since the parent type will have
	// a forwarded ProtoReflect method, but the Interface method will return
	// the underlying embedded message type.
	if m == nil {
		return nil
	}
	src := m.ProtoReflect()
	if !src.IsValid() {
		return src.Type().Zero().Interface()
	}
	dst := src.New()
	mergeOptions{}.mergeMessage(dst, src)
	return dst.Interface()
}

// mergeOptions provides a namespace for merge functions, and can be
// exported in the future if we add user-visible merge options.
type mergeOptions struct{}

func (o mergeOptions) mergeMessage(dst, src protoreflect.Message) {
	methods := protoMethods(dst)
	if methods != nil && methods.Merge != nil {
		in := protoiface.MergeInput{
			Destination: dst,
			Source:      src,
		}
		out := methods.Merge(in)
		if out.Flags&protoiface.MergeComplete != 0 {
			return
		}
	}

	if !dst.IsValid() {
		panic(fmt.Sprintf("cannot merge into invalid %v message", dst.Descriptor().FullName()))
	}

	src.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		switch {
		case fd.IsList():
			o.mergeList(dst.Mutable(fd).List(), v.List(), fd)
		case fd.IsMap():
			o.mergeMap(dst.Mutable(fd).Map(), v.Map(), fd.MapValue())
		case fd.Message() != nil:
			o.mergeMessage(dst.Mutable(fd).Message(), v.Message())
		case fd.Kind() == protoreflect.BytesKind:
			dst.Set(fd, o.cloneBytes(v))
		default:
			dst.Set(fd, v)
		}
		return true
	})

	if len(src.GetUnknown()) > 0 {
		dst.SetUnknown(append(dst.GetUnknown(), src.GetUnknown()...))
	}
}

func (o mergeOptions) mergeList(dst, src protoreflect.List, fd protoreflect.FieldDescriptor) {
	// Merge semantics appends to the end of the existing list.
	for i, n := 0, src.Len(); i < n; i++ {
		switch v := src.Get(i); {
		case fd.Message() != nil:
			dstv := dst.NewElement()
			o.mergeMessage(dstv.Message(), v.Message())
			dst.Append(dstv)
		case fd.Kind() == protoreflect.BytesKind:
			dst.Append(o.cloneBytes(v))
		default:
			dst.Append(v)
		}
	}
}

func (o mergeOptions) mergeMap(dst, src protoreflect.Map, fd protoreflect.FieldDescriptor) {
	// Merge semantics replaces, rather than merges into existing entries.
	src.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		switch {
		case fd.Message() != nil:
			dstv := dst.NewValue()
			o.mergeMessage(dstv.Message(), v.Message())
			dst.Set(k, dstv)
		case fd.Kind() == protoreflect.BytesKind:
			dst.Set(k, o.cloneBytes(v))
		default:
			dst.Set(k, v)
		}
		return true
	})
}

func (o mergeOptions) cloneBytes(v protoreflect.Value) protoreflect.Value {
	return protoreflect.ValueOfBytes(append([]byte{}, v.Bytes()...))
}
