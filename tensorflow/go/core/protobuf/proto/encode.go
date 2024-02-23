// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/order"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

// MarshalOptions configures the marshaler.
//
// Example usage:
//
//	b, err := MarshalOptions{Deterministic: true}.Marshal(m)
type MarshalOptions struct {
	pragma.NoUnkeyedLiterals

	// AllowPartial allows messages that have missing required fields to marshal
	// without returning an error. If AllowPartial is false (the default),
	// Marshal will return an error if there are any missing required fields.
	AllowPartial bool

	// Deterministic controls whether the same message will always be
	// serialized to the same bytes within the same binary.
	//
	// Setting this option guarantees that repeated serialization of
	// the same message will return the same bytes, and that different
	// processes of the same binary (which may be executing on different
	// machines) will serialize equal messages to the same bytes.
	// It has no effect on the resulting size of the encoded message compared
	// to a non-deterministic marshal.
	//
	// Note that the deterministic serialization is NOT canonical across
	// languages. It is not guaranteed to remain stable over time. It is
	// unstable across different builds with schema changes due to unknown
	// fields. Users who need canonical serialization (e.g., persistent
	// storage in a canonical form, fingerprinting, etc.) must define
	// their own canonicalization specification and implement their own
	// serializer rather than relying on this API.
	//
	// If deterministic serialization is requested, map entries will be
	// sorted by keys in lexographical order. This is an implementation
	// detail and subject to change.
	Deterministic bool

	// UseCachedSize indicates that the result of a previous Size call
	// may be reused.
	//
	// Setting this option asserts that:
	//
	// 1. Size has previously been called on this message with identical
	// options (except for UseCachedSize itself).
	//
	// 2. The message and all its submessages have not changed in any
	// way since the Size call.
	//
	// If either of these invariants is violated,
	// the results are undefined and may include panics or corrupted output.
	//
	// Implementations MAY take this option into account to provide
	// better performance, but there is no guarantee that they will do so.
	// There is absolutely no guarantee that Size followed by Marshal with
	// UseCachedSize set will perform equivalently to Marshal alone.
	UseCachedSize bool
}

// Marshal returns the wire-format encoding of m.
func Marshal(m Message) ([]byte, error) {
	// Treat nil message interface as an empty message; nothing to output.
	if m == nil {
		return nil, nil
	}

	out, err := MarshalOptions{}.marshal(nil, m.ProtoReflect())
	if len(out.Buf) == 0 && err == nil {
		out.Buf = emptyBytesForMessage(m)
	}
	return out.Buf, err
}

// Marshal returns the wire-format encoding of m.
func (o MarshalOptions) Marshal(m Message) ([]byte, error) {
	// Treat nil message interface as an empty message; nothing to output.
	if m == nil {
		return nil, nil
	}

	out, err := o.marshal(nil, m.ProtoReflect())
	if len(out.Buf) == 0 && err == nil {
		out.Buf = emptyBytesForMessage(m)
	}
	return out.Buf, err
}

// emptyBytesForMessage returns a nil buffer if and only if m is invalid,
// otherwise it returns a non-nil empty buffer.
//
// This is to assist the edge-case where user-code does the following:
//
//	m1.OptionalBytes, _ = proto.Marshal(m2)
//
// where they expect the proto2 "optional_bytes" field to be populated
// if any only if m2 is a valid message.
func emptyBytesForMessage(m Message) []byte {
	if m == nil || !m.ProtoReflect().IsValid() {
		return nil
	}
	return emptyBuf[:]
}

// MarshalAppend appends the wire-format encoding of m to b,
// returning the result.
func (o MarshalOptions) MarshalAppend(b []byte, m Message) ([]byte, error) {
	// Treat nil message interface as an empty message; nothing to append.
	if m == nil {
		return b, nil
	}

	out, err := o.marshal(b, m.ProtoReflect())
	return out.Buf, err
}

// MarshalState returns the wire-format encoding of a message.
//
// This method permits fine-grained control over the marshaler.
// Most users should use [Marshal] instead.
func (o MarshalOptions) MarshalState(in protoiface.MarshalInput) (protoiface.MarshalOutput, error) {
	return o.marshal(in.Buf, in.Message)
}

// marshal is a centralized function that all marshal operations go through.
// For profiling purposes, avoid changing the name of this function or
// introducing other code paths for marshal that do not go through this.
func (o MarshalOptions) marshal(b []byte, m protoreflect.Message) (out protoiface.MarshalOutput, err error) {
	allowPartial := o.AllowPartial
	o.AllowPartial = true
	if methods := protoMethods(m); methods != nil && methods.Marshal != nil &&
		!(o.Deterministic && methods.Flags&protoiface.SupportMarshalDeterministic == 0) {
		in := protoiface.MarshalInput{
			Message: m,
			Buf:     b,
		}
		if o.Deterministic {
			in.Flags |= protoiface.MarshalDeterministic
		}
		if o.UseCachedSize {
			in.Flags |= protoiface.MarshalUseCachedSize
		}
		if methods.Size != nil {
			sout := methods.Size(protoiface.SizeInput{
				Message: m,
				Flags:   in.Flags,
			})
			if cap(b) < len(b)+sout.Size {
				in.Buf = make([]byte, len(b), growcap(cap(b), len(b)+sout.Size))
				copy(in.Buf, b)
			}
			in.Flags |= protoiface.MarshalUseCachedSize
		}
		out, err = methods.Marshal(in)
	} else {
		out.Buf, err = o.marshalMessageSlow(b, m)
	}
	if err != nil {
		return out, err
	}
	if allowPartial {
		return out, nil
	}
	return out, checkInitialized(m)
}

func (o MarshalOptions) marshalMessage(b []byte, m protoreflect.Message) ([]byte, error) {
	out, err := o.marshal(b, m)
	return out.Buf, err
}

// growcap scales up the capacity of a slice.
//
// Given a slice with a current capacity of oldcap and a desired
// capacity of wantcap, growcap returns a new capacity >= wantcap.
//
// The algorithm is mostly identical to the one used by append as of Go 1.14.
func growcap(oldcap, wantcap int) (newcap int) {
	if wantcap > oldcap*2 {
		newcap = wantcap
	} else if oldcap < 1024 {
		// The Go 1.14 runtime takes this case when len(s) < 1024,
		// not when cap(s) < 1024. The difference doesn't seem
		// significant here.
		newcap = oldcap * 2
	} else {
		newcap = oldcap
		for 0 < newcap && newcap < wantcap {
			newcap += newcap / 4
		}
		if newcap <= 0 {
			newcap = wantcap
		}
	}
	return newcap
}

func (o MarshalOptions) marshalMessageSlow(b []byte, m protoreflect.Message) ([]byte, error) {
	if messageset.IsMessageSet(m.Descriptor()) {
		return o.marshalMessageSet(b, m)
	}
	fieldOrder := order.AnyFieldOrder
	if o.Deterministic {
		// TODO: This should use a more natural ordering like NumberFieldOrder,
		// but doing so breaks golden tests that make invalid assumption about
		// output stability of this implementation.
		fieldOrder = order.LegacyFieldOrder
	}
	var err error
	order.RangeFields(m, fieldOrder, func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		b, err = o.marshalField(b, fd, v)
		return err == nil
	})
	if err != nil {
		return b, err
	}
	b = append(b, m.GetUnknown()...)
	return b, nil
}

func (o MarshalOptions) marshalField(b []byte, fd protoreflect.FieldDescriptor, value protoreflect.Value) ([]byte, error) {
	switch {
	case fd.IsList():
		return o.marshalList(b, fd, value.List())
	case fd.IsMap():
		return o.marshalMap(b, fd, value.Map())
	default:
		b = protowire.AppendTag(b, fd.Number(), wireTypes[fd.Kind()])
		return o.marshalSingular(b, fd, value)
	}
}

func (o MarshalOptions) marshalList(b []byte, fd protoreflect.FieldDescriptor, list protoreflect.List) ([]byte, error) {
	if fd.IsPacked() && list.Len() > 0 {
		b = protowire.AppendTag(b, fd.Number(), protowire.BytesType)
		b, pos := appendSpeculativeLength(b)
		for i, llen := 0, list.Len(); i < llen; i++ {
			var err error
			b, err = o.marshalSingular(b, fd, list.Get(i))
			if err != nil {
				return b, err
			}
		}
		b = finishSpeculativeLength(b, pos)
		return b, nil
	}

	kind := fd.Kind()
	for i, llen := 0, list.Len(); i < llen; i++ {
		var err error
		b = protowire.AppendTag(b, fd.Number(), wireTypes[kind])
		b, err = o.marshalSingular(b, fd, list.Get(i))
		if err != nil {
			return b, err
		}
	}
	return b, nil
}

func (o MarshalOptions) marshalMap(b []byte, fd protoreflect.FieldDescriptor, mapv protoreflect.Map) ([]byte, error) {
	keyf := fd.MapKey()
	valf := fd.MapValue()
	keyOrder := order.AnyKeyOrder
	if o.Deterministic {
		keyOrder = order.GenericKeyOrder
	}
	var err error
	order.RangeEntries(mapv, keyOrder, func(key protoreflect.MapKey, value protoreflect.Value) bool {
		b = protowire.AppendTag(b, fd.Number(), protowire.BytesType)
		var pos int
		b, pos = appendSpeculativeLength(b)

		b, err = o.marshalField(b, keyf, key.Value())
		if err != nil {
			return false
		}
		b, err = o.marshalField(b, valf, value)
		if err != nil {
			return false
		}
		b = finishSpeculativeLength(b, pos)
		return true
	})
	return b, err
}

// When encoding length-prefixed fields, we speculatively set aside some number of bytes
// for the length, encode the data, and then encode the length (shifting the data if necessary
// to make room).
const speculativeLength = 1

func appendSpeculativeLength(b []byte) ([]byte, int) {
	pos := len(b)
	b = append(b, "\x00\x00\x00\x00"[:speculativeLength]...)
	return b, pos
}

func finishSpeculativeLength(b []byte, pos int) []byte {
	mlen := len(b) - pos - speculativeLength
	msiz := protowire.SizeVarint(uint64(mlen))
	if msiz != speculativeLength {
		for i := 0; i < msiz-speculativeLength; i++ {
			b = append(b, 0)
		}
		copy(b[pos+msiz:], b[pos+speculativeLength:])
		b = b[:pos+msiz+mlen]
	}
	protowire.AppendVarint(b[:pos], uint64(mlen))
	return b
}
