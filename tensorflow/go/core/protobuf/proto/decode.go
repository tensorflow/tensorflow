// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
)

// UnmarshalOptions configures the unmarshaler.
//
// Example usage:
//
//	err := UnmarshalOptions{DiscardUnknown: true}.Unmarshal(b, m)
type UnmarshalOptions struct {
	pragma.NoUnkeyedLiterals

	// Merge merges the input into the destination message.
	// The default behavior is to always reset the message before unmarshaling,
	// unless Merge is specified.
	Merge bool

	// AllowPartial accepts input for messages that will result in missing
	// required fields. If AllowPartial is false (the default), Unmarshal will
	// return an error if there are any missing required fields.
	AllowPartial bool

	// If DiscardUnknown is set, unknown fields are ignored.
	DiscardUnknown bool

	// Resolver is used for looking up types when unmarshaling extension fields.
	// If nil, this defaults to using protoregistry.GlobalTypes.
	Resolver interface {
		FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error)
		FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error)
	}

	// RecursionLimit limits how deeply messages may be nested.
	// If zero, a default limit is applied.
	RecursionLimit int
}

// Unmarshal parses the wire-format message in b and places the result in m.
// The provided message must be mutable (e.g., a non-nil pointer to a message).
func Unmarshal(b []byte, m Message) error {
	_, err := UnmarshalOptions{RecursionLimit: protowire.DefaultRecursionLimit}.unmarshal(b, m.ProtoReflect())
	return err
}

// Unmarshal parses the wire-format message in b and places the result in m.
// The provided message must be mutable (e.g., a non-nil pointer to a message).
func (o UnmarshalOptions) Unmarshal(b []byte, m Message) error {
	if o.RecursionLimit == 0 {
		o.RecursionLimit = protowire.DefaultRecursionLimit
	}
	_, err := o.unmarshal(b, m.ProtoReflect())
	return err
}

// UnmarshalState parses a wire-format message and places the result in m.
//
// This method permits fine-grained control over the unmarshaler.
// Most users should use [Unmarshal] instead.
func (o UnmarshalOptions) UnmarshalState(in protoiface.UnmarshalInput) (protoiface.UnmarshalOutput, error) {
	if o.RecursionLimit == 0 {
		o.RecursionLimit = protowire.DefaultRecursionLimit
	}
	return o.unmarshal(in.Buf, in.Message)
}

// unmarshal is a centralized function that all unmarshal operations go through.
// For profiling purposes, avoid changing the name of this function or
// introducing other code paths for unmarshal that do not go through this.
func (o UnmarshalOptions) unmarshal(b []byte, m protoreflect.Message) (out protoiface.UnmarshalOutput, err error) {
	if o.Resolver == nil {
		o.Resolver = protoregistry.GlobalTypes
	}
	if !o.Merge {
		Reset(m.Interface())
	}
	allowPartial := o.AllowPartial
	o.Merge = true
	o.AllowPartial = true
	methods := protoMethods(m)
	if methods != nil && methods.Unmarshal != nil &&
		!(o.DiscardUnknown && methods.Flags&protoiface.SupportUnmarshalDiscardUnknown == 0) {
		in := protoiface.UnmarshalInput{
			Message:  m,
			Buf:      b,
			Resolver: o.Resolver,
			Depth:    o.RecursionLimit,
		}
		if o.DiscardUnknown {
			in.Flags |= protoiface.UnmarshalDiscardUnknown
		}
		out, err = methods.Unmarshal(in)
	} else {
		o.RecursionLimit--
		if o.RecursionLimit < 0 {
			return out, errors.New("exceeded max recursion depth")
		}
		err = o.unmarshalMessageSlow(b, m)
	}
	if err != nil {
		return out, err
	}
	if allowPartial || (out.Flags&protoiface.UnmarshalInitialized != 0) {
		return out, nil
	}
	return out, checkInitialized(m)
}

func (o UnmarshalOptions) unmarshalMessage(b []byte, m protoreflect.Message) error {
	_, err := o.unmarshal(b, m)
	return err
}

func (o UnmarshalOptions) unmarshalMessageSlow(b []byte, m protoreflect.Message) error {
	md := m.Descriptor()
	if messageset.IsMessageSet(md) {
		return o.unmarshalMessageSet(b, m)
	}
	fields := md.Fields()
	for len(b) > 0 {
		// Parse the tag (field number and wire type).
		num, wtyp, tagLen := protowire.ConsumeTag(b)
		if tagLen < 0 {
			return errDecode
		}
		if num > protowire.MaxValidNumber {
			return errDecode
		}

		// Find the field descriptor for this field number.
		fd := fields.ByNumber(num)
		if fd == nil && md.ExtensionRanges().Has(num) {
			extType, err := o.Resolver.FindExtensionByNumber(md.FullName(), num)
			if err != nil && err != protoregistry.NotFound {
				return errors.New("%v: unable to resolve extension %v: %v", md.FullName(), num, err)
			}
			if extType != nil {
				fd = extType.TypeDescriptor()
			}
		}
		var err error
		if fd == nil {
			err = errUnknown
		} else if flags.ProtoLegacy {
			if fd.IsWeak() && fd.Message().IsPlaceholder() {
				err = errUnknown // weak referent is not linked in
			}
		}

		// Parse the field value.
		var valLen int
		switch {
		case err != nil:
		case fd.IsList():
			valLen, err = o.unmarshalList(b[tagLen:], wtyp, m.Mutable(fd).List(), fd)
		case fd.IsMap():
			valLen, err = o.unmarshalMap(b[tagLen:], wtyp, m.Mutable(fd).Map(), fd)
		default:
			valLen, err = o.unmarshalSingular(b[tagLen:], wtyp, m, fd)
		}
		if err != nil {
			if err != errUnknown {
				return err
			}
			valLen = protowire.ConsumeFieldValue(num, wtyp, b[tagLen:])
			if valLen < 0 {
				return errDecode
			}
			if !o.DiscardUnknown {
				m.SetUnknown(append(m.GetUnknown(), b[:tagLen+valLen]...))
			}
		}
		b = b[tagLen+valLen:]
	}
	return nil
}

func (o UnmarshalOptions) unmarshalSingular(b []byte, wtyp protowire.Type, m protoreflect.Message, fd protoreflect.FieldDescriptor) (n int, err error) {
	v, n, err := o.unmarshalScalar(b, wtyp, fd)
	if err != nil {
		return 0, err
	}
	switch fd.Kind() {
	case protoreflect.GroupKind, protoreflect.MessageKind:
		m2 := m.Mutable(fd).Message()
		if err := o.unmarshalMessage(v.Bytes(), m2); err != nil {
			return n, err
		}
	default:
		// Non-message scalars replace the previous value.
		m.Set(fd, v)
	}
	return n, nil
}

func (o UnmarshalOptions) unmarshalMap(b []byte, wtyp protowire.Type, mapv protoreflect.Map, fd protoreflect.FieldDescriptor) (n int, err error) {
	if wtyp != protowire.BytesType {
		return 0, errUnknown
	}
	b, n = protowire.ConsumeBytes(b)
	if n < 0 {
		return 0, errDecode
	}
	var (
		keyField = fd.MapKey()
		valField = fd.MapValue()
		key      protoreflect.Value
		val      protoreflect.Value
		haveKey  bool
		haveVal  bool
	)
	switch valField.Kind() {
	case protoreflect.GroupKind, protoreflect.MessageKind:
		val = mapv.NewValue()
	}
	// Map entries are represented as a two-element message with fields
	// containing the key and value.
	for len(b) > 0 {
		num, wtyp, n := protowire.ConsumeTag(b)
		if n < 0 {
			return 0, errDecode
		}
		if num > protowire.MaxValidNumber {
			return 0, errDecode
		}
		b = b[n:]
		err = errUnknown
		switch num {
		case genid.MapEntry_Key_field_number:
			key, n, err = o.unmarshalScalar(b, wtyp, keyField)
			if err != nil {
				break
			}
			haveKey = true
		case genid.MapEntry_Value_field_number:
			var v protoreflect.Value
			v, n, err = o.unmarshalScalar(b, wtyp, valField)
			if err != nil {
				break
			}
			switch valField.Kind() {
			case protoreflect.GroupKind, protoreflect.MessageKind:
				if err := o.unmarshalMessage(v.Bytes(), val.Message()); err != nil {
					return 0, err
				}
			default:
				val = v
			}
			haveVal = true
		}
		if err == errUnknown {
			n = protowire.ConsumeFieldValue(num, wtyp, b)
			if n < 0 {
				return 0, errDecode
			}
		} else if err != nil {
			return 0, err
		}
		b = b[n:]
	}
	// Every map entry should have entries for key and value, but this is not strictly required.
	if !haveKey {
		key = keyField.Default()
	}
	if !haveVal {
		switch valField.Kind() {
		case protoreflect.GroupKind, protoreflect.MessageKind:
		default:
			val = valField.Default()
		}
	}
	mapv.Set(key.MapKey(), val)
	return n, nil
}

// errUnknown is used internally to indicate fields which should be added
// to the unknown field set of a message. It is never returned from an exported
// function.
var errUnknown = errors.New("BUG: internal error (unknown)")

var errDecode = errors.New("cannot parse invalid wire-format data")
