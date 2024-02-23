// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protorange provides functionality to traverse a message value.
package protorange

import (
	"bytes"
	"errors"

	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/order"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protopath"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

var (
	// Break breaks traversal of children in the current value.
	// It has no effect when traversing values that are not composite types
	// (e.g., messages, lists, and maps).
	Break = errors.New("break traversal of children in current value")

	// Terminate terminates the entire range operation.
	// All necessary Pop operations continue to be called.
	Terminate = errors.New("terminate range operation")
)

// Range performs a depth-first traversal over reachable values in a message.
//
// See [Options.Range] for details.
func Range(m protoreflect.Message, f func(protopath.Values) error) error {
	return Options{}.Range(m, f, nil)
}

// Options configures traversal of a message value tree.
type Options struct {
	// Stable specifies whether to visit message fields and map entries
	// in a stable ordering. If false, then the ordering is undefined and
	// may be non-deterministic.
	//
	// Message fields are visited in ascending order by field number.
	// Map entries are visited in ascending order, where
	// boolean keys are ordered such that false sorts before true,
	// numeric keys are ordered based on the numeric value, and
	// string keys are lexicographically ordered by Unicode codepoints.
	Stable bool

	// Resolver is used for looking up types when expanding google.protobuf.Any
	// messages. If nil, this defaults to using protoregistry.GlobalTypes.
	// To prevent expansion of Any messages, pass an empty protoregistry.Types:
	//
	//	Options{Resolver: (*protoregistry.Types)(nil)}
	//
	Resolver interface {
		protoregistry.ExtensionTypeResolver
		protoregistry.MessageTypeResolver
	}
}

// Range performs a depth-first traversal over reachable values in a message.
// The first push and the last pop are to push/pop a [protopath.Root] step.
// If push or pop return any non-nil error (other than [Break] or [Terminate]),
// it terminates the traversal and is returned by Range.
//
// The rules for traversing a message is as follows:
//
//   - For messages, iterate over every populated known and extension field.
//     Each field is preceded by a push of a [protopath.FieldAccess] step,
//     followed by recursive application of the rules on the field value,
//     and succeeded by a pop of that step.
//     If the message has unknown fields, then push an [protopath.UnknownAccess] step
//     followed immediately by pop of that step.
//
//   - As an exception to the above rule, if the current message is a
//     google.protobuf.Any message, expand the underlying message (if resolvable).
//     The expanded message is preceded by a push of a [protopath.AnyExpand] step,
//     followed by recursive application of the rules on the underlying message,
//     and succeeded by a pop of that step. Mutations to the expanded message
//     are written back to the Any message when popping back out.
//
//   - For lists, iterate over every element. Each element is preceded by a push
//     of a [protopath.ListIndex] step, followed by recursive application of the rules
//     on the list element, and succeeded by a pop of that step.
//
//   - For maps, iterate over every entry. Each entry is preceded by a push
//     of a [protopath.MapIndex] step, followed by recursive application of the rules
//     on the map entry value, and succeeded by a pop of that step.
//
// Mutations should only be made to the last value, otherwise the effects on
// traversal will be undefined. If the mutation is made to the last value
// during to a push, then the effects of the mutation will affect traversal.
// For example, if the last value is currently a message, and the push function
// populates a few fields in that message, then the newly modified fields
// will be traversed.
//
// The [protopath.Values] provided to push functions is only valid until the
// corresponding pop call and the values provided to a pop call is only valid
// for the duration of the pop call itself.
func (o Options) Range(m protoreflect.Message, push, pop func(protopath.Values) error) error {
	var err error
	p := new(protopath.Values)
	if o.Resolver == nil {
		o.Resolver = protoregistry.GlobalTypes
	}

	pushStep(p, protopath.Root(m.Descriptor()), protoreflect.ValueOfMessage(m))
	if push != nil {
		err = amendError(err, push(*p))
	}
	if err == nil {
		err = o.rangeMessage(p, m, push, pop)
	}
	if pop != nil {
		err = amendError(err, pop(*p))
	}
	popStep(p)

	if err == Break || err == Terminate {
		err = nil
	}
	return err
}

func (o Options) rangeMessage(p *protopath.Values, m protoreflect.Message, push, pop func(protopath.Values) error) (err error) {
	if ok, err := o.rangeAnyMessage(p, m, push, pop); ok {
		return err
	}

	fieldOrder := order.AnyFieldOrder
	if o.Stable {
		fieldOrder = order.NumberFieldOrder
	}
	order.RangeFields(m, fieldOrder, func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		pushStep(p, protopath.FieldAccess(fd), v)
		if push != nil {
			err = amendError(err, push(*p))
		}
		if err == nil {
			switch {
			case fd.IsMap():
				err = o.rangeMap(p, fd, v.Map(), push, pop)
			case fd.IsList():
				err = o.rangeList(p, fd, v.List(), push, pop)
			case fd.Message() != nil:
				err = o.rangeMessage(p, v.Message(), push, pop)
			}
		}
		if pop != nil {
			err = amendError(err, pop(*p))
		}
		popStep(p)
		return err == nil
	})

	if b := m.GetUnknown(); len(b) > 0 && err == nil {
		pushStep(p, protopath.UnknownAccess(), protoreflect.ValueOfBytes(b))
		if push != nil {
			err = amendError(err, push(*p))
		}
		if pop != nil {
			err = amendError(err, pop(*p))
		}
		popStep(p)
	}

	if err == Break {
		err = nil
	}
	return err
}

func (o Options) rangeAnyMessage(p *protopath.Values, m protoreflect.Message, push, pop func(protopath.Values) error) (ok bool, err error) {
	md := m.Descriptor()
	if md.FullName() != "google.protobuf.Any" {
		return false, nil
	}

	fds := md.Fields()
	url := m.Get(fds.ByNumber(genid.Any_TypeUrl_field_number)).String()
	val := m.Get(fds.ByNumber(genid.Any_Value_field_number)).Bytes()
	mt, errFind := o.Resolver.FindMessageByURL(url)
	if errFind != nil {
		return false, nil
	}

	// Unmarshal the raw encoded message value into a structured message value.
	m2 := mt.New()
	errUnmarshal := proto.UnmarshalOptions{
		Merge:        true,
		AllowPartial: true,
		Resolver:     o.Resolver,
	}.Unmarshal(val, m2.Interface())
	if errUnmarshal != nil {
		// If the the underlying message cannot be unmarshaled,
		// then just treat this as an normal message type.
		return false, nil
	}

	// Marshal Any before ranging to detect possible mutations.
	b1, errMarshal := proto.MarshalOptions{
		AllowPartial:  true,
		Deterministic: true,
	}.Marshal(m2.Interface())
	if errMarshal != nil {
		return true, errMarshal
	}

	pushStep(p, protopath.AnyExpand(m2.Descriptor()), protoreflect.ValueOfMessage(m2))
	if push != nil {
		err = amendError(err, push(*p))
	}
	if err == nil {
		err = o.rangeMessage(p, m2, push, pop)
	}
	if pop != nil {
		err = amendError(err, pop(*p))
	}
	popStep(p)

	// Marshal Any after ranging to detect possible mutations.
	b2, errMarshal := proto.MarshalOptions{
		AllowPartial:  true,
		Deterministic: true,
	}.Marshal(m2.Interface())
	if errMarshal != nil {
		return true, errMarshal
	}

	// Mutations detected, write the new sequence of bytes to the Any message.
	if !bytes.Equal(b1, b2) {
		m.Set(fds.ByNumber(genid.Any_Value_field_number), protoreflect.ValueOfBytes(b2))
	}

	if err == Break {
		err = nil
	}
	return true, err
}

func (o Options) rangeList(p *protopath.Values, fd protoreflect.FieldDescriptor, ls protoreflect.List, push, pop func(protopath.Values) error) (err error) {
	for i := 0; i < ls.Len() && err == nil; i++ {
		v := ls.Get(i)
		pushStep(p, protopath.ListIndex(i), v)
		if push != nil {
			err = amendError(err, push(*p))
		}
		if err == nil && fd.Message() != nil {
			err = o.rangeMessage(p, v.Message(), push, pop)
		}
		if pop != nil {
			err = amendError(err, pop(*p))
		}
		popStep(p)
	}

	if err == Break {
		err = nil
	}
	return err
}

func (o Options) rangeMap(p *protopath.Values, fd protoreflect.FieldDescriptor, ms protoreflect.Map, push, pop func(protopath.Values) error) (err error) {
	keyOrder := order.AnyKeyOrder
	if o.Stable {
		keyOrder = order.GenericKeyOrder
	}
	order.RangeEntries(ms, keyOrder, func(k protoreflect.MapKey, v protoreflect.Value) bool {
		pushStep(p, protopath.MapIndex(k), v)
		if push != nil {
			err = amendError(err, push(*p))
		}
		if err == nil && fd.MapValue().Message() != nil {
			err = o.rangeMessage(p, v.Message(), push, pop)
		}
		if pop != nil {
			err = amendError(err, pop(*p))
		}
		popStep(p)
		return err == nil
	})

	if err == Break {
		err = nil
	}
	return err
}

func pushStep(p *protopath.Values, s protopath.Step, v protoreflect.Value) {
	p.Path = append(p.Path, s)
	p.Values = append(p.Values, v)
}

func popStep(p *protopath.Values) {
	p.Path = p.Path[:len(p.Path)-1]
	p.Values = p.Values[:len(p.Values)-1]
}

// amendError amends the previous error with the current error if it is
// considered more serious. The precedence order for errors is:
//
//	nil < Break < Terminate < previous non-nil < current non-nil
func amendError(prev, curr error) error {
	switch {
	case curr == nil:
		return prev
	case curr == Break && prev != nil:
		return prev
	case curr == Terminate && prev != nil && prev != Break:
		return prev
	default:
		return curr
	}
}
