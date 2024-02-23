// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prototext

import (
	"fmt"
	"unicode/utf8"

	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/encoding/text"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/internal/set"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// Unmarshal reads the given []byte into the given [proto.Message].
// The provided message must be mutable (e.g., a non-nil pointer to a message).
func Unmarshal(b []byte, m proto.Message) error {
	return UnmarshalOptions{}.Unmarshal(b, m)
}

// UnmarshalOptions is a configurable textproto format unmarshaler.
type UnmarshalOptions struct {
	pragma.NoUnkeyedLiterals

	// AllowPartial accepts input for messages that will result in missing
	// required fields. If AllowPartial is false (the default), Unmarshal will
	// return error if there are any missing required fields.
	AllowPartial bool

	// DiscardUnknown specifies whether to ignore unknown fields when parsing.
	// An unknown field is any field whose field name or field number does not
	// resolve to any known or extension field in the message.
	// By default, unmarshal rejects unknown fields as an error.
	DiscardUnknown bool

	// Resolver is used for looking up types when unmarshaling
	// google.protobuf.Any messages or extension fields.
	// If nil, this defaults to using protoregistry.GlobalTypes.
	Resolver interface {
		protoregistry.MessageTypeResolver
		protoregistry.ExtensionTypeResolver
	}
}

// Unmarshal reads the given []byte and populates the given [proto.Message]
// using options in the UnmarshalOptions object.
// The provided message must be mutable (e.g., a non-nil pointer to a message).
func (o UnmarshalOptions) Unmarshal(b []byte, m proto.Message) error {
	return o.unmarshal(b, m)
}

// unmarshal is a centralized function that all unmarshal operations go through.
// For profiling purposes, avoid changing the name of this function or
// introducing other code paths for unmarshal that do not go through this.
func (o UnmarshalOptions) unmarshal(b []byte, m proto.Message) error {
	proto.Reset(m)

	if o.Resolver == nil {
		o.Resolver = protoregistry.GlobalTypes
	}

	dec := decoder{text.NewDecoder(b), o}
	if err := dec.unmarshalMessage(m.ProtoReflect(), false); err != nil {
		return err
	}
	if o.AllowPartial {
		return nil
	}
	return proto.CheckInitialized(m)
}

type decoder struct {
	*text.Decoder
	opts UnmarshalOptions
}

// newError returns an error object with position info.
func (d decoder) newError(pos int, f string, x ...interface{}) error {
	line, column := d.Position(pos)
	head := fmt.Sprintf("(line %d:%d): ", line, column)
	return errors.New(head+f, x...)
}

// unexpectedTokenError returns a syntax error for the given unexpected token.
func (d decoder) unexpectedTokenError(tok text.Token) error {
	return d.syntaxError(tok.Pos(), "unexpected token: %s", tok.RawString())
}

// syntaxError returns a syntax error for given position.
func (d decoder) syntaxError(pos int, f string, x ...interface{}) error {
	line, column := d.Position(pos)
	head := fmt.Sprintf("syntax error (line %d:%d): ", line, column)
	return errors.New(head+f, x...)
}

// unmarshalMessage unmarshals into the given protoreflect.Message.
func (d decoder) unmarshalMessage(m protoreflect.Message, checkDelims bool) error {
	messageDesc := m.Descriptor()
	if !flags.ProtoLegacy && messageset.IsMessageSet(messageDesc) {
		return errors.New("no support for proto1 MessageSets")
	}

	if messageDesc.FullName() == genid.Any_message_fullname {
		return d.unmarshalAny(m, checkDelims)
	}

	if checkDelims {
		tok, err := d.Read()
		if err != nil {
			return err
		}

		if tok.Kind() != text.MessageOpen {
			return d.unexpectedTokenError(tok)
		}
	}

	var seenNums set.Ints
	var seenOneofs set.Ints
	fieldDescs := messageDesc.Fields()

	for {
		// Read field name.
		tok, err := d.Read()
		if err != nil {
			return err
		}
		switch typ := tok.Kind(); typ {
		case text.Name:
			// Continue below.
		case text.EOF:
			if checkDelims {
				return text.ErrUnexpectedEOF
			}
			return nil
		default:
			if checkDelims && typ == text.MessageClose {
				return nil
			}
			return d.unexpectedTokenError(tok)
		}

		// Resolve the field descriptor.
		var name protoreflect.Name
		var fd protoreflect.FieldDescriptor
		var xt protoreflect.ExtensionType
		var xtErr error
		var isFieldNumberName bool

		switch tok.NameKind() {
		case text.IdentName:
			name = protoreflect.Name(tok.IdentName())
			fd = fieldDescs.ByTextName(string(name))

		case text.TypeName:
			// Handle extensions only. This code path is not for Any.
			xt, xtErr = d.opts.Resolver.FindExtensionByName(protoreflect.FullName(tok.TypeName()))

		case text.FieldNumber:
			isFieldNumberName = true
			num := protoreflect.FieldNumber(tok.FieldNumber())
			if !num.IsValid() {
				return d.newError(tok.Pos(), "invalid field number: %d", num)
			}
			fd = fieldDescs.ByNumber(num)
			if fd == nil {
				xt, xtErr = d.opts.Resolver.FindExtensionByNumber(messageDesc.FullName(), num)
			}
		}

		if xt != nil {
			fd = xt.TypeDescriptor()
			if !messageDesc.ExtensionRanges().Has(fd.Number()) || fd.ContainingMessage().FullName() != messageDesc.FullName() {
				return d.newError(tok.Pos(), "message %v cannot be extended by %v", messageDesc.FullName(), fd.FullName())
			}
		} else if xtErr != nil && xtErr != protoregistry.NotFound {
			return d.newError(tok.Pos(), "unable to resolve [%s]: %v", tok.RawString(), xtErr)
		}
		if flags.ProtoLegacy {
			if fd != nil && fd.IsWeak() && fd.Message().IsPlaceholder() {
				fd = nil // reset since the weak reference is not linked in
			}
		}

		// Handle unknown fields.
		if fd == nil {
			if d.opts.DiscardUnknown || messageDesc.ReservedNames().Has(name) {
				d.skipValue()
				continue
			}
			return d.newError(tok.Pos(), "unknown field: %v", tok.RawString())
		}

		// Handle fields identified by field number.
		if isFieldNumberName {
			// TODO: Add an option to permit parsing field numbers.
			//
			// This requires careful thought as the MarshalOptions.EmitUnknown
			// option allows formatting unknown fields as the field number and the
			// best-effort textual representation of the field value.  In that case,
			// it may not be possible to unmarshal the value from a parser that does
			// have information about the unknown field.
			return d.newError(tok.Pos(), "cannot specify field by number: %v", tok.RawString())
		}

		switch {
		case fd.IsList():
			kind := fd.Kind()
			if kind != protoreflect.MessageKind && kind != protoreflect.GroupKind && !tok.HasSeparator() {
				return d.syntaxError(tok.Pos(), "missing field separator :")
			}

			list := m.Mutable(fd).List()
			if err := d.unmarshalList(fd, list); err != nil {
				return err
			}

		case fd.IsMap():
			mmap := m.Mutable(fd).Map()
			if err := d.unmarshalMap(fd, mmap); err != nil {
				return err
			}

		default:
			kind := fd.Kind()
			if kind != protoreflect.MessageKind && kind != protoreflect.GroupKind && !tok.HasSeparator() {
				return d.syntaxError(tok.Pos(), "missing field separator :")
			}

			// If field is a oneof, check if it has already been set.
			if od := fd.ContainingOneof(); od != nil {
				idx := uint64(od.Index())
				if seenOneofs.Has(idx) {
					return d.newError(tok.Pos(), "error parsing %q, oneof %v is already set", tok.RawString(), od.FullName())
				}
				seenOneofs.Set(idx)
			}

			num := uint64(fd.Number())
			if seenNums.Has(num) {
				return d.newError(tok.Pos(), "non-repeated field %q is repeated", tok.RawString())
			}

			if err := d.unmarshalSingular(fd, m); err != nil {
				return err
			}
			seenNums.Set(num)
		}
	}

	return nil
}

// unmarshalSingular unmarshals a non-repeated field value specified by the
// given FieldDescriptor.
func (d decoder) unmarshalSingular(fd protoreflect.FieldDescriptor, m protoreflect.Message) error {
	var val protoreflect.Value
	var err error
	switch fd.Kind() {
	case protoreflect.MessageKind, protoreflect.GroupKind:
		val = m.NewField(fd)
		err = d.unmarshalMessage(val.Message(), true)
	default:
		val, err = d.unmarshalScalar(fd)
	}
	if err == nil {
		m.Set(fd, val)
	}
	return err
}

// unmarshalScalar unmarshals a scalar/enum protoreflect.Value specified by the
// given FieldDescriptor.
func (d decoder) unmarshalScalar(fd protoreflect.FieldDescriptor) (protoreflect.Value, error) {
	tok, err := d.Read()
	if err != nil {
		return protoreflect.Value{}, err
	}

	if tok.Kind() != text.Scalar {
		return protoreflect.Value{}, d.unexpectedTokenError(tok)
	}

	kind := fd.Kind()
	switch kind {
	case protoreflect.BoolKind:
		if b, ok := tok.Bool(); ok {
			return protoreflect.ValueOfBool(b), nil
		}

	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		if n, ok := tok.Int32(); ok {
			return protoreflect.ValueOfInt32(n), nil
		}

	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		if n, ok := tok.Int64(); ok {
			return protoreflect.ValueOfInt64(n), nil
		}

	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		if n, ok := tok.Uint32(); ok {
			return protoreflect.ValueOfUint32(n), nil
		}

	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		if n, ok := tok.Uint64(); ok {
			return protoreflect.ValueOfUint64(n), nil
		}

	case protoreflect.FloatKind:
		if n, ok := tok.Float32(); ok {
			return protoreflect.ValueOfFloat32(n), nil
		}

	case protoreflect.DoubleKind:
		if n, ok := tok.Float64(); ok {
			return protoreflect.ValueOfFloat64(n), nil
		}

	case protoreflect.StringKind:
		if s, ok := tok.String(); ok {
			if strs.EnforceUTF8(fd) && !utf8.ValidString(s) {
				return protoreflect.Value{}, d.newError(tok.Pos(), "contains invalid UTF-8")
			}
			return protoreflect.ValueOfString(s), nil
		}

	case protoreflect.BytesKind:
		if b, ok := tok.String(); ok {
			return protoreflect.ValueOfBytes([]byte(b)), nil
		}

	case protoreflect.EnumKind:
		if lit, ok := tok.Enum(); ok {
			// Lookup EnumNumber based on name.
			if enumVal := fd.Enum().Values().ByName(protoreflect.Name(lit)); enumVal != nil {
				return protoreflect.ValueOfEnum(enumVal.Number()), nil
			}
		}
		if num, ok := tok.Int32(); ok {
			return protoreflect.ValueOfEnum(protoreflect.EnumNumber(num)), nil
		}

	default:
		panic(fmt.Sprintf("invalid scalar kind %v", kind))
	}

	return protoreflect.Value{}, d.newError(tok.Pos(), "invalid value for %v type: %v", kind, tok.RawString())
}

// unmarshalList unmarshals into given protoreflect.List. A list value can
// either be in [] syntax or simply just a single scalar/message value.
func (d decoder) unmarshalList(fd protoreflect.FieldDescriptor, list protoreflect.List) error {
	tok, err := d.Peek()
	if err != nil {
		return err
	}

	switch fd.Kind() {
	case protoreflect.MessageKind, protoreflect.GroupKind:
		switch tok.Kind() {
		case text.ListOpen:
			d.Read()
			for {
				tok, err := d.Peek()
				if err != nil {
					return err
				}

				switch tok.Kind() {
				case text.ListClose:
					d.Read()
					return nil
				case text.MessageOpen:
					pval := list.NewElement()
					if err := d.unmarshalMessage(pval.Message(), true); err != nil {
						return err
					}
					list.Append(pval)
				default:
					return d.unexpectedTokenError(tok)
				}
			}

		case text.MessageOpen:
			pval := list.NewElement()
			if err := d.unmarshalMessage(pval.Message(), true); err != nil {
				return err
			}
			list.Append(pval)
			return nil
		}

	default:
		switch tok.Kind() {
		case text.ListOpen:
			d.Read()
			for {
				tok, err := d.Peek()
				if err != nil {
					return err
				}

				switch tok.Kind() {
				case text.ListClose:
					d.Read()
					return nil
				case text.Scalar:
					pval, err := d.unmarshalScalar(fd)
					if err != nil {
						return err
					}
					list.Append(pval)
				default:
					return d.unexpectedTokenError(tok)
				}
			}

		case text.Scalar:
			pval, err := d.unmarshalScalar(fd)
			if err != nil {
				return err
			}
			list.Append(pval)
			return nil
		}
	}

	return d.unexpectedTokenError(tok)
}

// unmarshalMap unmarshals into given protoreflect.Map. A map value is a
// textproto message containing {key: <kvalue>, value: <mvalue>}.
func (d decoder) unmarshalMap(fd protoreflect.FieldDescriptor, mmap protoreflect.Map) error {
	// Determine ahead whether map entry is a scalar type or a message type in
	// order to call the appropriate unmarshalMapValue func inside
	// unmarshalMapEntry.
	var unmarshalMapValue func() (protoreflect.Value, error)
	switch fd.MapValue().Kind() {
	case protoreflect.MessageKind, protoreflect.GroupKind:
		unmarshalMapValue = func() (protoreflect.Value, error) {
			pval := mmap.NewValue()
			if err := d.unmarshalMessage(pval.Message(), true); err != nil {
				return protoreflect.Value{}, err
			}
			return pval, nil
		}
	default:
		unmarshalMapValue = func() (protoreflect.Value, error) {
			return d.unmarshalScalar(fd.MapValue())
		}
	}

	tok, err := d.Read()
	if err != nil {
		return err
	}
	switch tok.Kind() {
	case text.MessageOpen:
		return d.unmarshalMapEntry(fd, mmap, unmarshalMapValue)

	case text.ListOpen:
		for {
			tok, err := d.Read()
			if err != nil {
				return err
			}
			switch tok.Kind() {
			case text.ListClose:
				return nil
			case text.MessageOpen:
				if err := d.unmarshalMapEntry(fd, mmap, unmarshalMapValue); err != nil {
					return err
				}
			default:
				return d.unexpectedTokenError(tok)
			}
		}

	default:
		return d.unexpectedTokenError(tok)
	}
}

// unmarshalMap unmarshals into given protoreflect.Map. A map value is a
// textproto message containing {key: <kvalue>, value: <mvalue>}.
func (d decoder) unmarshalMapEntry(fd protoreflect.FieldDescriptor, mmap protoreflect.Map, unmarshalMapValue func() (protoreflect.Value, error)) error {
	var key protoreflect.MapKey
	var pval protoreflect.Value
Loop:
	for {
		// Read field name.
		tok, err := d.Read()
		if err != nil {
			return err
		}
		switch tok.Kind() {
		case text.Name:
			if tok.NameKind() != text.IdentName {
				if !d.opts.DiscardUnknown {
					return d.newError(tok.Pos(), "unknown map entry field %q", tok.RawString())
				}
				d.skipValue()
				continue Loop
			}
			// Continue below.
		case text.MessageClose:
			break Loop
		default:
			return d.unexpectedTokenError(tok)
		}

		switch name := protoreflect.Name(tok.IdentName()); name {
		case genid.MapEntry_Key_field_name:
			if !tok.HasSeparator() {
				return d.syntaxError(tok.Pos(), "missing field separator :")
			}
			if key.IsValid() {
				return d.newError(tok.Pos(), "map entry %q cannot be repeated", name)
			}
			val, err := d.unmarshalScalar(fd.MapKey())
			if err != nil {
				return err
			}
			key = val.MapKey()

		case genid.MapEntry_Value_field_name:
			if kind := fd.MapValue().Kind(); (kind != protoreflect.MessageKind) && (kind != protoreflect.GroupKind) {
				if !tok.HasSeparator() {
					return d.syntaxError(tok.Pos(), "missing field separator :")
				}
			}
			if pval.IsValid() {
				return d.newError(tok.Pos(), "map entry %q cannot be repeated", name)
			}
			pval, err = unmarshalMapValue()
			if err != nil {
				return err
			}

		default:
			if !d.opts.DiscardUnknown {
				return d.newError(tok.Pos(), "unknown map entry field %q", name)
			}
			d.skipValue()
		}
	}

	if !key.IsValid() {
		key = fd.MapKey().Default().MapKey()
	}
	if !pval.IsValid() {
		switch fd.MapValue().Kind() {
		case protoreflect.MessageKind, protoreflect.GroupKind:
			// If value field is not set for message/group types, construct an
			// empty one as default.
			pval = mmap.NewValue()
		default:
			pval = fd.MapValue().Default()
		}
	}
	mmap.Set(key, pval)
	return nil
}

// unmarshalAny unmarshals an Any textproto. It can either be in expanded form
// or non-expanded form.
func (d decoder) unmarshalAny(m protoreflect.Message, checkDelims bool) error {
	var typeURL string
	var bValue []byte
	var seenTypeUrl bool
	var seenValue bool
	var isExpanded bool

	if checkDelims {
		tok, err := d.Read()
		if err != nil {
			return err
		}

		if tok.Kind() != text.MessageOpen {
			return d.unexpectedTokenError(tok)
		}
	}

Loop:
	for {
		// Read field name. Can only have 3 possible field names, i.e. type_url,
		// value and type URL name inside [].
		tok, err := d.Read()
		if err != nil {
			return err
		}
		if typ := tok.Kind(); typ != text.Name {
			if checkDelims {
				if typ == text.MessageClose {
					break Loop
				}
			} else if typ == text.EOF {
				break Loop
			}
			return d.unexpectedTokenError(tok)
		}

		switch tok.NameKind() {
		case text.IdentName:
			// Both type_url and value fields require field separator :.
			if !tok.HasSeparator() {
				return d.syntaxError(tok.Pos(), "missing field separator :")
			}

			switch name := protoreflect.Name(tok.IdentName()); name {
			case genid.Any_TypeUrl_field_name:
				if seenTypeUrl {
					return d.newError(tok.Pos(), "duplicate %v field", genid.Any_TypeUrl_field_fullname)
				}
				if isExpanded {
					return d.newError(tok.Pos(), "conflict with [%s] field", typeURL)
				}
				tok, err := d.Read()
				if err != nil {
					return err
				}
				var ok bool
				typeURL, ok = tok.String()
				if !ok {
					return d.newError(tok.Pos(), "invalid %v field value: %v", genid.Any_TypeUrl_field_fullname, tok.RawString())
				}
				seenTypeUrl = true

			case genid.Any_Value_field_name:
				if seenValue {
					return d.newError(tok.Pos(), "duplicate %v field", genid.Any_Value_field_fullname)
				}
				if isExpanded {
					return d.newError(tok.Pos(), "conflict with [%s] field", typeURL)
				}
				tok, err := d.Read()
				if err != nil {
					return err
				}
				s, ok := tok.String()
				if !ok {
					return d.newError(tok.Pos(), "invalid %v field value: %v", genid.Any_Value_field_fullname, tok.RawString())
				}
				bValue = []byte(s)
				seenValue = true

			default:
				if !d.opts.DiscardUnknown {
					return d.newError(tok.Pos(), "invalid field name %q in %v message", tok.RawString(), genid.Any_message_fullname)
				}
			}

		case text.TypeName:
			if isExpanded {
				return d.newError(tok.Pos(), "cannot have more than one type")
			}
			if seenTypeUrl {
				return d.newError(tok.Pos(), "conflict with type_url field")
			}
			typeURL = tok.TypeName()
			var err error
			bValue, err = d.unmarshalExpandedAny(typeURL, tok.Pos())
			if err != nil {
				return err
			}
			isExpanded = true

		default:
			if !d.opts.DiscardUnknown {
				return d.newError(tok.Pos(), "invalid field name %q in %v message", tok.RawString(), genid.Any_message_fullname)
			}
		}
	}

	fds := m.Descriptor().Fields()
	if len(typeURL) > 0 {
		m.Set(fds.ByNumber(genid.Any_TypeUrl_field_number), protoreflect.ValueOfString(typeURL))
	}
	if len(bValue) > 0 {
		m.Set(fds.ByNumber(genid.Any_Value_field_number), protoreflect.ValueOfBytes(bValue))
	}
	return nil
}

func (d decoder) unmarshalExpandedAny(typeURL string, pos int) ([]byte, error) {
	mt, err := d.opts.Resolver.FindMessageByURL(typeURL)
	if err != nil {
		return nil, d.newError(pos, "unable to resolve message [%v]: %v", typeURL, err)
	}
	// Create new message for the embedded message type and unmarshal the value
	// field into it.
	m := mt.New()
	if err := d.unmarshalMessage(m, true); err != nil {
		return nil, err
	}
	// Serialize the embedded message and return the resulting bytes.
	b, err := proto.MarshalOptions{
		AllowPartial:  true, // Never check required fields inside an Any.
		Deterministic: true,
	}.Marshal(m.Interface())
	if err != nil {
		return nil, d.newError(pos, "error in marshaling message into Any.value: %v", err)
	}
	return b, nil
}

// skipValue makes the decoder parse a field value in order to advance the read
// to the next field. It relies on Read returning an error if the types are not
// in valid sequence.
func (d decoder) skipValue() error {
	tok, err := d.Read()
	if err != nil {
		return err
	}
	// Only need to continue reading for messages and lists.
	switch tok.Kind() {
	case text.MessageOpen:
		return d.skipMessageValue()

	case text.ListOpen:
		for {
			tok, err := d.Read()
			if err != nil {
				return err
			}
			switch tok.Kind() {
			case text.ListClose:
				return nil
			case text.MessageOpen:
				if err := d.skipMessageValue(); err != nil {
					return err
				}
			default:
				// Skip items. This will not validate whether skipped values are
				// of the same type or not, same behavior as C++
				// TextFormat::Parser::AllowUnknownField(true) version 3.8.0.
			}
		}
	}
	return nil
}

// skipMessageValue makes the decoder parse and skip over all fields in a
// message. It assumes that the previous read type is MessageOpen.
func (d decoder) skipMessageValue() error {
	for {
		tok, err := d.Read()
		if err != nil {
			return err
		}
		switch tok.Kind() {
		case text.MessageClose:
			return nil
		case text.Name:
			if err := d.skipValue(); err != nil {
				return err
			}
		}
	}
}
