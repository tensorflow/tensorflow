// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protopack enables manual encoding and decoding of protobuf wire data.
//
// This package is intended for use in debugging and/or creation of test data.
// Proper usage of this package requires knowledge of the wire format.
//
// See https://protobuf.dev/programming-guides/encoding.
package protopack

import (
	"fmt"
	"io"
	"math"
	"path"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Number is the field number; aliased from the [protowire] package for convenience.
type Number = protowire.Number

// Number type constants; copied from the [protowire] package for convenience.
const (
	MinValidNumber      Number = protowire.MinValidNumber
	FirstReservedNumber Number = protowire.FirstReservedNumber
	LastReservedNumber  Number = protowire.LastReservedNumber
	MaxValidNumber      Number = protowire.MaxValidNumber
)

// Type is the wire type; aliased from the [protowire] package for convenience.
type Type = protowire.Type

// Wire type constants; copied from the [protowire] package for convenience.
const (
	VarintType     Type = protowire.VarintType
	Fixed32Type    Type = protowire.Fixed32Type
	Fixed64Type    Type = protowire.Fixed64Type
	BytesType      Type = protowire.BytesType
	StartGroupType Type = protowire.StartGroupType
	EndGroupType   Type = protowire.EndGroupType
)

type (
	// Token is any other type (e.g., [Message], [Tag], [Varint], [Float32], etc).
	Token token
	// Message is an ordered sequence of [Token] values, where certain tokens may
	// contain other tokens. It is functionally a concrete syntax tree that
	// losslessly represents any arbitrary wire data (including invalid input).
	Message []Token

	// Tag is a tuple of the field number and the wire type.
	Tag struct {
		Number Number
		Type   Type
	}
	// Bool is a boolean.
	Bool bool
	// Varint is a signed varint using 64-bit two's complement encoding.
	Varint int64
	// Svarint is a signed varint using zig-zag encoding.
	Svarint int64
	// Uvarint is a unsigned varint.
	Uvarint uint64

	// Int32 is a signed 32-bit fixed-width integer.
	Int32 int32
	// Uint32 is an unsigned 32-bit fixed-width integer.
	Uint32 uint32
	// Float32 is a 32-bit fixed-width floating point number.
	Float32 float32

	// Int64 is a signed 64-bit fixed-width integer.
	Int64 int64
	// Uint64 is an unsigned 64-bit fixed-width integer.
	Uint64 uint64
	// Float64 is a 64-bit fixed-width floating point number.
	Float64 float64

	// String is a length-prefixed string.
	String string
	// Bytes is a length-prefixed bytes.
	Bytes []byte
	// LengthPrefix is a length-prefixed message.
	LengthPrefix Message

	// Denormalized is a denormalized varint value, where a varint is encoded
	// using more bytes than is strictly necessary. The number of extra bytes
	// alone is sufficient to losslessly represent the denormalized varint.
	//
	// The value may be one of [Tag], [Bool], [Varint], [Svarint], or [Uvarint],
	// where the varint representation of each token is denormalized.
	//
	// Alternatively, the value may be one of [String], [Bytes], or [LengthPrefix],
	// where the varint representation of the length-prefix is denormalized.
	Denormalized struct {
		Count uint // number of extra bytes
		Value Token
	}

	// Raw are bytes directly appended to output.
	Raw []byte
)

type token interface {
	isToken()
}

func (Message) isToken()      {}
func (Tag) isToken()          {}
func (Bool) isToken()         {}
func (Varint) isToken()       {}
func (Svarint) isToken()      {}
func (Uvarint) isToken()      {}
func (Int32) isToken()        {}
func (Uint32) isToken()       {}
func (Float32) isToken()      {}
func (Int64) isToken()        {}
func (Uint64) isToken()       {}
func (Float64) isToken()      {}
func (String) isToken()       {}
func (Bytes) isToken()        {}
func (LengthPrefix) isToken() {}
func (Denormalized) isToken() {}
func (Raw) isToken()          {}

// Size reports the size in bytes of the marshaled message.
func (m Message) Size() int {
	var n int
	for _, v := range m {
		switch v := v.(type) {
		case Message:
			n += v.Size()
		case Tag:
			n += protowire.SizeTag(v.Number)
		case Bool:
			n += protowire.SizeVarint(protowire.EncodeBool(false))
		case Varint:
			n += protowire.SizeVarint(uint64(v))
		case Svarint:
			n += protowire.SizeVarint(protowire.EncodeZigZag(int64(v)))
		case Uvarint:
			n += protowire.SizeVarint(uint64(v))
		case Int32, Uint32, Float32:
			n += protowire.SizeFixed32()
		case Int64, Uint64, Float64:
			n += protowire.SizeFixed64()
		case String:
			n += protowire.SizeBytes(len(v))
		case Bytes:
			n += protowire.SizeBytes(len(v))
		case LengthPrefix:
			n += protowire.SizeBytes(Message(v).Size())
		case Denormalized:
			n += int(v.Count) + Message{v.Value}.Size()
		case Raw:
			n += len(v)
		default:
			panic(fmt.Sprintf("unknown type: %T", v))
		}
	}
	return n
}

// Marshal encodes a syntax tree into the protobuf wire format.
//
// Example message definition:
//
//	message MyMessage {
//		string field1 = 1;
//		int64 field2 = 2;
//		repeated float32 field3 = 3;
//	}
//
// Example encoded message:
//
//	b := Message{
//		Tag{1, BytesType}, String("Hello, world!"),
//		Tag{2, VarintType}, Varint(-10),
//		Tag{3, BytesType}, LengthPrefix{
//			Float32(1.1), Float32(2.2), Float32(3.3),
//		},
//	}.Marshal()
//
// Resulting wire data:
//
//	0x0000  0a 0d 48 65 6c 6c 6f 2c  20 77 6f 72 6c 64 21 10  |..Hello, world!.|
//	0x0010  f6 ff ff ff ff ff ff ff  ff 01 1a 0c cd cc 8c 3f  |...............?|
//	0x0020  cd cc 0c 40 33 33 53 40                           |...@33S@|
func (m Message) Marshal() []byte {
	var out []byte
	for _, v := range m {
		switch v := v.(type) {
		case Message:
			out = append(out, v.Marshal()...)
		case Tag:
			out = protowire.AppendTag(out, v.Number, v.Type)
		case Bool:
			out = protowire.AppendVarint(out, protowire.EncodeBool(bool(v)))
		case Varint:
			out = protowire.AppendVarint(out, uint64(v))
		case Svarint:
			out = protowire.AppendVarint(out, protowire.EncodeZigZag(int64(v)))
		case Uvarint:
			out = protowire.AppendVarint(out, uint64(v))
		case Int32:
			out = protowire.AppendFixed32(out, uint32(v))
		case Uint32:
			out = protowire.AppendFixed32(out, uint32(v))
		case Float32:
			out = protowire.AppendFixed32(out, math.Float32bits(float32(v)))
		case Int64:
			out = protowire.AppendFixed64(out, uint64(v))
		case Uint64:
			out = protowire.AppendFixed64(out, uint64(v))
		case Float64:
			out = protowire.AppendFixed64(out, math.Float64bits(float64(v)))
		case String:
			out = protowire.AppendBytes(out, []byte(v))
		case Bytes:
			out = protowire.AppendBytes(out, []byte(v))
		case LengthPrefix:
			out = protowire.AppendBytes(out, Message(v).Marshal())
		case Denormalized:
			b := Message{v.Value}.Marshal()
			_, n := protowire.ConsumeVarint(b)
			out = append(out, b[:n]...)
			for i := uint(0); i < v.Count; i++ {
				out[len(out)-1] |= 0x80 // set continuation bit on previous
				out = append(out, 0)
			}
			out = append(out, b[n:]...)
		case Raw:
			return append(out, v...)
		default:
			panic(fmt.Sprintf("unknown type: %T", v))
		}
	}
	return out
}

// Unmarshal parses the input protobuf wire data as a syntax tree.
// Any parsing error results in the remainder of the input being
// concatenated to the message as a [Raw] type.
//
// Each tag (a tuple of the field number and wire type) encountered is
// inserted into the syntax tree as a [Tag].
//
// The contents of each wire type is mapped to the following Go types:
//
//   - [VarintType] ⇒ [Uvarint]
//   - [Fixed32Type] ⇒ [Uint32]
//   - [Fixed64Type] ⇒ [Uint64]
//   - [BytesType] ⇒ [Bytes]
//   - [StartGroupType] and [StartGroupType] ⇒ [Message]
//
// Since the wire format is not self-describing, this function cannot parse
// sub-messages and will leave them as the [Bytes] type. Further manual parsing
// can be performed as such:
//
//	var m, m1, m2 Message
//	m.Unmarshal(b)
//	m1.Unmarshal(m[3].(Bytes))
//	m[3] = LengthPrefix(m1)
//	m2.Unmarshal(m[3].(LengthPrefix)[1].(Bytes))
//	m[3].(LengthPrefix)[1] = LengthPrefix(m2)
//
// Unmarshal is useful for debugging the protobuf wire format.
func (m *Message) Unmarshal(in []byte) {
	m.unmarshal(in, nil, false)
}

// UnmarshalDescriptor parses the input protobuf wire data as a syntax tree
// using the provided message descriptor for more accurate parsing of fields.
// It operates like [Message.Unmarshal], but may use a wider range of Go types to
// represent the wire data.
//
// The contents of each wire type is mapped to one of the following Go types:
//
//   - [VarintType] ⇒ [Bool], [Varint], [Svarint], [Uvarint]
//   - [Fixed32Type] ⇒ [Int32], [Uint32], [Float32]
//   - [Fixed64Type] ⇒ [Uint32], [Uint64], [Float64]
//   - [BytesType] ⇒ [String], [Bytes], [LengthPrefix]
//   - [StartGroupType] and [StartGroupType] ⇒ [Message]
//
// If the field is unknown, it uses the same mapping as [Message.Unmarshal].
// Known sub-messages are parsed as a Message and packed repeated fields are
// parsed as a [LengthPrefix].
func (m *Message) UnmarshalDescriptor(in []byte, desc protoreflect.MessageDescriptor) {
	m.unmarshal(in, desc, false)
}

// UnmarshalAbductive is like [Message.UnmarshalDescriptor], but infers abductively
// whether any unknown bytes values is a message based on whether it is
// a syntactically well-formed message.
//
// Note that the protobuf wire format is not fully self-describing,
// so abductive inference may attempt to expand a bytes value as a message
// that is not actually a message. It is a best-effort guess.
func (m *Message) UnmarshalAbductive(in []byte, desc protoreflect.MessageDescriptor) {
	m.unmarshal(in, desc, true)
}

func (m *Message) unmarshal(in []byte, desc protoreflect.MessageDescriptor, inferMessage bool) {
	p := parser{in: in, out: *m}
	p.parseMessage(desc, false, inferMessage)
	*m = p.out
}

type parser struct {
	in  []byte
	out []Token

	invalid bool
}

func (p *parser) parseMessage(msgDesc protoreflect.MessageDescriptor, group, inferMessage bool) {
	for len(p.in) > 0 {
		v, n := protowire.ConsumeVarint(p.in)
		num, typ := protowire.DecodeTag(v)
		if n < 0 || num <= 0 || v > math.MaxUint32 {
			p.out, p.in = append(p.out, Raw(p.in)), nil
			p.invalid = true
			return
		}
		if typ == EndGroupType && group {
			return // if inside a group, then stop
		}
		p.out, p.in = append(p.out, Tag{num, typ}), p.in[n:]
		if m := n - protowire.SizeVarint(v); m > 0 {
			p.out[len(p.out)-1] = Denormalized{uint(m), p.out[len(p.out)-1]}
		}

		// If descriptor is available, use it for more accurate parsing.
		var isPacked bool
		var kind protoreflect.Kind
		var subDesc protoreflect.MessageDescriptor
		if msgDesc != nil && !msgDesc.IsPlaceholder() {
			if fieldDesc := msgDesc.Fields().ByNumber(num); fieldDesc != nil {
				isPacked = fieldDesc.IsPacked()
				kind = fieldDesc.Kind()
				switch kind {
				case protoreflect.MessageKind, protoreflect.GroupKind:
					subDesc = fieldDesc.Message()
					if subDesc == nil || subDesc.IsPlaceholder() {
						kind = 0
					}
				}
			}
		}

		switch typ {
		case VarintType:
			p.parseVarint(kind)
		case Fixed32Type:
			p.parseFixed32(kind)
		case Fixed64Type:
			p.parseFixed64(kind)
		case BytesType:
			p.parseBytes(isPacked, kind, subDesc, inferMessage)
		case StartGroupType:
			p.parseGroup(num, subDesc, inferMessage)
		case EndGroupType:
			// Handled by p.parseGroup.
		default:
			p.out, p.in = append(p.out, Raw(p.in)), nil
			p.invalid = true
		}
	}
}

func (p *parser) parseVarint(kind protoreflect.Kind) {
	v, n := protowire.ConsumeVarint(p.in)
	if n < 0 {
		p.out, p.in = append(p.out, Raw(p.in)), nil
		p.invalid = true
		return
	}
	switch kind {
	case protoreflect.BoolKind:
		switch v {
		case 0:
			p.out, p.in = append(p.out, Bool(false)), p.in[n:]
		case 1:
			p.out, p.in = append(p.out, Bool(true)), p.in[n:]
		default:
			p.out, p.in = append(p.out, Uvarint(v)), p.in[n:]
		}
	case protoreflect.Int32Kind, protoreflect.Int64Kind:
		p.out, p.in = append(p.out, Varint(v)), p.in[n:]
	case protoreflect.Sint32Kind, protoreflect.Sint64Kind:
		p.out, p.in = append(p.out, Svarint(protowire.DecodeZigZag(v))), p.in[n:]
	default:
		p.out, p.in = append(p.out, Uvarint(v)), p.in[n:]
	}
	if m := n - protowire.SizeVarint(v); m > 0 {
		p.out[len(p.out)-1] = Denormalized{uint(m), p.out[len(p.out)-1]}
	}
}

func (p *parser) parseFixed32(kind protoreflect.Kind) {
	v, n := protowire.ConsumeFixed32(p.in)
	if n < 0 {
		p.out, p.in = append(p.out, Raw(p.in)), nil
		p.invalid = true
		return
	}
	switch kind {
	case protoreflect.FloatKind:
		p.out, p.in = append(p.out, Float32(math.Float32frombits(v))), p.in[n:]
	case protoreflect.Sfixed32Kind:
		p.out, p.in = append(p.out, Int32(v)), p.in[n:]
	default:
		p.out, p.in = append(p.out, Uint32(v)), p.in[n:]
	}
}

func (p *parser) parseFixed64(kind protoreflect.Kind) {
	v, n := protowire.ConsumeFixed64(p.in)
	if n < 0 {
		p.out, p.in = append(p.out, Raw(p.in)), nil
		p.invalid = true
		return
	}
	switch kind {
	case protoreflect.DoubleKind:
		p.out, p.in = append(p.out, Float64(math.Float64frombits(v))), p.in[n:]
	case protoreflect.Sfixed64Kind:
		p.out, p.in = append(p.out, Int64(v)), p.in[n:]
	default:
		p.out, p.in = append(p.out, Uint64(v)), p.in[n:]
	}
}

func (p *parser) parseBytes(isPacked bool, kind protoreflect.Kind, desc protoreflect.MessageDescriptor, inferMessage bool) {
	v, n := protowire.ConsumeVarint(p.in)
	if n < 0 {
		p.out, p.in = append(p.out, Raw(p.in)), nil
		p.invalid = true
		return
	}
	p.out, p.in = append(p.out, Uvarint(v)), p.in[n:]
	if m := n - protowire.SizeVarint(v); m > 0 {
		p.out[len(p.out)-1] = Denormalized{uint(m), p.out[len(p.out)-1]}
	}
	if v > uint64(len(p.in)) {
		p.out, p.in = append(p.out, Raw(p.in)), nil
		p.invalid = true
		return
	}
	p.out = p.out[:len(p.out)-1] // subsequent tokens contain prefix-length

	if isPacked {
		p.parsePacked(int(v), kind)
	} else {
		switch kind {
		case protoreflect.MessageKind:
			p2 := parser{in: p.in[:v]}
			p2.parseMessage(desc, false, inferMessage)
			p.out, p.in = append(p.out, LengthPrefix(p2.out)), p.in[v:]
		case protoreflect.StringKind:
			p.out, p.in = append(p.out, String(p.in[:v])), p.in[v:]
		case protoreflect.BytesKind:
			p.out, p.in = append(p.out, Bytes(p.in[:v])), p.in[v:]
		default:
			if inferMessage {
				// Check whether this is a syntactically valid message.
				p2 := parser{in: p.in[:v]}
				p2.parseMessage(nil, false, inferMessage)
				if !p2.invalid {
					p.out, p.in = append(p.out, LengthPrefix(p2.out)), p.in[v:]
					break
				}
			}
			p.out, p.in = append(p.out, Bytes(p.in[:v])), p.in[v:]
		}
	}
	if m := n - protowire.SizeVarint(v); m > 0 {
		p.out[len(p.out)-1] = Denormalized{uint(m), p.out[len(p.out)-1]}
	}
}

func (p *parser) parsePacked(n int, kind protoreflect.Kind) {
	p2 := parser{in: p.in[:n]}
	for len(p2.in) > 0 {
		switch kind {
		case protoreflect.BoolKind, protoreflect.EnumKind,
			protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Uint32Kind,
			protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Uint64Kind:
			p2.parseVarint(kind)
		case protoreflect.Fixed32Kind, protoreflect.Sfixed32Kind, protoreflect.FloatKind:
			p2.parseFixed32(kind)
		case protoreflect.Fixed64Kind, protoreflect.Sfixed64Kind, protoreflect.DoubleKind:
			p2.parseFixed64(kind)
		default:
			panic(fmt.Sprintf("invalid packed kind: %v", kind))
		}
	}
	p.out, p.in = append(p.out, LengthPrefix(p2.out)), p.in[n:]
}

func (p *parser) parseGroup(startNum protowire.Number, desc protoreflect.MessageDescriptor, inferMessage bool) {
	p2 := parser{in: p.in}
	p2.parseMessage(desc, true, inferMessage)
	if len(p2.out) > 0 {
		p.out = append(p.out, Message(p2.out))
	}
	p.in = p2.in

	// Append the trailing end group.
	v, n := protowire.ConsumeVarint(p.in)
	if endNum, typ := protowire.DecodeTag(v); typ == EndGroupType {
		if startNum != endNum {
			p.invalid = true
		}
		p.out, p.in = append(p.out, Tag{endNum, typ}), p.in[n:]
		if m := n - protowire.SizeVarint(v); m > 0 {
			p.out[len(p.out)-1] = Denormalized{uint(m), p.out[len(p.out)-1]}
		}
	}
}

// Format implements a custom formatter to visualize the syntax tree.
// Using "%#v" formats the Message in Go source code.
func (m Message) Format(s fmt.State, r rune) {
	switch r {
	case 'x':
		io.WriteString(s, fmt.Sprintf("%x", m.Marshal()))
	case 'X':
		io.WriteString(s, fmt.Sprintf("%X", m.Marshal()))
	case 'v':
		switch {
		case s.Flag('#'):
			io.WriteString(s, m.format(true, true))
		case s.Flag('+'):
			io.WriteString(s, m.format(false, true))
		default:
			io.WriteString(s, m.format(false, false))
		}
	default:
		panic("invalid verb: " + string(r))
	}
}

// format formats the message.
// If source is enabled, this emits valid Go source.
// If multi is enabled, the output may span multiple lines.
func (m Message) format(source, multi bool) string {
	var ss []string
	var prefix, nextPrefix string
	for _, v := range m {
		// Ensure certain tokens have preceding or succeeding newlines.
		prefix, nextPrefix = nextPrefix, " "
		if multi {
			switch v := v.(type) {
			case Tag: // only has preceding newline
				prefix = "\n"
			case Denormalized: // only has preceding newline
				if _, ok := v.Value.(Tag); ok {
					prefix = "\n"
				}
			case Message, Raw: // has preceding and succeeding newlines
				prefix, nextPrefix = "\n", "\n"
			}
		}

		s := formatToken(v, source, multi)
		ss = append(ss, prefix+s+",")
	}

	var s string
	if len(ss) > 0 {
		s = strings.TrimSpace(strings.Join(ss, ""))
		if multi {
			s = "\n\t" + strings.Join(strings.Split(s, "\n"), "\n\t") + "\n"
		} else {
			s = strings.TrimSuffix(s, ",")
		}
	}
	s = fmt.Sprintf("%T{%s}", m, s)
	if !source {
		s = trimPackage(s)
	}
	return s
}

// formatToken formats a single token.
func formatToken(t Token, source, multi bool) (s string) {
	switch v := t.(type) {
	case Message:
		s = v.format(source, multi)
	case LengthPrefix:
		s = formatPacked(v, source, multi)
		if s == "" {
			ms := Message(v).format(source, multi)
			s = fmt.Sprintf("%T(%s)", v, ms)
		}
	case Tag:
		s = fmt.Sprintf("%T{%d, %s}", v, v.Number, formatType(v.Type, source))
	case Bool, Varint, Svarint, Uvarint, Int32, Uint32, Float32, Int64, Uint64, Float64:
		if source {
			// Print floats in a way that preserves exact precision.
			if f, _ := v.(Float32); math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				switch {
				case f > 0:
					s = fmt.Sprintf("%T(math.Inf(+1))", v)
				case f < 0:
					s = fmt.Sprintf("%T(math.Inf(-1))", v)
				case math.Float32bits(float32(math.NaN())) == math.Float32bits(float32(f)):
					s = fmt.Sprintf("%T(math.NaN())", v)
				default:
					s = fmt.Sprintf("%T(math.Float32frombits(0x%08x))", v, math.Float32bits(float32(f)))
				}
				break
			}
			if f, _ := v.(Float64); math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				switch {
				case f > 0:
					s = fmt.Sprintf("%T(math.Inf(+1))", v)
				case f < 0:
					s = fmt.Sprintf("%T(math.Inf(-1))", v)
				case math.Float64bits(float64(math.NaN())) == math.Float64bits(float64(f)):
					s = fmt.Sprintf("%T(math.NaN())", v)
				default:
					s = fmt.Sprintf("%T(math.Float64frombits(0x%016x))", v, math.Float64bits(float64(f)))
				}
				break
			}
		}
		s = fmt.Sprintf("%T(%v)", v, v)
	case String, Bytes, Raw:
		s = fmt.Sprintf("%s", v)
		s = fmt.Sprintf("%T(%s)", v, formatString(s))
	case Denormalized:
		s = fmt.Sprintf("%T{+%d, %v}", v, v.Count, formatToken(v.Value, source, multi))
	default:
		panic(fmt.Sprintf("unknown type: %T", v))
	}
	if !source {
		s = trimPackage(s)
	}
	return s
}

// formatPacked returns a non-empty string if LengthPrefix looks like a packed
// repeated field of primitives.
func formatPacked(v LengthPrefix, source, multi bool) string {
	var ss []string
	for _, v := range v {
		switch v.(type) {
		case Bool, Varint, Svarint, Uvarint, Int32, Uint32, Float32, Int64, Uint64, Float64, Denormalized, Raw:
			if v, ok := v.(Denormalized); ok {
				switch v.Value.(type) {
				case Bool, Varint, Svarint, Uvarint:
				default:
					return ""
				}
			}
			ss = append(ss, formatToken(v, source, multi))
		default:
			return ""
		}
	}
	s := fmt.Sprintf("%T{%s}", v, strings.Join(ss, ", "))
	if !source {
		s = trimPackage(s)
	}
	return s
}

// formatType returns the name for Type.
func formatType(t Type, source bool) (s string) {
	switch t {
	case VarintType:
		s = pkg + ".VarintType"
	case Fixed32Type:
		s = pkg + ".Fixed32Type"
	case Fixed64Type:
		s = pkg + ".Fixed64Type"
	case BytesType:
		s = pkg + ".BytesType"
	case StartGroupType:
		s = pkg + ".StartGroupType"
	case EndGroupType:
		s = pkg + ".EndGroupType"
	default:
		s = fmt.Sprintf("Type(%d)", t)
	}
	if !source {
		s = strings.TrimSuffix(trimPackage(s), "Type")
	}
	return s
}

// formatString returns a quoted string for s.
func formatString(s string) string {
	// Use quoted string if it the same length as a raw string literal.
	// Otherwise, attempt to use the raw string form.
	qs := strconv.Quote(s)
	if len(qs) == 1+len(s)+1 {
		return qs
	}

	// Disallow newlines to ensure output is a single line.
	// Disallow non-printable runes for readability purposes.
	rawInvalid := func(r rune) bool {
		return r == '`' || r == '\n' || r == utf8.RuneError || !unicode.IsPrint(r)
	}
	if strings.IndexFunc(s, rawInvalid) < 0 {
		return "`" + s + "`"
	}
	return qs
}

var pkg = path.Base(reflect.TypeOf(Tag{}).PkgPath())

func trimPackage(s string) string {
	return strings.TrimPrefix(strings.TrimPrefix(s, pkg), ".")
}
