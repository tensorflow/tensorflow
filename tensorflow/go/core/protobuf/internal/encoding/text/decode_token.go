// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/flags"
)

// Kind represents a token kind expressible in the textproto format.
type Kind uint8

// Kind values.
const (
	Invalid Kind = iota
	EOF
	Name   // Name indicates the field name.
	Scalar // Scalar are scalar values, e.g. "string", 47, ENUM_LITERAL, true.
	MessageOpen
	MessageClose
	ListOpen
	ListClose

	// comma and semi-colon are only for parsing in between values and should not be exposed.
	comma
	semicolon

	// bof indicates beginning of file, which is the default token
	// kind at the beginning of parsing.
	bof = Invalid
)

func (t Kind) String() string {
	switch t {
	case Invalid:
		return "<invalid>"
	case EOF:
		return "eof"
	case Scalar:
		return "scalar"
	case Name:
		return "name"
	case MessageOpen:
		return "{"
	case MessageClose:
		return "}"
	case ListOpen:
		return "["
	case ListClose:
		return "]"
	case comma:
		return ","
	case semicolon:
		return ";"
	default:
		return fmt.Sprintf("<invalid:%v>", uint8(t))
	}
}

// NameKind represents different types of field names.
type NameKind uint8

// NameKind values.
const (
	IdentName NameKind = iota + 1
	TypeName
	FieldNumber
)

func (t NameKind) String() string {
	switch t {
	case IdentName:
		return "IdentName"
	case TypeName:
		return "TypeName"
	case FieldNumber:
		return "FieldNumber"
	default:
		return fmt.Sprintf("<invalid:%v>", uint8(t))
	}
}

// Bit mask in Token.attrs to indicate if a Name token is followed by the
// separator char ':'. The field name separator char is optional for message
// field or repeated message field, but required for all other types. Decoder
// simply indicates whether a Name token is followed by separator or not.  It is
// up to the prototext package to validate.
const hasSeparator = 1 << 7

// Scalar value types.
const (
	numberValue = iota + 1
	stringValue
	literalValue
)

// Bit mask in Token.numAttrs to indicate that the number is a negative.
const isNegative = 1 << 7

// Token provides a parsed token kind and value. Values are provided by the
// different accessor methods.
type Token struct {
	// Kind of the Token object.
	kind Kind
	// attrs contains metadata for the following Kinds:
	// Name: hasSeparator bit and one of NameKind.
	// Scalar: one of numberValue, stringValue, literalValue.
	attrs uint8
	// numAttrs contains metadata for numberValue:
	// - highest bit is whether negative or positive.
	// - lower bits indicate one of numDec, numHex, numOct, numFloat.
	numAttrs uint8
	// pos provides the position of the token in the original input.
	pos int
	// raw bytes of the serialized token.
	// This is a subslice into the original input.
	raw []byte
	// str contains parsed string for the following:
	// - stringValue of Scalar kind
	// - numberValue of Scalar kind
	// - TypeName of Name kind
	str string
}

// Kind returns the token kind.
func (t Token) Kind() Kind {
	return t.kind
}

// RawString returns the read value in string.
func (t Token) RawString() string {
	return string(t.raw)
}

// Pos returns the token position from the input.
func (t Token) Pos() int {
	return t.pos
}

// NameKind returns IdentName, TypeName or FieldNumber.
// It panics if type is not Name.
func (t Token) NameKind() NameKind {
	if t.kind == Name {
		return NameKind(t.attrs &^ hasSeparator)
	}
	panic(fmt.Sprintf("Token is not a Name type: %s", t.kind))
}

// HasSeparator returns true if the field name is followed by the separator char
// ':', else false. It panics if type is not Name.
func (t Token) HasSeparator() bool {
	if t.kind == Name {
		return t.attrs&hasSeparator != 0
	}
	panic(fmt.Sprintf("Token is not a Name type: %s", t.kind))
}

// IdentName returns the value for IdentName type.
func (t Token) IdentName() string {
	if t.kind == Name && t.attrs&uint8(IdentName) != 0 {
		return string(t.raw)
	}
	panic(fmt.Sprintf("Token is not an IdentName: %s:%s", t.kind, NameKind(t.attrs&^hasSeparator)))
}

// TypeName returns the value for TypeName type.
func (t Token) TypeName() string {
	if t.kind == Name && t.attrs&uint8(TypeName) != 0 {
		return t.str
	}
	panic(fmt.Sprintf("Token is not a TypeName: %s:%s", t.kind, NameKind(t.attrs&^hasSeparator)))
}

// FieldNumber returns the value for FieldNumber type. It returns a
// non-negative int32 value. Caller will still need to validate for the correct
// field number range.
func (t Token) FieldNumber() int32 {
	if t.kind != Name || t.attrs&uint8(FieldNumber) == 0 {
		panic(fmt.Sprintf("Token is not a FieldNumber: %s:%s", t.kind, NameKind(t.attrs&^hasSeparator)))
	}
	// Following should not return an error as it had already been called right
	// before this Token was constructed.
	num, _ := strconv.ParseInt(string(t.raw), 10, 32)
	return int32(num)
}

// String returns the string value for a Scalar type.
func (t Token) String() (string, bool) {
	if t.kind != Scalar || t.attrs != stringValue {
		return "", false
	}
	return t.str, true
}

// Enum returns the literal value for a Scalar type for use as enum literals.
func (t Token) Enum() (string, bool) {
	if t.kind != Scalar || t.attrs != literalValue || (len(t.raw) > 0 && t.raw[0] == '-') {
		return "", false
	}
	return string(t.raw), true
}

// Bool returns the bool value for a Scalar type.
func (t Token) Bool() (bool, bool) {
	if t.kind != Scalar {
		return false, false
	}
	switch t.attrs {
	case literalValue:
		if b, ok := boolLits[string(t.raw)]; ok {
			return b, true
		}
	case numberValue:
		// Unsigned integer representation of 0 or 1 is permitted: 00, 0x0, 01,
		// 0x1, etc.
		n, err := strconv.ParseUint(t.str, 0, 64)
		if err == nil {
			switch n {
			case 0:
				return false, true
			case 1:
				return true, true
			}
		}
	}
	return false, false
}

// These exact boolean literals are the ones supported in C++.
var boolLits = map[string]bool{
	"t":     true,
	"true":  true,
	"True":  true,
	"f":     false,
	"false": false,
	"False": false,
}

// Uint64 returns the uint64 value for a Scalar type.
func (t Token) Uint64() (uint64, bool) {
	if t.kind != Scalar || t.attrs != numberValue ||
		t.numAttrs&isNegative > 0 || t.numAttrs&numFloat > 0 {
		return 0, false
	}
	n, err := strconv.ParseUint(t.str, 0, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

// Uint32 returns the uint32 value for a Scalar type.
func (t Token) Uint32() (uint32, bool) {
	if t.kind != Scalar || t.attrs != numberValue ||
		t.numAttrs&isNegative > 0 || t.numAttrs&numFloat > 0 {
		return 0, false
	}
	n, err := strconv.ParseUint(t.str, 0, 32)
	if err != nil {
		return 0, false
	}
	return uint32(n), true
}

// Int64 returns the int64 value for a Scalar type.
func (t Token) Int64() (int64, bool) {
	if t.kind != Scalar || t.attrs != numberValue || t.numAttrs&numFloat > 0 {
		return 0, false
	}
	if n, err := strconv.ParseInt(t.str, 0, 64); err == nil {
		return n, true
	}
	// C++ accepts large positive hex numbers as negative values.
	// This feature is here for proto1 backwards compatibility purposes.
	if flags.ProtoLegacy && (t.numAttrs == numHex) {
		if n, err := strconv.ParseUint(t.str, 0, 64); err == nil {
			return int64(n), true
		}
	}
	return 0, false
}

// Int32 returns the int32 value for a Scalar type.
func (t Token) Int32() (int32, bool) {
	if t.kind != Scalar || t.attrs != numberValue || t.numAttrs&numFloat > 0 {
		return 0, false
	}
	if n, err := strconv.ParseInt(t.str, 0, 32); err == nil {
		return int32(n), true
	}
	// C++ accepts large positive hex numbers as negative values.
	// This feature is here for proto1 backwards compatibility purposes.
	if flags.ProtoLegacy && (t.numAttrs == numHex) {
		if n, err := strconv.ParseUint(t.str, 0, 32); err == nil {
			return int32(n), true
		}
	}
	return 0, false
}

// Float64 returns the float64 value for a Scalar type.
func (t Token) Float64() (float64, bool) {
	if t.kind != Scalar {
		return 0, false
	}
	switch t.attrs {
	case literalValue:
		if f, ok := floatLits[strings.ToLower(string(t.raw))]; ok {
			return f, true
		}
	case numberValue:
		n, err := strconv.ParseFloat(t.str, 64)
		if err == nil {
			return n, true
		}
		nerr := err.(*strconv.NumError)
		if nerr.Err == strconv.ErrRange {
			return n, true
		}
	}
	return 0, false
}

// Float32 returns the float32 value for a Scalar type.
func (t Token) Float32() (float32, bool) {
	if t.kind != Scalar {
		return 0, false
	}
	switch t.attrs {
	case literalValue:
		if f, ok := floatLits[strings.ToLower(string(t.raw))]; ok {
			return float32(f), true
		}
	case numberValue:
		n, err := strconv.ParseFloat(t.str, 64)
		if err == nil {
			// Overflows are treated as (-)infinity.
			return float32(n), true
		}
		nerr := err.(*strconv.NumError)
		if nerr.Err == strconv.ErrRange {
			return float32(n), true
		}
	}
	return 0, false
}

// These are the supported float literals which C++ permits case-insensitive
// variants of these.
var floatLits = map[string]float64{
	"nan":       math.NaN(),
	"inf":       math.Inf(1),
	"infinity":  math.Inf(1),
	"-inf":      math.Inf(-1),
	"-infinity": math.Inf(-1),
}

// TokenEquals returns true if given Tokens are equal, else false.
func TokenEquals(x, y Token) bool {
	return x.kind == y.kind &&
		x.attrs == y.attrs &&
		x.numAttrs == y.numAttrs &&
		x.pos == y.pos &&
		bytes.Equal(x.raw, y.raw) &&
		x.str == y.str
}
