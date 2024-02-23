// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"fmt"
	"strconv"
)

// Kind represents a token kind expressible in the JSON format.
type Kind uint16

const (
	Invalid Kind = (1 << iota) / 2
	EOF
	Null
	Bool
	Number
	String
	Name
	ObjectOpen
	ObjectClose
	ArrayOpen
	ArrayClose

	// comma is only for parsing in between tokens and
	// does not need to be exported.
	comma
)

func (k Kind) String() string {
	switch k {
	case EOF:
		return "eof"
	case Null:
		return "null"
	case Bool:
		return "bool"
	case Number:
		return "number"
	case String:
		return "string"
	case ObjectOpen:
		return "{"
	case ObjectClose:
		return "}"
	case Name:
		return "name"
	case ArrayOpen:
		return "["
	case ArrayClose:
		return "]"
	case comma:
		return ","
	}
	return "<invalid>"
}

// Token provides a parsed token kind and value.
//
// Values are provided by the difference accessor methods. The accessor methods
// Name, Bool, and ParsedString will panic if called on the wrong kind. There
// are different accessor methods for the Number kind for converting to the
// appropriate Go numeric type and those methods have the ok return value.
type Token struct {
	// Token kind.
	kind Kind
	// pos provides the position of the token in the original input.
	pos int
	// raw bytes of the serialized token.
	// This is a subslice into the original input.
	raw []byte
	// boo is parsed boolean value.
	boo bool
	// str is parsed string value.
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

// Name returns the object name if token is Name, else it panics.
func (t Token) Name() string {
	if t.kind == Name {
		return t.str
	}
	panic(fmt.Sprintf("Token is not a Name: %v", t.RawString()))
}

// Bool returns the bool value if token kind is Bool, else it panics.
func (t Token) Bool() bool {
	if t.kind == Bool {
		return t.boo
	}
	panic(fmt.Sprintf("Token is not a Bool: %v", t.RawString()))
}

// ParsedString returns the string value for a JSON string token or the read
// value in string if token is not a string.
func (t Token) ParsedString() string {
	if t.kind == String {
		return t.str
	}
	panic(fmt.Sprintf("Token is not a String: %v", t.RawString()))
}

// Float returns the floating-point number if token kind is Number.
//
// The floating-point precision is specified by the bitSize parameter: 32 for
// float32 or 64 for float64. If bitSize=32, the result still has type float64,
// but it will be convertible to float32 without changing its value. It will
// return false if the number exceeds the floating point limits for given
// bitSize.
func (t Token) Float(bitSize int) (float64, bool) {
	if t.kind != Number {
		return 0, false
	}
	f, err := strconv.ParseFloat(t.RawString(), bitSize)
	if err != nil {
		return 0, false
	}
	return f, true
}

// Int returns the signed integer number if token is Number.
//
// The given bitSize specifies the integer type that the result must fit into.
// It returns false if the number is not an integer value or if the result
// exceeds the limits for given bitSize.
func (t Token) Int(bitSize int) (int64, bool) {
	s, ok := t.getIntStr()
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, bitSize)
	if err != nil {
		return 0, false
	}
	return n, true
}

// Uint returns the signed integer number if token is Number.
//
// The given bitSize specifies the unsigned integer type that the result must
// fit into. It returns false if the number is not an unsigned integer value
// or if the result exceeds the limits for given bitSize.
func (t Token) Uint(bitSize int) (uint64, bool) {
	s, ok := t.getIntStr()
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseUint(s, 10, bitSize)
	if err != nil {
		return 0, false
	}
	return n, true
}

func (t Token) getIntStr() (string, bool) {
	if t.kind != Number {
		return "", false
	}
	parts, ok := parseNumberParts(t.raw)
	if !ok {
		return "", false
	}
	return normalizeToIntString(parts)
}

// TokenEquals returns true if given Tokens are equal, else false.
func TokenEquals(x, y Token) bool {
	return x.kind == y.kind &&
		x.pos == y.pos &&
		bytes.Equal(x.raw, y.raw) &&
		x.boo == y.boo &&
		x.str == y.str
}
