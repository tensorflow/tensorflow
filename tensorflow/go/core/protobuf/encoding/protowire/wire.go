// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protowire parses and formats the raw wire encoding.
// See https://protobuf.dev/programming-guides/encoding.
//
// For marshaling and unmarshaling entire protobuf messages,
// use the [google.golang.org/protobuf/proto] package instead.
package protowire

import (
	"io"
	"math"
	"math/bits"

	"google.golang.org/protobuf/internal/errors"
)

// Number represents the field number.
type Number int32

const (
	MinValidNumber        Number = 1
	FirstReservedNumber   Number = 19000
	LastReservedNumber    Number = 19999
	MaxValidNumber        Number = 1<<29 - 1
	DefaultRecursionLimit        = 10000
)

// IsValid reports whether the field number is semantically valid.
func (n Number) IsValid() bool {
	return MinValidNumber <= n && n <= MaxValidNumber
}

// Type represents the wire type.
type Type int8

const (
	VarintType     Type = 0
	Fixed32Type    Type = 5
	Fixed64Type    Type = 1
	BytesType      Type = 2
	StartGroupType Type = 3
	EndGroupType   Type = 4
)

const (
	_ = -iota
	errCodeTruncated
	errCodeFieldNumber
	errCodeOverflow
	errCodeReserved
	errCodeEndGroup
	errCodeRecursionDepth
)

var (
	errFieldNumber = errors.New("invalid field number")
	errOverflow    = errors.New("variable length integer overflow")
	errReserved    = errors.New("cannot parse reserved wire type")
	errEndGroup    = errors.New("mismatching end group marker")
	errParse       = errors.New("parse error")
)

// ParseError converts an error code into an error value.
// This returns nil if n is a non-negative number.
func ParseError(n int) error {
	if n >= 0 {
		return nil
	}
	switch n {
	case errCodeTruncated:
		return io.ErrUnexpectedEOF
	case errCodeFieldNumber:
		return errFieldNumber
	case errCodeOverflow:
		return errOverflow
	case errCodeReserved:
		return errReserved
	case errCodeEndGroup:
		return errEndGroup
	default:
		return errParse
	}
}

// ConsumeField parses an entire field record (both tag and value) and returns
// the field number, the wire type, and the total length.
// This returns a negative length upon an error (see [ParseError]).
//
// The total length includes the tag header and the end group marker (if the
// field is a group).
func ConsumeField(b []byte) (Number, Type, int) {
	num, typ, n := ConsumeTag(b)
	if n < 0 {
		return 0, 0, n // forward error code
	}
	m := ConsumeFieldValue(num, typ, b[n:])
	if m < 0 {
		return 0, 0, m // forward error code
	}
	return num, typ, n + m
}

// ConsumeFieldValue parses a field value and returns its length.
// This assumes that the field [Number] and wire [Type] have already been parsed.
// This returns a negative length upon an error (see [ParseError]).
//
// When parsing a group, the length includes the end group marker and
// the end group is verified to match the starting field number.
func ConsumeFieldValue(num Number, typ Type, b []byte) (n int) {
	return consumeFieldValueD(num, typ, b, DefaultRecursionLimit)
}

func consumeFieldValueD(num Number, typ Type, b []byte, depth int) (n int) {
	switch typ {
	case VarintType:
		_, n = ConsumeVarint(b)
		return n
	case Fixed32Type:
		_, n = ConsumeFixed32(b)
		return n
	case Fixed64Type:
		_, n = ConsumeFixed64(b)
		return n
	case BytesType:
		_, n = ConsumeBytes(b)
		return n
	case StartGroupType:
		if depth < 0 {
			return errCodeRecursionDepth
		}
		n0 := len(b)
		for {
			num2, typ2, n := ConsumeTag(b)
			if n < 0 {
				return n // forward error code
			}
			b = b[n:]
			if typ2 == EndGroupType {
				if num != num2 {
					return errCodeEndGroup
				}
				return n0 - len(b)
			}

			n = consumeFieldValueD(num2, typ2, b, depth-1)
			if n < 0 {
				return n // forward error code
			}
			b = b[n:]
		}
	case EndGroupType:
		return errCodeEndGroup
	default:
		return errCodeReserved
	}
}

// AppendTag encodes num and typ as a varint-encoded tag and appends it to b.
func AppendTag(b []byte, num Number, typ Type) []byte {
	return AppendVarint(b, EncodeTag(num, typ))
}

// ConsumeTag parses b as a varint-encoded tag, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeTag(b []byte) (Number, Type, int) {
	v, n := ConsumeVarint(b)
	if n < 0 {
		return 0, 0, n // forward error code
	}
	num, typ := DecodeTag(v)
	if num < MinValidNumber {
		return 0, 0, errCodeFieldNumber
	}
	return num, typ, n
}

func SizeTag(num Number) int {
	return SizeVarint(EncodeTag(num, 0)) // wire type has no effect on size
}

// AppendVarint appends v to b as a varint-encoded uint64.
func AppendVarint(b []byte, v uint64) []byte {
	switch {
	case v < 1<<7:
		b = append(b, byte(v))
	case v < 1<<14:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte(v>>7))
	case v < 1<<21:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte(v>>14))
	case v < 1<<28:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte(v>>21))
	case v < 1<<35:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte(v>>28))
	case v < 1<<42:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte((v>>28)&0x7f|0x80),
			byte(v>>35))
	case v < 1<<49:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte((v>>28)&0x7f|0x80),
			byte((v>>35)&0x7f|0x80),
			byte(v>>42))
	case v < 1<<56:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte((v>>28)&0x7f|0x80),
			byte((v>>35)&0x7f|0x80),
			byte((v>>42)&0x7f|0x80),
			byte(v>>49))
	case v < 1<<63:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte((v>>28)&0x7f|0x80),
			byte((v>>35)&0x7f|0x80),
			byte((v>>42)&0x7f|0x80),
			byte((v>>49)&0x7f|0x80),
			byte(v>>56))
	default:
		b = append(b,
			byte((v>>0)&0x7f|0x80),
			byte((v>>7)&0x7f|0x80),
			byte((v>>14)&0x7f|0x80),
			byte((v>>21)&0x7f|0x80),
			byte((v>>28)&0x7f|0x80),
			byte((v>>35)&0x7f|0x80),
			byte((v>>42)&0x7f|0x80),
			byte((v>>49)&0x7f|0x80),
			byte((v>>56)&0x7f|0x80),
			1)
	}
	return b
}

// ConsumeVarint parses b as a varint-encoded uint64, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeVarint(b []byte) (v uint64, n int) {
	var y uint64
	if len(b) <= 0 {
		return 0, errCodeTruncated
	}
	v = uint64(b[0])
	if v < 0x80 {
		return v, 1
	}
	v -= 0x80

	if len(b) <= 1 {
		return 0, errCodeTruncated
	}
	y = uint64(b[1])
	v += y << 7
	if y < 0x80 {
		return v, 2
	}
	v -= 0x80 << 7

	if len(b) <= 2 {
		return 0, errCodeTruncated
	}
	y = uint64(b[2])
	v += y << 14
	if y < 0x80 {
		return v, 3
	}
	v -= 0x80 << 14

	if len(b) <= 3 {
		return 0, errCodeTruncated
	}
	y = uint64(b[3])
	v += y << 21
	if y < 0x80 {
		return v, 4
	}
	v -= 0x80 << 21

	if len(b) <= 4 {
		return 0, errCodeTruncated
	}
	y = uint64(b[4])
	v += y << 28
	if y < 0x80 {
		return v, 5
	}
	v -= 0x80 << 28

	if len(b) <= 5 {
		return 0, errCodeTruncated
	}
	y = uint64(b[5])
	v += y << 35
	if y < 0x80 {
		return v, 6
	}
	v -= 0x80 << 35

	if len(b) <= 6 {
		return 0, errCodeTruncated
	}
	y = uint64(b[6])
	v += y << 42
	if y < 0x80 {
		return v, 7
	}
	v -= 0x80 << 42

	if len(b) <= 7 {
		return 0, errCodeTruncated
	}
	y = uint64(b[7])
	v += y << 49
	if y < 0x80 {
		return v, 8
	}
	v -= 0x80 << 49

	if len(b) <= 8 {
		return 0, errCodeTruncated
	}
	y = uint64(b[8])
	v += y << 56
	if y < 0x80 {
		return v, 9
	}
	v -= 0x80 << 56

	if len(b) <= 9 {
		return 0, errCodeTruncated
	}
	y = uint64(b[9])
	v += y << 63
	if y < 2 {
		return v, 10
	}
	return 0, errCodeOverflow
}

// SizeVarint returns the encoded size of a varint.
// The size is guaranteed to be within 1 and 10, inclusive.
func SizeVarint(v uint64) int {
	// This computes 1 + (bits.Len64(v)-1)/7.
	// 9/64 is a good enough approximation of 1/7
	return int(9*uint32(bits.Len64(v))+64) / 64
}

// AppendFixed32 appends v to b as a little-endian uint32.
func AppendFixed32(b []byte, v uint32) []byte {
	return append(b,
		byte(v>>0),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24))
}

// ConsumeFixed32 parses b as a little-endian uint32, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeFixed32(b []byte) (v uint32, n int) {
	if len(b) < 4 {
		return 0, errCodeTruncated
	}
	v = uint32(b[0])<<0 | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return v, 4
}

// SizeFixed32 returns the encoded size of a fixed32; which is always 4.
func SizeFixed32() int {
	return 4
}

// AppendFixed64 appends v to b as a little-endian uint64.
func AppendFixed64(b []byte, v uint64) []byte {
	return append(b,
		byte(v>>0),
		byte(v>>8),
		byte(v>>16),
		byte(v>>24),
		byte(v>>32),
		byte(v>>40),
		byte(v>>48),
		byte(v>>56))
}

// ConsumeFixed64 parses b as a little-endian uint64, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeFixed64(b []byte) (v uint64, n int) {
	if len(b) < 8 {
		return 0, errCodeTruncated
	}
	v = uint64(b[0])<<0 | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 | uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	return v, 8
}

// SizeFixed64 returns the encoded size of a fixed64; which is always 8.
func SizeFixed64() int {
	return 8
}

// AppendBytes appends v to b as a length-prefixed bytes value.
func AppendBytes(b []byte, v []byte) []byte {
	return append(AppendVarint(b, uint64(len(v))), v...)
}

// ConsumeBytes parses b as a length-prefixed bytes value, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeBytes(b []byte) (v []byte, n int) {
	m, n := ConsumeVarint(b)
	if n < 0 {
		return nil, n // forward error code
	}
	if m > uint64(len(b[n:])) {
		return nil, errCodeTruncated
	}
	return b[n:][:m], n + int(m)
}

// SizeBytes returns the encoded size of a length-prefixed bytes value,
// given only the length.
func SizeBytes(n int) int {
	return SizeVarint(uint64(n)) + n
}

// AppendString appends v to b as a length-prefixed bytes value.
func AppendString(b []byte, v string) []byte {
	return append(AppendVarint(b, uint64(len(v))), v...)
}

// ConsumeString parses b as a length-prefixed bytes value, reporting its length.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeString(b []byte) (v string, n int) {
	bb, n := ConsumeBytes(b)
	return string(bb), n
}

// AppendGroup appends v to b as group value, with a trailing end group marker.
// The value v must not contain the end marker.
func AppendGroup(b []byte, num Number, v []byte) []byte {
	return AppendVarint(append(b, v...), EncodeTag(num, EndGroupType))
}

// ConsumeGroup parses b as a group value until the trailing end group marker,
// and verifies that the end marker matches the provided num. The value v
// does not contain the end marker, while the length does contain the end marker.
// This returns a negative length upon an error (see [ParseError]).
func ConsumeGroup(num Number, b []byte) (v []byte, n int) {
	n = ConsumeFieldValue(num, StartGroupType, b)
	if n < 0 {
		return nil, n // forward error code
	}
	b = b[:n]

	// Truncate off end group marker, but need to handle denormalized varints.
	// Assuming end marker is never 0 (which is always the case since
	// EndGroupType is non-zero), we can truncate all trailing bytes where the
	// lower 7 bits are all zero (implying that the varint is denormalized).
	for len(b) > 0 && b[len(b)-1]&0x7f == 0 {
		b = b[:len(b)-1]
	}
	b = b[:len(b)-SizeTag(num)]
	return b, n
}

// SizeGroup returns the encoded size of a group, given only the length.
func SizeGroup(num Number, n int) int {
	return n + SizeTag(num)
}

// DecodeTag decodes the field [Number] and wire [Type] from its unified form.
// The [Number] is -1 if the decoded field number overflows int32.
// Other than overflow, this does not check for field number validity.
func DecodeTag(x uint64) (Number, Type) {
	// NOTE: MessageSet allows for larger field numbers than normal.
	if x>>3 > uint64(math.MaxInt32) {
		return -1, 0
	}
	return Number(x >> 3), Type(x & 7)
}

// EncodeTag encodes the field [Number] and wire [Type] into its unified form.
func EncodeTag(num Number, typ Type) uint64 {
	return uint64(num)<<3 | uint64(typ&7)
}

// DecodeZigZag decodes a zig-zag-encoded uint64 as an int64.
//
//	Input:  {…,  5,  3,  1,  0,  2,  4,  6, …}
//	Output: {…, -3, -2, -1,  0, +1, +2, +3, …}
func DecodeZigZag(x uint64) int64 {
	return int64(x>>1) ^ int64(x)<<63>>63
}

// EncodeZigZag encodes an int64 as a zig-zag-encoded uint64.
//
//	Input:  {…, -3, -2, -1,  0, +1, +2, +3, …}
//	Output: {…,  5,  3,  1,  0,  2,  4,  6, …}
func EncodeZigZag(x int64) uint64 {
	return uint64(x<<1) ^ uint64(x>>63)
}

// DecodeBool decodes a uint64 as a bool.
//
//	Input:  {    0,    1,    2, …}
//	Output: {false, true, true, …}
func DecodeBool(x uint64) bool {
	return x != 0
}

// EncodeBool encodes a bool as a uint64.
//
//	Input:  {false, true}
//	Output: {    0,    1}
func EncodeBool(x bool) uint64 {
	if x {
		return 1
	}
	return 0
}
