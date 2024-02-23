// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"math"
	"math/bits"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/errors"
)

// kind represents an encoding type.
type kind uint8

const (
	_ kind = (1 << iota) / 2
	name
	scalar
	objectOpen
	objectClose
	arrayOpen
	arrayClose
)

// Encoder provides methods to write out JSON constructs and values. The user is
// responsible for producing valid sequences of JSON constructs and values.
type Encoder struct {
	indent   string
	lastKind kind
	indents  []byte
	out      []byte
}

// NewEncoder returns an Encoder.
//
// If indent is a non-empty string, it causes every entry for an Array or Object
// to be preceded by the indent and trailed by a newline.
func NewEncoder(buf []byte, indent string) (*Encoder, error) {
	e := &Encoder{
		out: buf,
	}
	if len(indent) > 0 {
		if strings.Trim(indent, " \t") != "" {
			return nil, errors.New("indent may only be composed of space or tab characters")
		}
		e.indent = indent
	}
	return e, nil
}

// Bytes returns the content of the written bytes.
func (e *Encoder) Bytes() []byte {
	return e.out
}

// WriteNull writes out the null value.
func (e *Encoder) WriteNull() {
	e.prepareNext(scalar)
	e.out = append(e.out, "null"...)
}

// WriteBool writes out the given boolean value.
func (e *Encoder) WriteBool(b bool) {
	e.prepareNext(scalar)
	if b {
		e.out = append(e.out, "true"...)
	} else {
		e.out = append(e.out, "false"...)
	}
}

// WriteString writes out the given string in JSON string value. Returns error
// if input string contains invalid UTF-8.
func (e *Encoder) WriteString(s string) error {
	e.prepareNext(scalar)
	var err error
	if e.out, err = appendString(e.out, s); err != nil {
		return err
	}
	return nil
}

// Sentinel error used for indicating invalid UTF-8.
var errInvalidUTF8 = errors.New("invalid UTF-8")

func appendString(out []byte, in string) ([]byte, error) {
	out = append(out, '"')
	i := indexNeedEscapeInString(in)
	in, out = in[i:], append(out, in[:i]...)
	for len(in) > 0 {
		switch r, n := utf8.DecodeRuneInString(in); {
		case r == utf8.RuneError && n == 1:
			return out, errInvalidUTF8
		case r < ' ' || r == '"' || r == '\\':
			out = append(out, '\\')
			switch r {
			case '"', '\\':
				out = append(out, byte(r))
			case '\b':
				out = append(out, 'b')
			case '\f':
				out = append(out, 'f')
			case '\n':
				out = append(out, 'n')
			case '\r':
				out = append(out, 'r')
			case '\t':
				out = append(out, 't')
			default:
				out = append(out, 'u')
				out = append(out, "0000"[1+(bits.Len32(uint32(r))-1)/4:]...)
				out = strconv.AppendUint(out, uint64(r), 16)
			}
			in = in[n:]
		default:
			i := indexNeedEscapeInString(in[n:])
			in, out = in[n+i:], append(out, in[:n+i]...)
		}
	}
	out = append(out, '"')
	return out, nil
}

// indexNeedEscapeInString returns the index of the character that needs
// escaping. If no characters need escaping, this returns the input length.
func indexNeedEscapeInString(s string) int {
	for i, r := range s {
		if r < ' ' || r == '\\' || r == '"' || r == utf8.RuneError {
			return i
		}
	}
	return len(s)
}

// WriteFloat writes out the given float and bitSize in JSON number value.
func (e *Encoder) WriteFloat(n float64, bitSize int) {
	e.prepareNext(scalar)
	e.out = appendFloat(e.out, n, bitSize)
}

// appendFloat formats given float in bitSize, and appends to the given []byte.
func appendFloat(out []byte, n float64, bitSize int) []byte {
	switch {
	case math.IsNaN(n):
		return append(out, `"NaN"`...)
	case math.IsInf(n, +1):
		return append(out, `"Infinity"`...)
	case math.IsInf(n, -1):
		return append(out, `"-Infinity"`...)
	}

	// JSON number formatting logic based on encoding/json.
	// See floatEncoder.encode for reference.
	fmt := byte('f')
	if abs := math.Abs(n); abs != 0 {
		if bitSize == 64 && (abs < 1e-6 || abs >= 1e21) ||
			bitSize == 32 && (float32(abs) < 1e-6 || float32(abs) >= 1e21) {
			fmt = 'e'
		}
	}
	out = strconv.AppendFloat(out, n, fmt, -1, bitSize)
	if fmt == 'e' {
		n := len(out)
		if n >= 4 && out[n-4] == 'e' && out[n-3] == '-' && out[n-2] == '0' {
			out[n-2] = out[n-1]
			out = out[:n-1]
		}
	}
	return out
}

// WriteInt writes out the given signed integer in JSON number value.
func (e *Encoder) WriteInt(n int64) {
	e.prepareNext(scalar)
	e.out = strconv.AppendInt(e.out, n, 10)
}

// WriteUint writes out the given unsigned integer in JSON number value.
func (e *Encoder) WriteUint(n uint64) {
	e.prepareNext(scalar)
	e.out = strconv.AppendUint(e.out, n, 10)
}

// StartObject writes out the '{' symbol.
func (e *Encoder) StartObject() {
	e.prepareNext(objectOpen)
	e.out = append(e.out, '{')
}

// EndObject writes out the '}' symbol.
func (e *Encoder) EndObject() {
	e.prepareNext(objectClose)
	e.out = append(e.out, '}')
}

// WriteName writes out the given string in JSON string value and the name
// separator ':'. Returns error if input string contains invalid UTF-8, which
// should not be likely as protobuf field names should be valid.
func (e *Encoder) WriteName(s string) error {
	e.prepareNext(name)
	var err error
	// Append to output regardless of error.
	e.out, err = appendString(e.out, s)
	e.out = append(e.out, ':')
	return err
}

// StartArray writes out the '[' symbol.
func (e *Encoder) StartArray() {
	e.prepareNext(arrayOpen)
	e.out = append(e.out, '[')
}

// EndArray writes out the ']' symbol.
func (e *Encoder) EndArray() {
	e.prepareNext(arrayClose)
	e.out = append(e.out, ']')
}

// prepareNext adds possible comma and indentation for the next value based
// on last type and indent option. It also updates lastKind to next.
func (e *Encoder) prepareNext(next kind) {
	defer func() {
		// Set lastKind to next.
		e.lastKind = next
	}()

	if len(e.indent) == 0 {
		// Need to add comma on the following condition.
		if e.lastKind&(scalar|objectClose|arrayClose) != 0 &&
			next&(name|scalar|objectOpen|arrayOpen) != 0 {
			e.out = append(e.out, ',')
			// For single-line output, add a random extra space after each
			// comma to make output unstable.
			if detrand.Bool() {
				e.out = append(e.out, ' ')
			}
		}
		return
	}

	switch {
	case e.lastKind&(objectOpen|arrayOpen) != 0:
		// If next type is NOT closing, add indent and newline.
		if next&(objectClose|arrayClose) == 0 {
			e.indents = append(e.indents, e.indent...)
			e.out = append(e.out, '\n')
			e.out = append(e.out, e.indents...)
		}

	case e.lastKind&(scalar|objectClose|arrayClose) != 0:
		switch {
		// If next type is either a value or name, add comma and newline.
		case next&(name|scalar|objectOpen|arrayOpen) != 0:
			e.out = append(e.out, ',', '\n')

		// If next type is a closing object or array, adjust indentation.
		case next&(objectClose|arrayClose) != 0:
			e.indents = e.indents[:len(e.indents)-len(e.indent)]
			e.out = append(e.out, '\n')
		}
		e.out = append(e.out, e.indents...)

	case e.lastKind&name != 0:
		e.out = append(e.out, ' ')
		// For multi-line output, add a random extra space after key: to make
		// output unstable.
		if detrand.Bool() {
			e.out = append(e.out, ' ')
		}
	}
}
