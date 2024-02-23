// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"unicode/utf8"

	"google.golang.org/protobuf/internal/errors"
)

// call specifies which Decoder method was invoked.
type call uint8

const (
	readCall call = iota
	peekCall
)

const unexpectedFmt = "unexpected token %s"

// ErrUnexpectedEOF means that EOF was encountered in the middle of the input.
var ErrUnexpectedEOF = errors.New("%v", io.ErrUnexpectedEOF)

// Decoder is a token-based JSON decoder.
type Decoder struct {
	// lastCall is last method called, either readCall or peekCall.
	// Initial value is readCall.
	lastCall call

	// lastToken contains the last read token.
	lastToken Token

	// lastErr contains the last read error.
	lastErr error

	// openStack is a stack containing ObjectOpen and ArrayOpen values. The
	// top of stack represents the object or the array the current value is
	// directly located in.
	openStack []Kind

	// orig is used in reporting line and column.
	orig []byte
	// in contains the unconsumed input.
	in []byte
}

// NewDecoder returns a Decoder to read the given []byte.
func NewDecoder(b []byte) *Decoder {
	return &Decoder{orig: b, in: b}
}

// Peek looks ahead and returns the next token kind without advancing a read.
func (d *Decoder) Peek() (Token, error) {
	defer func() { d.lastCall = peekCall }()
	if d.lastCall == readCall {
		d.lastToken, d.lastErr = d.Read()
	}
	return d.lastToken, d.lastErr
}

// Read returns the next JSON token.
// It will return an error if there is no valid token.
func (d *Decoder) Read() (Token, error) {
	const scalar = Null | Bool | Number | String

	defer func() { d.lastCall = readCall }()
	if d.lastCall == peekCall {
		return d.lastToken, d.lastErr
	}

	tok, err := d.parseNext()
	if err != nil {
		return Token{}, err
	}

	switch tok.kind {
	case EOF:
		if len(d.openStack) != 0 ||
			d.lastToken.kind&scalar|ObjectClose|ArrayClose == 0 {
			return Token{}, ErrUnexpectedEOF
		}

	case Null:
		if !d.isValueNext() {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}

	case Bool, Number:
		if !d.isValueNext() {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}

	case String:
		if d.isValueNext() {
			break
		}
		// This string token should only be for a field name.
		if d.lastToken.kind&(ObjectOpen|comma) == 0 {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}
		if len(d.in) == 0 {
			return Token{}, ErrUnexpectedEOF
		}
		if c := d.in[0]; c != ':' {
			return Token{}, d.newSyntaxError(d.currPos(), `unexpected character %s, missing ":" after field name`, string(c))
		}
		tok.kind = Name
		d.consume(1)

	case ObjectOpen, ArrayOpen:
		if !d.isValueNext() {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}
		d.openStack = append(d.openStack, tok.kind)

	case ObjectClose:
		if len(d.openStack) == 0 ||
			d.lastToken.kind == comma ||
			d.openStack[len(d.openStack)-1] != ObjectOpen {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}
		d.openStack = d.openStack[:len(d.openStack)-1]

	case ArrayClose:
		if len(d.openStack) == 0 ||
			d.lastToken.kind == comma ||
			d.openStack[len(d.openStack)-1] != ArrayOpen {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}
		d.openStack = d.openStack[:len(d.openStack)-1]

	case comma:
		if len(d.openStack) == 0 ||
			d.lastToken.kind&(scalar|ObjectClose|ArrayClose) == 0 {
			return Token{}, d.newSyntaxError(tok.pos, unexpectedFmt, tok.RawString())
		}
	}

	// Update d.lastToken only after validating token to be in the right sequence.
	d.lastToken = tok

	if d.lastToken.kind == comma {
		return d.Read()
	}
	return tok, nil
}

// Any sequence that looks like a non-delimiter (for error reporting).
var errRegexp = regexp.MustCompile(`^([-+._a-zA-Z0-9]{1,32}|.)`)

// parseNext parses for the next JSON token. It returns a Token object for
// different types, except for Name. It does not handle whether the next token
// is in a valid sequence or not.
func (d *Decoder) parseNext() (Token, error) {
	// Trim leading spaces.
	d.consume(0)

	in := d.in
	if len(in) == 0 {
		return d.consumeToken(EOF, 0), nil
	}

	switch in[0] {
	case 'n':
		if n := matchWithDelim("null", in); n != 0 {
			return d.consumeToken(Null, n), nil
		}

	case 't':
		if n := matchWithDelim("true", in); n != 0 {
			return d.consumeBoolToken(true, n), nil
		}

	case 'f':
		if n := matchWithDelim("false", in); n != 0 {
			return d.consumeBoolToken(false, n), nil
		}

	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		if n, ok := parseNumber(in); ok {
			return d.consumeToken(Number, n), nil
		}

	case '"':
		s, n, err := d.parseString(in)
		if err != nil {
			return Token{}, err
		}
		return d.consumeStringToken(s, n), nil

	case '{':
		return d.consumeToken(ObjectOpen, 1), nil

	case '}':
		return d.consumeToken(ObjectClose, 1), nil

	case '[':
		return d.consumeToken(ArrayOpen, 1), nil

	case ']':
		return d.consumeToken(ArrayClose, 1), nil

	case ',':
		return d.consumeToken(comma, 1), nil
	}
	return Token{}, d.newSyntaxError(d.currPos(), "invalid value %s", errRegexp.Find(in))
}

// newSyntaxError returns an error with line and column information useful for
// syntax errors.
func (d *Decoder) newSyntaxError(pos int, f string, x ...interface{}) error {
	e := errors.New(f, x...)
	line, column := d.Position(pos)
	return errors.New("syntax error (line %d:%d): %v", line, column, e)
}

// Position returns line and column number of given index of the original input.
// It will panic if index is out of range.
func (d *Decoder) Position(idx int) (line int, column int) {
	b := d.orig[:idx]
	line = bytes.Count(b, []byte("\n")) + 1
	if i := bytes.LastIndexByte(b, '\n'); i >= 0 {
		b = b[i+1:]
	}
	column = utf8.RuneCount(b) + 1 // ignore multi-rune characters
	return line, column
}

// currPos returns the current index position of d.in from d.orig.
func (d *Decoder) currPos() int {
	return len(d.orig) - len(d.in)
}

// matchWithDelim matches s with the input b and verifies that the match
// terminates with a delimiter of some form (e.g., r"[^-+_.a-zA-Z0-9]").
// As a special case, EOF is considered a delimiter. It returns the length of s
// if there is a match, else 0.
func matchWithDelim(s string, b []byte) int {
	if !bytes.HasPrefix(b, []byte(s)) {
		return 0
	}

	n := len(s)
	if n < len(b) && isNotDelim(b[n]) {
		return 0
	}
	return n
}

// isNotDelim returns true if given byte is a not delimiter character.
func isNotDelim(c byte) bool {
	return (c == '-' || c == '+' || c == '.' || c == '_' ||
		('a' <= c && c <= 'z') ||
		('A' <= c && c <= 'Z') ||
		('0' <= c && c <= '9'))
}

// consume consumes n bytes of input and any subsequent whitespace.
func (d *Decoder) consume(n int) {
	d.in = d.in[n:]
	for len(d.in) > 0 {
		switch d.in[0] {
		case ' ', '\n', '\r', '\t':
			d.in = d.in[1:]
		default:
			return
		}
	}
}

// isValueNext returns true if next type should be a JSON value: Null,
// Number, String or Bool.
func (d *Decoder) isValueNext() bool {
	if len(d.openStack) == 0 {
		return d.lastToken.kind == 0
	}

	start := d.openStack[len(d.openStack)-1]
	switch start {
	case ObjectOpen:
		return d.lastToken.kind&Name != 0
	case ArrayOpen:
		return d.lastToken.kind&(ArrayOpen|comma) != 0
	}
	panic(fmt.Sprintf(
		"unreachable logic in Decoder.isValueNext, lastToken.kind: %v, openStack: %v",
		d.lastToken.kind, start))
}

// consumeToken constructs a Token for given Kind with raw value derived from
// current d.in and given size, and consumes the given size-length of it.
func (d *Decoder) consumeToken(kind Kind, size int) Token {
	tok := Token{
		kind: kind,
		raw:  d.in[:size],
		pos:  len(d.orig) - len(d.in),
	}
	d.consume(size)
	return tok
}

// consumeBoolToken constructs a Token for a Bool kind with raw value derived from
// current d.in and given size.
func (d *Decoder) consumeBoolToken(b bool, size int) Token {
	tok := Token{
		kind: Bool,
		raw:  d.in[:size],
		pos:  len(d.orig) - len(d.in),
		boo:  b,
	}
	d.consume(size)
	return tok
}

// consumeStringToken constructs a Token for a String kind with raw value derived
// from current d.in and given size.
func (d *Decoder) consumeStringToken(s string, size int) Token {
	tok := Token{
		kind: String,
		raw:  d.in[:size],
		pos:  len(d.orig) - len(d.in),
		str:  s,
	}
	d.consume(size)
	return tok
}

// Clone returns a copy of the Decoder for use in reading ahead the next JSON
// object, array or other values without affecting current Decoder.
func (d *Decoder) Clone() *Decoder {
	ret := *d
	ret.openStack = append([]Kind(nil), ret.openStack...)
	return &ret
}
