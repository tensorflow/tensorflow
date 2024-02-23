// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"unicode/utf8"

	"google.golang.org/protobuf/internal/errors"
)

// Decoder is a token-based textproto decoder.
type Decoder struct {
	// lastCall is last method called, either readCall or peekCall.
	// Initial value is readCall.
	lastCall call

	// lastToken contains the last read token.
	lastToken Token

	// lastErr contains the last read error.
	lastErr error

	// openStack is a stack containing the byte characters for MessageOpen and
	// ListOpen kinds. The top of stack represents the message or the list that
	// the current token is nested in. An empty stack means the current token is
	// at the top level message. The characters '{' and '<' both represent the
	// MessageOpen kind.
	openStack []byte

	// orig is used in reporting line and column.
	orig []byte
	// in contains the unconsumed input.
	in []byte
}

// NewDecoder returns a Decoder to read the given []byte.
func NewDecoder(b []byte) *Decoder {
	return &Decoder{orig: b, in: b}
}

// ErrUnexpectedEOF means that EOF was encountered in the middle of the input.
var ErrUnexpectedEOF = errors.New("%v", io.ErrUnexpectedEOF)

// call specifies which Decoder method was invoked.
type call uint8

const (
	readCall call = iota
	peekCall
)

// Peek looks ahead and returns the next token and error without advancing a read.
func (d *Decoder) Peek() (Token, error) {
	defer func() { d.lastCall = peekCall }()
	if d.lastCall == readCall {
		d.lastToken, d.lastErr = d.Read()
	}
	return d.lastToken, d.lastErr
}

// Read returns the next token.
// It will return an error if there is no valid token.
func (d *Decoder) Read() (Token, error) {
	defer func() { d.lastCall = readCall }()
	if d.lastCall == peekCall {
		return d.lastToken, d.lastErr
	}

	tok, err := d.parseNext(d.lastToken.kind)
	if err != nil {
		return Token{}, err
	}

	switch tok.kind {
	case comma, semicolon:
		tok, err = d.parseNext(tok.kind)
		if err != nil {
			return Token{}, err
		}
	}
	d.lastToken = tok
	return tok, nil
}

const (
	mismatchedFmt = "mismatched close character %q"
	unexpectedFmt = "unexpected character %q"
)

// parseNext parses the next Token based on given last kind.
func (d *Decoder) parseNext(lastKind Kind) (Token, error) {
	// Trim leading spaces.
	d.consume(0)
	isEOF := false
	if len(d.in) == 0 {
		isEOF = true
	}

	switch lastKind {
	case EOF:
		return d.consumeToken(EOF, 0, 0), nil

	case bof:
		// Start of top level message. Next token can be EOF or Name.
		if isEOF {
			return d.consumeToken(EOF, 0, 0), nil
		}
		return d.parseFieldName()

	case Name:
		// Next token can be MessageOpen, ListOpen or Scalar.
		if isEOF {
			return Token{}, ErrUnexpectedEOF
		}
		switch ch := d.in[0]; ch {
		case '{', '<':
			d.pushOpenStack(ch)
			return d.consumeToken(MessageOpen, 1, 0), nil
		case '[':
			d.pushOpenStack(ch)
			return d.consumeToken(ListOpen, 1, 0), nil
		default:
			return d.parseScalar()
		}

	case Scalar:
		openKind, closeCh := d.currentOpenKind()
		switch openKind {
		case bof:
			// Top level message.
			// 	Next token can be EOF, comma, semicolon or Name.
			if isEOF {
				return d.consumeToken(EOF, 0, 0), nil
			}
			switch d.in[0] {
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		case MessageOpen:
			// Next token can be MessageClose, comma, semicolon or Name.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case closeCh:
				d.popOpenStack()
				return d.consumeToken(MessageClose, 1, 0), nil
			case otherCloseChar[closeCh]:
				return Token{}, d.newSyntaxError(mismatchedFmt, ch)
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		case ListOpen:
			// Next token can be ListClose or comma.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case ']':
				d.popOpenStack()
				return d.consumeToken(ListClose, 1, 0), nil
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			default:
				return Token{}, d.newSyntaxError(unexpectedFmt, ch)
			}
		}

	case MessageOpen:
		// Next token can be MessageClose or Name.
		if isEOF {
			return Token{}, ErrUnexpectedEOF
		}
		_, closeCh := d.currentOpenKind()
		switch ch := d.in[0]; ch {
		case closeCh:
			d.popOpenStack()
			return d.consumeToken(MessageClose, 1, 0), nil
		case otherCloseChar[closeCh]:
			return Token{}, d.newSyntaxError(mismatchedFmt, ch)
		default:
			return d.parseFieldName()
		}

	case MessageClose:
		openKind, closeCh := d.currentOpenKind()
		switch openKind {
		case bof:
			// Top level message.
			// Next token can be EOF, comma, semicolon or Name.
			if isEOF {
				return d.consumeToken(EOF, 0, 0), nil
			}
			switch ch := d.in[0]; ch {
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		case MessageOpen:
			// Next token can be MessageClose, comma, semicolon or Name.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case closeCh:
				d.popOpenStack()
				return d.consumeToken(MessageClose, 1, 0), nil
			case otherCloseChar[closeCh]:
				return Token{}, d.newSyntaxError(mismatchedFmt, ch)
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		case ListOpen:
			// Next token can be ListClose or comma
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case closeCh:
				d.popOpenStack()
				return d.consumeToken(ListClose, 1, 0), nil
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			default:
				return Token{}, d.newSyntaxError(unexpectedFmt, ch)
			}
		}

	case ListOpen:
		// Next token can be ListClose, MessageStart or Scalar.
		if isEOF {
			return Token{}, ErrUnexpectedEOF
		}
		switch ch := d.in[0]; ch {
		case ']':
			d.popOpenStack()
			return d.consumeToken(ListClose, 1, 0), nil
		case '{', '<':
			d.pushOpenStack(ch)
			return d.consumeToken(MessageOpen, 1, 0), nil
		default:
			return d.parseScalar()
		}

	case ListClose:
		openKind, closeCh := d.currentOpenKind()
		switch openKind {
		case bof:
			// Top level message.
			// Next token can be EOF, comma, semicolon or Name.
			if isEOF {
				return d.consumeToken(EOF, 0, 0), nil
			}
			switch ch := d.in[0]; ch {
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		case MessageOpen:
			// Next token can be MessageClose, comma, semicolon or Name.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case closeCh:
				d.popOpenStack()
				return d.consumeToken(MessageClose, 1, 0), nil
			case otherCloseChar[closeCh]:
				return Token{}, d.newSyntaxError(mismatchedFmt, ch)
			case ',':
				return d.consumeToken(comma, 1, 0), nil
			case ';':
				return d.consumeToken(semicolon, 1, 0), nil
			default:
				return d.parseFieldName()
			}

		default:
			// It is not possible to have this case. Let it panic below.
		}

	case comma, semicolon:
		openKind, closeCh := d.currentOpenKind()
		switch openKind {
		case bof:
			// Top level message. Next token can be EOF or Name.
			if isEOF {
				return d.consumeToken(EOF, 0, 0), nil
			}
			return d.parseFieldName()

		case MessageOpen:
			// Next token can be MessageClose or Name.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case closeCh:
				d.popOpenStack()
				return d.consumeToken(MessageClose, 1, 0), nil
			case otherCloseChar[closeCh]:
				return Token{}, d.newSyntaxError(mismatchedFmt, ch)
			default:
				return d.parseFieldName()
			}

		case ListOpen:
			if lastKind == semicolon {
				// It is not be possible to have this case as logic here
				// should not have produced a semicolon Token when inside a
				// list. Let it panic below.
				break
			}
			// Next token can be MessageOpen or Scalar.
			if isEOF {
				return Token{}, ErrUnexpectedEOF
			}
			switch ch := d.in[0]; ch {
			case '{', '<':
				d.pushOpenStack(ch)
				return d.consumeToken(MessageOpen, 1, 0), nil
			default:
				return d.parseScalar()
			}
		}
	}

	line, column := d.Position(len(d.orig) - len(d.in))
	panic(fmt.Sprintf("Decoder.parseNext: bug at handling line %d:%d with lastKind=%v", line, column, lastKind))
}

var otherCloseChar = map[byte]byte{
	'}': '>',
	'>': '}',
}

// currentOpenKind indicates whether current position is inside a message, list
// or top-level message by returning MessageOpen, ListOpen or bof respectively.
// If the returned kind is either a MessageOpen or ListOpen, it also returns the
// corresponding closing character.
func (d *Decoder) currentOpenKind() (Kind, byte) {
	if len(d.openStack) == 0 {
		return bof, 0
	}
	openCh := d.openStack[len(d.openStack)-1]
	switch openCh {
	case '{':
		return MessageOpen, '}'
	case '<':
		return MessageOpen, '>'
	case '[':
		return ListOpen, ']'
	}
	panic(fmt.Sprintf("Decoder: openStack contains invalid byte %c", openCh))
}

func (d *Decoder) pushOpenStack(ch byte) {
	d.openStack = append(d.openStack, ch)
}

func (d *Decoder) popOpenStack() {
	d.openStack = d.openStack[:len(d.openStack)-1]
}

// parseFieldName parses field name and separator.
func (d *Decoder) parseFieldName() (tok Token, err error) {
	defer func() {
		if err == nil && d.tryConsumeChar(':') {
			tok.attrs |= hasSeparator
		}
	}()

	// Extension or Any type URL.
	if d.in[0] == '[' {
		return d.parseTypeName()
	}

	// Identifier.
	if size := parseIdent(d.in, false); size > 0 {
		return d.consumeToken(Name, size, uint8(IdentName)), nil
	}

	// Field number. Identify if input is a valid number that is not negative
	// and is decimal integer within 32-bit range.
	if num := parseNumber(d.in); num.size > 0 {
		str := num.string(d.in)
		if !num.neg && num.kind == numDec {
			if _, err := strconv.ParseInt(str, 10, 32); err == nil {
				return d.consumeToken(Name, num.size, uint8(FieldNumber)), nil
			}
		}
		return Token{}, d.newSyntaxError("invalid field number: %s", str)
	}

	return Token{}, d.newSyntaxError("invalid field name: %s", errId(d.in))
}

// parseTypeName parses Any type URL or extension field name. The name is
// enclosed in [ and ] characters. The C++ parser does not handle many legal URL
// strings. This implementation is more liberal and allows for the pattern
// ^[-_a-zA-Z0-9]+([./][-_a-zA-Z0-9]+)*`). Whitespaces and comments are allowed
// in between [ ], '.', '/' and the sub names.
func (d *Decoder) parseTypeName() (Token, error) {
	startPos := len(d.orig) - len(d.in)
	// Use alias s to advance first in order to use d.in for error handling.
	// Caller already checks for [ as first character.
	s := consume(d.in[1:], 0)
	if len(s) == 0 {
		return Token{}, ErrUnexpectedEOF
	}

	var name []byte
	for len(s) > 0 && isTypeNameChar(s[0]) {
		name = append(name, s[0])
		s = s[1:]
	}
	s = consume(s, 0)

	var closed bool
	for len(s) > 0 && !closed {
		switch {
		case s[0] == ']':
			s = s[1:]
			closed = true

		case s[0] == '/', s[0] == '.':
			if len(name) > 0 && (name[len(name)-1] == '/' || name[len(name)-1] == '.') {
				return Token{}, d.newSyntaxError("invalid type URL/extension field name: %s",
					d.orig[startPos:len(d.orig)-len(s)+1])
			}
			name = append(name, s[0])
			s = s[1:]
			s = consume(s, 0)
			for len(s) > 0 && isTypeNameChar(s[0]) {
				name = append(name, s[0])
				s = s[1:]
			}
			s = consume(s, 0)

		default:
			return Token{}, d.newSyntaxError(
				"invalid type URL/extension field name: %s", d.orig[startPos:len(d.orig)-len(s)+1])
		}
	}

	if !closed {
		return Token{}, ErrUnexpectedEOF
	}

	// First character cannot be '.'. Last character cannot be '.' or '/'.
	size := len(name)
	if size == 0 || name[0] == '.' || name[size-1] == '.' || name[size-1] == '/' {
		return Token{}, d.newSyntaxError("invalid type URL/extension field name: %s",
			d.orig[startPos:len(d.orig)-len(s)])
	}

	d.in = s
	endPos := len(d.orig) - len(d.in)
	d.consume(0)

	return Token{
		kind:  Name,
		attrs: uint8(TypeName),
		pos:   startPos,
		raw:   d.orig[startPos:endPos],
		str:   string(name),
	}, nil
}

func isTypeNameChar(b byte) bool {
	return (b == '-' || b == '_' ||
		('0' <= b && b <= '9') ||
		('a' <= b && b <= 'z') ||
		('A' <= b && b <= 'Z'))
}

func isWhiteSpace(b byte) bool {
	switch b {
	case ' ', '\n', '\r', '\t':
		return true
	default:
		return false
	}
}

// parseIdent parses an unquoted proto identifier and returns size.
// If allowNeg is true, it allows '-' to be the first character in the
// identifier. This is used when parsing literal values like -infinity, etc.
// Regular expression matches an identifier: `^[_a-zA-Z][_a-zA-Z0-9]*`
func parseIdent(input []byte, allowNeg bool) int {
	var size int

	s := input
	if len(s) == 0 {
		return 0
	}

	if allowNeg && s[0] == '-' {
		s = s[1:]
		size++
		if len(s) == 0 {
			return 0
		}
	}

	switch {
	case s[0] == '_',
		'a' <= s[0] && s[0] <= 'z',
		'A' <= s[0] && s[0] <= 'Z':
		s = s[1:]
		size++
	default:
		return 0
	}

	for len(s) > 0 && (s[0] == '_' ||
		'a' <= s[0] && s[0] <= 'z' ||
		'A' <= s[0] && s[0] <= 'Z' ||
		'0' <= s[0] && s[0] <= '9') {
		s = s[1:]
		size++
	}

	if len(s) > 0 && !isDelim(s[0]) {
		return 0
	}

	return size
}

// parseScalar parses for a string, literal or number value.
func (d *Decoder) parseScalar() (Token, error) {
	if d.in[0] == '"' || d.in[0] == '\'' {
		return d.parseStringValue()
	}

	if tok, ok := d.parseLiteralValue(); ok {
		return tok, nil
	}

	if tok, ok := d.parseNumberValue(); ok {
		return tok, nil
	}

	return Token{}, d.newSyntaxError("invalid scalar value: %s", errId(d.in))
}

// parseLiteralValue parses a literal value. A literal value is used for
// bools, special floats and enums. This function simply identifies that the
// field value is a literal.
func (d *Decoder) parseLiteralValue() (Token, bool) {
	size := parseIdent(d.in, true)
	if size == 0 {
		return Token{}, false
	}
	return d.consumeToken(Scalar, size, literalValue), true
}

// consumeToken constructs a Token for given Kind from d.in and consumes given
// size-length from it.
func (d *Decoder) consumeToken(kind Kind, size int, attrs uint8) Token {
	// Important to compute raw and pos before consuming.
	tok := Token{
		kind:  kind,
		attrs: attrs,
		pos:   len(d.orig) - len(d.in),
		raw:   d.in[:size],
	}
	d.consume(size)
	return tok
}

// newSyntaxError returns a syntax error with line and column information for
// current position.
func (d *Decoder) newSyntaxError(f string, x ...interface{}) error {
	e := errors.New(f, x...)
	line, column := d.Position(len(d.orig) - len(d.in))
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

func (d *Decoder) tryConsumeChar(c byte) bool {
	if len(d.in) > 0 && d.in[0] == c {
		d.consume(1)
		return true
	}
	return false
}

// consume consumes n bytes of input and any subsequent whitespace or comments.
func (d *Decoder) consume(n int) {
	d.in = consume(d.in, n)
	return
}

// consume consumes n bytes of input and any subsequent whitespace or comments.
func consume(b []byte, n int) []byte {
	b = b[n:]
	for len(b) > 0 {
		switch b[0] {
		case ' ', '\n', '\r', '\t':
			b = b[1:]
		case '#':
			if i := bytes.IndexByte(b, '\n'); i >= 0 {
				b = b[i+len("\n"):]
			} else {
				b = nil
			}
		default:
			return b
		}
	}
	return b
}

// errId extracts a byte sequence that looks like an invalid ID
// (for the purposes of error reporting).
func errId(seq []byte) []byte {
	const maxLen = 32
	for i := 0; i < len(seq); {
		if i > maxLen {
			return append(seq[:i:i], "…"...)
		}
		r, size := utf8.DecodeRune(seq[i:])
		if r > utf8.RuneSelf || (r != '/' && isDelim(byte(r))) {
			if i == 0 {
				// Either the first byte is invalid UTF-8 or a
				// delimiter, or the first rune is non-ASCII.
				// Return it as-is.
				i = size
			}
			return seq[:i:i]
		}
		i += size
	}
	// No delimiter found.
	return seq
}

// isDelim returns true if given byte is a delimiter character.
func isDelim(c byte) bool {
	return !(c == '-' || c == '+' || c == '.' || c == '_' ||
		('a' <= c && c <= 'z') ||
		('A' <= c && c <= 'Z') ||
		('0' <= c && c <= '9'))
}
