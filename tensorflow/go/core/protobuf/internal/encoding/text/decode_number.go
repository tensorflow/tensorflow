// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text

// parseNumberValue parses a number from the input and returns a Token object.
func (d *Decoder) parseNumberValue() (Token, bool) {
	in := d.in
	num := parseNumber(in)
	if num.size == 0 {
		return Token{}, false
	}
	numAttrs := num.kind
	if num.neg {
		numAttrs |= isNegative
	}
	tok := Token{
		kind:     Scalar,
		attrs:    numberValue,
		pos:      len(d.orig) - len(d.in),
		raw:      d.in[:num.size],
		str:      num.string(d.in),
		numAttrs: numAttrs,
	}
	d.consume(num.size)
	return tok, true
}

const (
	numDec uint8 = (1 << iota) / 2
	numHex
	numOct
	numFloat
)

// number is the result of parsing out a valid number from parseNumber. It
// contains data for doing float or integer conversion via the strconv package
// in conjunction with the input bytes.
type number struct {
	kind uint8
	neg  bool
	size int
	// if neg, this is the length of whitespace and comments between
	// the minus sign and the rest fo the number literal
	sep int
}

func (num number) string(data []byte) string {
	strSize := num.size
	last := num.size - 1
	if num.kind == numFloat && (data[last] == 'f' || data[last] == 'F') {
		strSize = last
	}
	if num.neg && num.sep > 0 {
		// strip whitespace/comments between negative sign and the rest
		strLen := strSize - num.sep
		str := make([]byte, strLen)
		str[0] = data[0]
		copy(str[1:], data[num.sep+1:strSize])
		return string(str)
	}
	return string(data[:strSize])

}

// parseNumber constructs a number object from given input. It allows for the
// following patterns:
//
//	integer: ^-?([1-9][0-9]*|0[xX][0-9a-fA-F]+|0[0-7]*)
//	float: ^-?((0|[1-9][0-9]*)?([.][0-9]*)?([eE][+-]?[0-9]+)?[fF]?)
//
// It also returns the number of parsed bytes for the given number, 0 if it is
// not a number.
func parseNumber(input []byte) number {
	kind := numDec
	var size int
	var neg bool

	s := input
	if len(s) == 0 {
		return number{}
	}

	// Optional -
	var sep int
	if s[0] == '-' {
		neg = true
		s = s[1:]
		size++
		// Consume any whitespace or comments between the
		// negative sign and the rest of the number
		lenBefore := len(s)
		s = consume(s, 0)
		sep = lenBefore - len(s)
		size += sep
		if len(s) == 0 {
			return number{}
		}
	}

	switch {
	case s[0] == '0':
		if len(s) > 1 {
			switch {
			case s[1] == 'x' || s[1] == 'X':
				// Parse as hex number.
				kind = numHex
				n := 2
				s = s[2:]
				for len(s) > 0 && (('0' <= s[0] && s[0] <= '9') ||
					('a' <= s[0] && s[0] <= 'f') ||
					('A' <= s[0] && s[0] <= 'F')) {
					s = s[1:]
					n++
				}
				if n == 2 {
					return number{}
				}
				size += n

			case '0' <= s[1] && s[1] <= '7':
				// Parse as octal number.
				kind = numOct
				n := 2
				s = s[2:]
				for len(s) > 0 && '0' <= s[0] && s[0] <= '7' {
					s = s[1:]
					n++
				}
				size += n
			}

			if kind&(numHex|numOct) > 0 {
				if len(s) > 0 && !isDelim(s[0]) {
					return number{}
				}
				return number{kind: kind, neg: neg, size: size, sep: sep}
			}
		}
		s = s[1:]
		size++

	case '1' <= s[0] && s[0] <= '9':
		n := 1
		s = s[1:]
		for len(s) > 0 && '0' <= s[0] && s[0] <= '9' {
			s = s[1:]
			n++
		}
		size += n

	case s[0] == '.':
		// Set kind to numFloat to signify the intent to parse as float. And
		// that it needs to have other digits after '.'.
		kind = numFloat

	default:
		return number{}
	}

	// . followed by 0 or more digits.
	if len(s) > 0 && s[0] == '.' {
		n := 1
		s = s[1:]
		// If decimal point was before any digits, it should be followed by
		// other digits.
		if len(s) == 0 && kind == numFloat {
			return number{}
		}
		for len(s) > 0 && '0' <= s[0] && s[0] <= '9' {
			s = s[1:]
			n++
		}
		size += n
		kind = numFloat
	}

	// e or E followed by an optional - or + and 1 or more digits.
	if len(s) >= 2 && (s[0] == 'e' || s[0] == 'E') {
		kind = numFloat
		s = s[1:]
		n := 1
		if s[0] == '+' || s[0] == '-' {
			s = s[1:]
			n++
			if len(s) == 0 {
				return number{}
			}
		}
		for len(s) > 0 && '0' <= s[0] && s[0] <= '9' {
			s = s[1:]
			n++
		}
		size += n
	}

	// Optional suffix f or F for floats.
	if len(s) > 0 && (s[0] == 'f' || s[0] == 'F') {
		kind = numFloat
		s = s[1:]
		size++
	}

	// Check that next byte is a delimiter or it is at the end.
	if len(s) > 0 && !isDelim(s[0]) {
		return number{}
	}

	return number{kind: kind, neg: neg, size: size, sep: sep}
}
