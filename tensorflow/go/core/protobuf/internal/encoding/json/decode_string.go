// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"strconv"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"

	"google.golang.org/protobuf/internal/strs"
)

func (d *Decoder) parseString(in []byte) (string, int, error) {
	in0 := in
	if len(in) == 0 {
		return "", 0, ErrUnexpectedEOF
	}
	if in[0] != '"' {
		return "", 0, d.newSyntaxError(d.currPos(), "invalid character %q at start of string", in[0])
	}
	in = in[1:]
	i := indexNeedEscapeInBytes(in)
	in, out := in[i:], in[:i:i] // set cap to prevent mutations
	for len(in) > 0 {
		switch r, n := utf8.DecodeRune(in); {
		case r == utf8.RuneError && n == 1:
			return "", 0, d.newSyntaxError(d.currPos(), "invalid UTF-8 in string")
		case r < ' ':
			return "", 0, d.newSyntaxError(d.currPos(), "invalid character %q in string", r)
		case r == '"':
			in = in[1:]
			n := len(in0) - len(in)
			return string(out), n, nil
		case r == '\\':
			if len(in) < 2 {
				return "", 0, ErrUnexpectedEOF
			}
			switch r := in[1]; r {
			case '"', '\\', '/':
				in, out = in[2:], append(out, r)
			case 'b':
				in, out = in[2:], append(out, '\b')
			case 'f':
				in, out = in[2:], append(out, '\f')
			case 'n':
				in, out = in[2:], append(out, '\n')
			case 'r':
				in, out = in[2:], append(out, '\r')
			case 't':
				in, out = in[2:], append(out, '\t')
			case 'u':
				if len(in) < 6 {
					return "", 0, ErrUnexpectedEOF
				}
				v, err := strconv.ParseUint(string(in[2:6]), 16, 16)
				if err != nil {
					return "", 0, d.newSyntaxError(d.currPos(), "invalid escape code %q in string", in[:6])
				}
				in = in[6:]

				r := rune(v)
				if utf16.IsSurrogate(r) {
					if len(in) < 6 {
						return "", 0, ErrUnexpectedEOF
					}
					v, err := strconv.ParseUint(string(in[2:6]), 16, 16)
					r = utf16.DecodeRune(r, rune(v))
					if in[0] != '\\' || in[1] != 'u' ||
						r == unicode.ReplacementChar || err != nil {
						return "", 0, d.newSyntaxError(d.currPos(), "invalid escape code %q in string", in[:6])
					}
					in = in[6:]
				}
				out = append(out, string(r)...)
			default:
				return "", 0, d.newSyntaxError(d.currPos(), "invalid escape code %q in string", in[:2])
			}
		default:
			i := indexNeedEscapeInBytes(in[n:])
			in, out = in[n+i:], append(out, in[:n+i]...)
		}
	}
	return "", 0, ErrUnexpectedEOF
}

// indexNeedEscapeInBytes returns the index of the character that needs
// escaping. If no characters need escaping, this returns the input length.
func indexNeedEscapeInBytes(b []byte) int { return indexNeedEscapeInString(strs.UnsafeString(b)) }
