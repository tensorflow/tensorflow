// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text_test

import (
	"math"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/encoding/text"
)

// Disable detrand to enable direct comparisons on outputs.
func init() { detrand.Disable() }

func TestEncoder(t *testing.T) {
	tests := []encoderTestCase{
		{
			desc:          "no-opt",
			write:         func(e *text.Encoder) {},
			wantOut:       ``,
			wantOutIndent: ``,
		},
		{
			desc: "true",
			write: func(e *text.Encoder) {
				e.WriteName("bool")
				e.WriteBool(true)
			},
			wantOut:       `bool:true`,
			wantOutIndent: `bool: true`,
		},
		{
			desc: "false",
			write: func(e *text.Encoder) {
				e.WriteName("bool")
				e.WriteBool(false)
			},
			wantOut:       `bool:false`,
			wantOutIndent: `bool: false`,
		},
		{
			desc: "bracket name",
			write: func(e *text.Encoder) {
				e.WriteName("[extension]")
				e.WriteString("hello")
			},
			wantOut:       `[extension]:"hello"`,
			wantOutIndent: `[extension]: "hello"`,
		},
		{
			desc: "numeric name",
			write: func(e *text.Encoder) {
				e.WriteName("01234")
				e.WriteString("hello")
			},
			wantOut:       `01234:"hello"`,
			wantOutIndent: `01234: "hello"`,
		},
		{
			desc: "string",
			write: func(e *text.Encoder) {
				e.WriteName("str")
				e.WriteString("hello world")
			},
			wantOut:       `str:"hello world"`,
			wantOutIndent: `str: "hello world"`,
		},
		{
			desc: "enum",
			write: func(e *text.Encoder) {
				e.WriteName("enum")
				e.WriteLiteral("ENUM_VALUE")
			},
			wantOut:       `enum:ENUM_VALUE`,
			wantOutIndent: `enum: ENUM_VALUE`,
		},
		{
			desc: "float64",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(1.0199999809265137, 64)
			},
			wantOut:       `float64:1.0199999809265137`,
			wantOutIndent: `float64: 1.0199999809265137`,
		},
		{
			desc: "float64 max value",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(math.MaxFloat64, 64)
			},
			wantOut:       `float64:1.7976931348623157e+308`,
			wantOutIndent: `float64: 1.7976931348623157e+308`,
		},
		{
			desc: "float64 min value",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(-math.MaxFloat64, 64)
			},
			wantOut:       `float64:-1.7976931348623157e+308`,
			wantOutIndent: `float64: -1.7976931348623157e+308`,
		},
		{
			desc: "float64 nan",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(math.NaN(), 64)
			},
			wantOut:       `float64:nan`,
			wantOutIndent: `float64: nan`,
		},
		{
			desc: "float64 inf",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(math.Inf(+1), 64)
			},
			wantOut:       `float64:inf`,
			wantOutIndent: `float64: inf`,
		},
		{
			desc: "float64 -inf",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(math.Inf(-1), 64)
			},
			wantOut:       `float64:-inf`,
			wantOutIndent: `float64: -inf`,
		},
		{
			desc: "float64 negative zero",
			write: func(e *text.Encoder) {
				e.WriteName("float64")
				e.WriteFloat(math.Copysign(0, -1), 64)
			},
			wantOut:       `float64:-0`,
			wantOutIndent: `float64: -0`,
		},
		{
			desc: "float32",
			write: func(e *text.Encoder) {
				e.WriteName("float")
				e.WriteFloat(1.02, 32)
			},
			wantOut:       `float:1.02`,
			wantOutIndent: `float: 1.02`,
		},
		{
			desc: "float32 max value",
			write: func(e *text.Encoder) {
				e.WriteName("float32")
				e.WriteFloat(math.MaxFloat32, 32)
			},
			wantOut:       `float32:3.4028235e+38`,
			wantOutIndent: `float32: 3.4028235e+38`,
		},
		{
			desc: "float32 nan",
			write: func(e *text.Encoder) {
				e.WriteName("float32")
				e.WriteFloat(math.NaN(), 32)
			},
			wantOut:       `float32:nan`,
			wantOutIndent: `float32: nan`,
		},
		{
			desc: "float32 inf",
			write: func(e *text.Encoder) {
				e.WriteName("float32")
				e.WriteFloat(math.Inf(+1), 32)
			},
			wantOut:       `float32:inf`,
			wantOutIndent: `float32: inf`,
		},
		{
			desc: "float32 -inf",
			write: func(e *text.Encoder) {
				e.WriteName("float32")
				e.WriteFloat(math.Inf(-1), 32)
			},
			wantOut:       `float32:-inf`,
			wantOutIndent: `float32: -inf`,
		},
		{
			desc: "float32 negative zero",
			write: func(e *text.Encoder) {
				e.WriteName("float32")
				e.WriteFloat(math.Copysign(0, -1), 32)
			},
			wantOut:       `float32:-0`,
			wantOutIndent: `float32: -0`,
		},
		{
			desc: "int64 max value",
			write: func(e *text.Encoder) {
				e.WriteName("int")
				e.WriteInt(math.MaxInt64)
			},
			wantOut:       `int:9223372036854775807`,
			wantOutIndent: `int: 9223372036854775807`,
		},
		{
			desc: "int64 min value",
			write: func(e *text.Encoder) {
				e.WriteName("int")
				e.WriteInt(math.MinInt64)
			},
			wantOut:       `int:-9223372036854775808`,
			wantOutIndent: `int: -9223372036854775808`,
		},
		{
			desc: "uint",
			write: func(e *text.Encoder) {
				e.WriteName("uint")
				e.WriteUint(math.MaxUint64)
			},
			wantOut:       `uint:18446744073709551615`,
			wantOutIndent: `uint: 18446744073709551615`,
		},
		{
			desc: "empty message field",
			write: func(e *text.Encoder) {
				e.WriteName("m")
				e.StartMessage()
				e.EndMessage()
			},
			wantOut:       `m:{}`,
			wantOutIndent: `m: {}`,
		},
		{
			desc: "multiple fields",
			write: func(e *text.Encoder) {
				e.WriteName("bool")
				e.WriteBool(true)
				e.WriteName("str")
				e.WriteString("hello")
				e.WriteName("str")
				e.WriteString("world")
				e.WriteName("m")
				e.StartMessage()
				e.EndMessage()
				e.WriteName("[int]")
				e.WriteInt(49)
				e.WriteName("float64")
				e.WriteFloat(1.00023e4, 64)
				e.WriteName("101")
				e.WriteString("unknown")
			},
			wantOut: `bool:true str:"hello" str:"world" m:{} [int]:49 float64:10002.3 101:"unknown"`,
			wantOutIndent: `bool: true
str: "hello"
str: "world"
m: {}
[int]: 49
float64: 10002.3
101: "unknown"`,
		},
		{
			desc: "populated message fields",
			write: func(e *text.Encoder) {
				e.WriteName("m1")
				e.StartMessage()
				{
					e.WriteName("str")
					e.WriteString("hello")
				}
				e.EndMessage()

				e.WriteName("bool")
				e.WriteBool(true)

				e.WriteName("m2")
				e.StartMessage()
				{
					e.WriteName("str")
					e.WriteString("world")
					e.WriteName("m2-1")
					e.StartMessage()
					e.EndMessage()
					e.WriteName("m2-2")
					e.StartMessage()
					{
						e.WriteName("[int]")
						e.WriteInt(49)
					}
					e.EndMessage()
					e.WriteName("float64")
					e.WriteFloat(1.00023e4, 64)
				}
				e.EndMessage()

				e.WriteName("101")
				e.WriteString("unknown")
			},
			wantOut: `m1:{str:"hello"} bool:true m2:{str:"world" m2-1:{} m2-2:{[int]:49} float64:10002.3} 101:"unknown"`,
			wantOutIndent: `m1: {
	str: "hello"
}
bool: true
m2: {
	str: "world"
	m2-1: {}
	m2-2: {
		[int]: 49
	}
	float64: 10002.3
}
101: "unknown"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			runEncoderTest(t, tc, [2]byte{})

			// Test using the angle brackets.
			// Testcases should not contain characters '{' and '}'.
			tc.wantOut = replaceDelims(tc.wantOut)
			tc.wantOutIndent = replaceDelims(tc.wantOutIndent)
			runEncoderTest(t, tc, [2]byte{'<', '>'})
		})
	}
}

type encoderTestCase struct {
	desc          string
	write         func(*text.Encoder)
	wantOut       string
	wantOutIndent string
}

func runEncoderTest(t *testing.T, tc encoderTestCase, delims [2]byte) {
	t.Helper()

	if tc.wantOut != "" {
		enc, err := text.NewEncoder(nil, "", delims, false)
		if err != nil {
			t.Fatalf("NewEncoder returned error: %v", err)
		}
		tc.write(enc)
		got := string(enc.Bytes())
		if got != tc.wantOut {
			t.Errorf("(compact)\n<got>\n%v\n<want>\n%v\n", got, tc.wantOut)
		}
	}
	if tc.wantOutIndent != "" {
		enc, err := text.NewEncoder(nil, "\t", delims, false)
		if err != nil {
			t.Fatalf("NewEncoder returned error: %v", err)
		}
		tc.write(enc)
		got, want := string(enc.Bytes()), tc.wantOutIndent
		if got != want {
			t.Errorf("(multi-line)\n<got>\n%v\n<want>\n%v\n<diff -want +got>\n%v\n",
				got, want, cmp.Diff(want, got))
		}
	}
}

func replaceDelims(s string) string {
	s = strings.Replace(s, "{", "<", -1)
	return strings.Replace(s, "}", ">", -1)
}

// Test for UTF-8 and ASCII outputs.
func TestEncodeStrings(t *testing.T) {
	tests := []struct {
		in           string
		wantOut      string
		wantOutASCII string
	}{
		{
			in:      `"`,
			wantOut: `"\""`,
		},
		{
			in:      `'`,
			wantOut: `"'"`,
		},
		{
			in:           "hello\u1234world",
			wantOut:      "\"hello\u1234world\"",
			wantOutASCII: `"hello\u1234world"`,
		},
		{
			// String that has as few escaped characters as possible.
			in: func() string {
				var b []byte
				for i := rune(0); i <= 0x00a0; i++ {
					switch i {
					case 0, '\\', '\n', '\'': // these must be escaped, so ignore them
					default:
						var r [utf8.UTFMax]byte
						n := utf8.EncodeRune(r[:], i)
						b = append(b, r[:n]...)
					}
				}
				return string(b)
			}(),
			wantOut:      `"\x01\x02\x03\x04\x05\x06\x07\x08\t\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_` + "`" + `abcdefghijklmnopqrstuvwxyz{|}~\x7f\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089\u008a\u008b\u008c\u008d\u008e\u008f\u0090\u0091\u0092\u0093\u0094\u0095\u0096\u0097\u0098\u0099\u009a\u009b\u009c\u009d\u009e\u009f` + "\u00a0" + `"`,
			wantOutASCII: `"\x01\x02\x03\x04\x05\x06\x07\x08\t\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_` + "`" + `abcdefghijklmnopqrstuvwxyz{|}~\x7f\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089\u008a\u008b\u008c\u008d\u008e\u008f\u0090\u0091\u0092\u0093\u0094\u0095\u0096\u0097\u0098\u0099\u009a\u009b\u009c\u009d\u009e\u009f\u00a0"`,
		},
		{
			// Valid UTF-8 wire encoding of the RuneError rune.
			in:           string(utf8.RuneError),
			wantOut:      `"` + string(utf8.RuneError) + `"`,
			wantOutASCII: `"\ufffd"`,
		},
		{
			in:           "\"'\\?\a\b\n\r\t\v\f\x01\nS\n\xab\x12\uab8f\U0010ffff",
			wantOut:      `"\"'\\?\x07\x08\n\r\t\x0b\x0c\x01\nS\n\xab\x12` + "\uab8f\U0010ffff" + `"`,
			wantOutASCII: `"\"'\\?\x07\x08\n\r\t\x0b\x0c\x01\nS\n\xab\x12\uab8f\U0010ffff"`,
		},
		{
			in:           "\001x",
			wantOut:      `"\x01x"`,
			wantOutASCII: `"\x01x"`,
		},
		{
			in:           "\012x",
			wantOut:      `"\nx"`,
			wantOutASCII: `"\nx"`,
		},
		{
			in:           "\123x",
			wantOut:      `"Sx"`,
			wantOutASCII: `"Sx"`,
		},
		{
			in:           "\1234x",
			wantOut:      `"S4x"`,
			wantOutASCII: `"S4x"`,
		},
		{
			in:           "\001",
			wantOut:      `"\x01"`,
			wantOutASCII: `"\x01"`,
		},
		{
			in:           "\012",
			wantOut:      `"\n"`,
			wantOutASCII: `"\n"`,
		},
		{
			in:           "\123",
			wantOut:      `"S"`,
			wantOutASCII: `"S"`,
		},
		{
			in:           "\1234",
			wantOut:      `"S4"`,
			wantOutASCII: `"S4"`,
		},
		{
			in:           "\377",
			wantOut:      `"\xff"`,
			wantOutASCII: `"\xff"`,
		},
		{
			in:           "\x0fx",
			wantOut:      `"\x0fx"`,
			wantOutASCII: `"\x0fx"`,
		},
		{
			in:           "\xffx",
			wantOut:      `"\xffx"`,
			wantOutASCII: `"\xffx"`,
		},
		{
			in:           "\xfffx",
			wantOut:      `"\xfffx"`,
			wantOutASCII: `"\xfffx"`,
		},
		{
			in:           "\x0f",
			wantOut:      `"\x0f"`,
			wantOutASCII: `"\x0f"`,
		},
		{
			in:           "\x7f",
			wantOut:      `"\x7f"`,
			wantOutASCII: `"\x7f"`,
		},
		{
			in:           "\xff",
			wantOut:      `"\xff"`,
			wantOutASCII: `"\xff"`,
		},
		{
			in:           "\xfff",
			wantOut:      `"\xfff"`,
			wantOutASCII: `"\xfff"`,
		},
	}
	for _, tc := range tests {
		t.Run("", func(t *testing.T) {
			if tc.wantOut != "" {
				runEncodeStringsTest(t, tc.in, tc.wantOut, false)
			}
			if tc.wantOutASCII != "" {
				runEncodeStringsTest(t, tc.in, tc.wantOutASCII, true)
			}
		})
	}
}

func runEncodeStringsTest(t *testing.T, in string, want string, outputASCII bool) {
	t.Helper()

	charType := "UTF-8"
	if outputASCII {
		charType = "ASCII"
	}

	enc, err := text.NewEncoder(nil, "", [2]byte{}, outputASCII)
	if err != nil {
		t.Fatalf("[%s] NewEncoder returned error: %v", charType, err)
	}
	enc.WriteString(in)
	got := string(enc.Bytes())
	if got != want {
		t.Errorf("[%s] WriteString(%q)\n<got>\n%v\n<want>\n%v\n", charType, in, got, want)
	}
}

func TestReset(t *testing.T) {
	enc, err := text.NewEncoder(nil, "\t", [2]byte{}, false)
	if err != nil {
		t.Fatalf("NewEncoder returned error: %v", err)
	}

	enc.WriteName("foo")
	pos := enc.Snapshot()

	// Attempt to write a message value.
	enc.StartMessage()
	enc.WriteName("bar")
	enc.WriteUint(10)

	// Reset the value and decided to write a string value instead.
	enc.Reset(pos)
	enc.WriteString("0123456789")

	got := string(enc.Bytes())
	want := `foo: "0123456789"`
	if got != want {
		t.Errorf("Reset did not restore given position:\n<got>\n%v\n<want>\n%v\n", got, want)
	}
}
