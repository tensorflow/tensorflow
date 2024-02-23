// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json_test

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/encoding/json"
)

type R struct {
	// E is expected error substring from calling Decoder.Read if set.
	E string
	// V is one of the checker implementations that validates the token value.
	V checker
	// P is expected Token.Pos() if set > 0.
	P int
	// RS is expected result from Token.RawString() if not empty.
	RS string
}

// checker defines API for Token validation.
type checker interface {
	// check checks and expects for token API call to return and compare
	// against implementation-stored value. Returns empty string if success,
	// else returns error message describing the error.
	check(json.Token) string
}

// checkers that checks the token kind only.
var (
	EOF         = kindOnly{json.EOF}
	Null        = kindOnly{json.Null}
	ObjectOpen  = kindOnly{json.ObjectOpen}
	ObjectClose = kindOnly{json.ObjectClose}
	ArrayOpen   = kindOnly{json.ArrayOpen}
	ArrayClose  = kindOnly{json.ArrayClose}
)

type kindOnly struct {
	want json.Kind
}

func (x kindOnly) check(tok json.Token) string {
	if got := tok.Kind(); got != x.want {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, x.want)
	}
	return ""
}

type Name struct {
	val string
}

func (x Name) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Name {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Name)
	}

	if got := tok.Name(); got != x.val {
		return fmt.Sprintf("Token.Name(): got %v, want %v", got, x.val)
	}
	return ""
}

type Bool struct {
	val bool
}

func (x Bool) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Bool {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Bool)
	}

	if got := tok.Bool(); got != x.val {
		return fmt.Sprintf("Token.Bool(): got %v, want %v", got, x.val)
	}
	return ""
}

type Str struct {
	val string
}

func (x Str) check(tok json.Token) string {
	if got := tok.Kind(); got != json.String {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.String)
	}

	if got := tok.ParsedString(); got != x.val {
		return fmt.Sprintf("Token.ParsedString(): got %v, want %v", got, x.val)
	}
	return ""
}

type F64 struct {
	val float64
}

func (x F64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Float(64)
	if !ok {
		return fmt.Sprintf("Token.Float(64): returned not ok")
	}
	if got != x.val {
		return fmt.Sprintf("Token.Float(64): got %v, want %v", got, x.val)
	}
	return ""
}

type F32 struct {
	val float32
}

func (x F32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Float(32)
	if !ok {
		return fmt.Sprintf("Token.Float(32): returned not ok")
	}
	if float32(got) != x.val {
		return fmt.Sprintf("Token.Float(32): got %v, want %v", got, x.val)
	}
	return ""
}

// NotF64 is a checker to validate a Number token where Token.Float(64) returns not ok.
var NotF64 = xf64{}

type xf64 struct{}

func (x xf64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Float(64)
	if ok {
		return fmt.Sprintf("Token.Float(64): returned ok")
	}
	return ""
}

// NotF32 is a checker to validate a Number token where Token.Float(32) returns not ok.
var NotF32 = xf32{}

type xf32 struct{}

func (x xf32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Float(32)
	if ok {
		return fmt.Sprintf("Token.Float(32): returned ok")
	}
	return ""
}

type I64 struct {
	val int64
}

func (x I64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Int(64)
	if !ok {
		return fmt.Sprintf("Token.Int(64): returned not ok")
	}
	if got != x.val {
		return fmt.Sprintf("Token.Int(64): got %v, want %v", got, x.val)
	}
	return ""
}

type I32 struct {
	val int32
}

func (x I32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Int(32)
	if !ok {
		return fmt.Sprintf("Token.Int(32): returned not ok")
	}
	if int32(got) != x.val {
		return fmt.Sprintf("Token.Int(32): got %v, want %v", got, x.val)
	}
	return ""
}

// NotI64 is a checker to validate a Number token where Token.Int(64) returns not ok.
var NotI64 = xi64{}

type xi64 struct{}

func (x xi64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Int(64)
	if ok {
		return fmt.Sprintf("Token.Int(64): returned ok")
	}
	return ""
}

// NotI32 is a checker to validate a Number token where Token.Int(32) returns not ok.
var NotI32 = xi32{}

type xi32 struct{}

func (x xi32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Int(32)
	if ok {
		return fmt.Sprintf("Token.Int(32): returned ok")
	}
	return ""
}

type Ui64 struct {
	val uint64
}

func (x Ui64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Uint(64)
	if !ok {
		return fmt.Sprintf("Token.Uint(64): returned not ok")
	}
	if got != x.val {
		return fmt.Sprintf("Token.Uint(64): got %v, want %v", got, x.val)
	}
	return ""
}

type Ui32 struct {
	val uint32
}

func (x Ui32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	got, ok := tok.Uint(32)
	if !ok {
		return fmt.Sprintf("Token.Uint(32): returned not ok")
	}
	if uint32(got) != x.val {
		return fmt.Sprintf("Token.Uint(32): got %v, want %v", got, x.val)
	}
	return ""
}

// NotUi64 is a checker to validate a Number token where Token.Uint(64) returns not ok.
var NotUi64 = xui64{}

type xui64 struct{}

func (x xui64) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Uint(64)
	if ok {
		return fmt.Sprintf("Token.Uint(64): returned ok")
	}
	return ""
}

// NotI32 is a checker to validate a Number token where Token.Uint(32) returns not ok.
var NotUi32 = xui32{}

type xui32 struct{}

func (x xui32) check(tok json.Token) string {
	if got := tok.Kind(); got != json.Number {
		return fmt.Sprintf("Token.Kind(): got %v, want %v", got, json.Number)
	}

	_, ok := tok.Uint(32)
	if ok {
		return fmt.Sprintf("Token.Uint(32): returned ok")
	}
	return ""
}

var errEOF = json.ErrUnexpectedEOF.Error()

func TestDecoder(t *testing.T) {
	const space = " \n\r\t"

	tests := []struct {
		in string
		// want is a list of expected values returned from calling
		// Decoder.Read. An item makes the test code invoke
		// Decoder.Read and compare against R.E for error returned or use R.V to
		// validate the returned Token object.
		want []R
	}{
		{
			in:   ``,
			want: []R{{V: EOF}},
		},
		{
			in:   space,
			want: []R{{V: EOF}},
		},
		{
			// Calling Read after EOF will keep returning EOF for
			// succeeding Read calls.
			in: space,
			want: []R{
				{V: EOF},
				{V: EOF},
				{V: EOF},
			},
		},

		// JSON literals.
		{
			in: space + `null` + space,
			want: []R{
				{V: Null, P: len(space), RS: `null`},
				{V: EOF},
			},
		},
		{
			in: space + `true` + space,
			want: []R{
				{V: Bool{true}},
				{V: EOF},
			},
		},
		{
			in: space + `false` + space,
			want: []R{
				{V: Bool{false}},
				{V: EOF},
			},
		},
		{
			// Error returned will produce the same error again.
			in: space + `foo` + space,
			want: []R{
				{E: `invalid value foo`},
				{E: `invalid value foo`},
			},
		},

		// JSON strings.
		{
			in: space + `""` + space,
			want: []R{
				{V: Str{}},
				{V: EOF},
			},
		},
		{
			in: space + `"hello"` + space,
			want: []R{
				{V: Str{"hello"}, RS: `"hello"`},
				{V: EOF},
			},
		},
		{
			in:   `"hello`,
			want: []R{{E: errEOF}},
		},
		{
			in:   "\"\x00\"",
			want: []R{{E: `invalid character '\x00' in string`}},
		},
		{
			in: "\"\u0031\u0032\"",
			want: []R{
				{V: Str{"12"}, RS: "\"\u0031\u0032\""},
				{V: EOF},
			},
		},
		{
			// Invalid UTF-8 error is returned in ReadString instead of Read.
			in:   "\"\xff\"",
			want: []R{{E: `syntax error (line 1:1): invalid UTF-8 in string`}},
		},
		{
			in: `"` + string(utf8.RuneError) + `"`,
			want: []R{
				{V: Str{string(utf8.RuneError)}},
				{V: EOF},
			},
		},
		{
			in: `"\uFFFD"`,
			want: []R{
				{V: Str{string(utf8.RuneError)}},
				{V: EOF},
			},
		},
		{
			in:   `"\x"`,
			want: []R{{E: `invalid escape code "\\x" in string`}},
		},
		{
			in:   `"\uXXXX"`,
			want: []R{{E: `invalid escape code "\\uXXXX" in string`}},
		},
		{
			in:   `"\uDEAD"`, // unmatched surrogate pair
			want: []R{{E: errEOF}},
		},
		{
			in:   `"\uDEAD\uBEEF"`, // invalid surrogate half
			want: []R{{E: `invalid escape code "\\uBEEF" in string`}},
		},
		{
			in: `"\uD800\udead"`, // valid surrogate pair
			want: []R{
				{V: Str{`𐊭`}},
				{V: EOF},
			},
		},
		{
			in: `"\u0000\"\\\/\b\f\n\r\t"`,
			want: []R{
				{V: Str{"\u0000\"\\/\b\f\n\r\t"}},
				{V: EOF},
			},
		},

		// Invalid JSON numbers.
		{
			in:   `-`,
			want: []R{{E: `invalid value -`}},
		},
		{
			in:   `+0`,
			want: []R{{E: `invalid value +0`}},
		},
		{
			in:   `-+`,
			want: []R{{E: `invalid value -+`}},
		},
		{
			in:   `0.`,
			want: []R{{E: `invalid value 0.`}},
		},
		{
			in:   `.1`,
			want: []R{{E: `invalid value .1`}},
		},
		{
			in:   `1.0.1`,
			want: []R{{E: `invalid value 1.0.1`}},
		},
		{
			in:   `1..1`,
			want: []R{{E: `invalid value 1..1`}},
		},
		{
			in:   `-1-2`,
			want: []R{{E: `invalid value -1-2`}},
		},
		{
			in:   `01`,
			want: []R{{E: `invalid value 01`}},
		},
		{
			in:   `1e`,
			want: []R{{E: `invalid value 1e`}},
		},
		{
			in:   `1e1.2`,
			want: []R{{E: `invalid value 1e1.2`}},
		},
		{
			in:   `1Ee`,
			want: []R{{E: `invalid value 1Ee`}},
		},
		{
			in:   `1.e1`,
			want: []R{{E: `invalid value 1.e1`}},
		},
		{
			in:   `1.e+`,
			want: []R{{E: `invalid value 1.e+`}},
		},
		{
			in:   `1e+-2`,
			want: []R{{E: `invalid value 1e+-2`}},
		},
		{
			in:   `1e--2`,
			want: []R{{E: `invalid value 1e--2`}},
		},
		{
			in:   `1.0true`,
			want: []R{{E: `invalid value 1.0true`}},
		},

		// JSON numbers as floating point.
		{
			in: space + `0.0` + space,
			want: []R{
				{V: F32{0}, P: len(space), RS: `0.0`},
				{V: EOF},
			},
		},
		{
			in: space + `0` + space,
			want: []R{
				{V: F32{0}},
				{V: EOF},
			},
		},
		{
			in: space + `-0` + space,
			want: []R{
				{V: F32{float32(math.Copysign(0, -1))}},
				{V: EOF},
			},
		},
		{
			in: `-0`,
			want: []R{
				{V: F64{math.Copysign(0, -1)}},
				{V: EOF},
			},
		},
		{
			in: `-0.0`,
			want: []R{
				{V: F32{float32(math.Copysign(0, -1))}},
				{V: EOF},
			},
		},
		{
			in: `-0.0`,
			want: []R{
				{V: F64{math.Copysign(0, -1)}},
				{V: EOF},
			},
		},
		{
			in: `-1.02`,
			want: []R{
				{V: F32{-1.02}},
				{V: EOF},
			},
		},
		{
			in: `1.020000`,
			want: []R{
				{V: F32{1.02}},
				{V: EOF},
			},
		},
		{
			in: `-1.0e0`,
			want: []R{
				{V: F32{-1}},
				{V: EOF},
			},
		},
		{
			in: `1.0e-000`,
			want: []R{
				{V: F32{1}},
				{V: EOF},
			},
		},
		{
			in: `1e+00`,
			want: []R{
				{V: F32{1}},
				{V: EOF},
			},
		},
		{
			in: `1.02e3`,
			want: []R{
				{V: F32{1.02e3}},
				{V: EOF},
			},
		},
		{
			in: `-1.02E03`,
			want: []R{
				{V: F32{-1.02e3}},
				{V: EOF},
			},
		},
		{
			in: `1.0200e+3`,
			want: []R{
				{V: F32{1.02e3}},
				{V: EOF},
			},
		},
		{
			in: `-1.0200E+03`,
			want: []R{
				{V: F32{-1.02e3}},
				{V: EOF},
			},
		},
		{
			in: `1.0200e-3`,
			want: []R{
				{V: F32{1.02e-3}},
				{V: EOF},
			},
		},
		{
			in: `-1.0200E-03`,
			want: []R{
				{V: F32{-1.02e-3}},
				{V: EOF},
			},
		},
		{
			// Exceeds max float32 limit, but should be ok for float64.
			in: `3.4e39`,
			want: []R{
				{V: F64{3.4e39}},
				{V: EOF},
			},
		},

		{
			// Exceeds max float32 limit.
			in: `3.4e39`,
			want: []R{
				{V: NotF32},
				{V: EOF},
			},
		},
		{
			// Less than negative max float32 limit.
			in: `-3.4e39`,
			want: []R{
				{V: NotF32},
				{V: EOF},
			},
		},
		{
			// Exceeds max float64 limit.
			in: `1.79e+309`,
			want: []R{
				{V: NotF64},
				{V: EOF},
			},
		},
		{
			// Less than negative max float64 limit.
			in: `-1.79e+309`,
			want: []R{
				{V: NotF64},
				{V: EOF},
			},
		},

		// JSON numbers as signed integers.
		{
			in: space + `0` + space,
			want: []R{
				{V: I32{0}},
				{V: EOF},
			},
		},
		{
			in: space + `-0` + space,
			want: []R{
				{V: I32{0}},
				{V: EOF},
			},
		},
		{
			// Fractional part equals 0 is ok.
			in: `1.00000`,
			want: []R{
				{V: I32{1}},
				{V: EOF},
			},
		},
		{
			// Fractional part not equals 0 returns error.
			in: `1.0000000001`,
			want: []R{
				{V: NotI32},
				{V: EOF},
			},
		},
		{
			in: `0e0`,
			want: []R{
				{V: I32{0}},
				{V: EOF},
			},
		},
		{
			in: `0.0E0`,
			want: []R{
				{V: I32{0}},
				{V: EOF},
			},
		},
		{
			in: `0.0E10`,
			want: []R{
				{V: I32{0}},
				{V: EOF},
			},
		},
		{
			in: `-1`,
			want: []R{
				{V: I32{-1}},
				{V: EOF},
			},
		},
		{
			in: `1.0e+0`,
			want: []R{
				{V: I32{1}},
				{V: EOF},
			},
		},
		{
			in: `-1E-0`,
			want: []R{
				{V: I32{-1}},
				{V: EOF},
			},
		},
		{
			in: `1E1`,
			want: []R{
				{V: I32{10}},
				{V: EOF},
			},
		},
		{
			in: `-100.00e-02`,
			want: []R{
				{V: I32{-1}},
				{V: EOF},
			},
		},
		{
			in: `0.1200E+02`,
			want: []R{
				{V: I64{12}},
				{V: EOF},
			},
		},
		{
			in: `0.012e2`,
			want: []R{
				{V: NotI32},
				{V: EOF},
			},
		},
		{
			in: `12e-2`,
			want: []R{
				{V: NotI32},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MaxInt32.
			in: `2147483648`,
			want: []R{
				{V: NotI32},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MinInt32.
			in: `-2147483649`,
			want: []R{
				{V: NotI32},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MaxInt32, but ok for int64.
			in: `2147483648`,
			want: []R{
				{V: I64{2147483648}},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MinInt32, but ok for int64.
			in: `-2147483649`,
			want: []R{
				{V: I64{-2147483649}},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MaxInt64.
			in: `9223372036854775808`,
			want: []R{
				{V: NotI64},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MinInt64.
			in: `-9223372036854775809`,
			want: []R{
				{V: NotI64},
				{V: EOF},
			},
		},

		// JSON numbers as unsigned integers.
		{
			in: space + `0` + space,
			want: []R{
				{V: Ui32{0}},
				{V: EOF},
			},
		},
		{
			in: space + `-0` + space,
			want: []R{
				{V: Ui32{0}},
				{V: EOF},
			},
		},
		{
			in: `-1`,
			want: []R{
				{V: NotUi32},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MaxUint32.
			in: `4294967296`,
			want: []R{
				{V: NotUi32},
				{V: EOF},
			},
		},
		{
			// Exceeds math.MaxUint64.
			in: `18446744073709551616`,
			want: []R{
				{V: NotUi64},
				{V: EOF},
			},
		},

		// JSON sequence of values.
		{
			in: `true null`,
			want: []R{
				{V: Bool{true}},
				{E: `(line 1:6): unexpected token null`},
			},
		},
		{
			in: "null false",
			want: []R{
				{V: Null},
				{E: `unexpected token false`},
			},
		},
		{
			in: `true,false`,
			want: []R{
				{V: Bool{true}},
				{E: `unexpected token ,`},
			},
		},
		{
			in: `47"hello"`,
			want: []R{
				{V: I32{47}},
				{E: `unexpected token "hello"`},
			},
		},
		{
			in: `47 "hello"`,
			want: []R{
				{V: I32{47}},
				{E: `unexpected token "hello"`},
			},
		},
		{
			in: `true 42`,
			want: []R{
				{V: Bool{true}},
				{E: `unexpected token 42`},
			},
		},

		// JSON arrays.
		{
			in: space + `[]` + space,
			want: []R{
				{V: ArrayOpen},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: space + `[` + space + `]` + space,
			want: []R{
				{V: ArrayOpen, P: len(space), RS: `[`},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: space + `[` + space,
			want: []R{
				{V: ArrayOpen},
				{E: errEOF},
			},
		},
		{
			in:   space + `]` + space,
			want: []R{{E: `unexpected token ]`}},
		},
		{
			in: `[null,true,false,  1e1, "hello"   ]`,
			want: []R{
				{V: ArrayOpen},
				{V: Null},
				{V: Bool{true}},
				{V: Bool{false}},
				{V: I32{10}},
				{V: Str{"hello"}},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: `[` + space + `true` + space + `,` + space + `"hello"` + space + `]`,
			want: []R{
				{V: ArrayOpen},
				{V: Bool{true}},
				{V: Str{"hello"}},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: `[` + space + `true` + space + `,` + space + `]`,
			want: []R{
				{V: ArrayOpen},
				{V: Bool{true}},
				{E: `unexpected token ]`},
			},
		},
		{
			in: `[` + space + `false` + space + `]`,
			want: []R{
				{V: ArrayOpen},
				{V: Bool{false}},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: `[` + space + `1` + space + `0` + space + `]`,
			want: []R{
				{V: ArrayOpen},
				{V: I64{1}},
				{E: `unexpected token 0`},
			},
		},
		{
			in: `[null`,
			want: []R{
				{V: ArrayOpen},
				{V: Null},
				{E: errEOF},
			},
		},
		{
			in: `[foo]`,
			want: []R{
				{V: ArrayOpen},
				{E: `invalid value foo`},
			},
		},
		{
			in: `[{}, "hello", [true, false], null]`,
			want: []R{
				{V: ArrayOpen},
				{V: ObjectOpen},
				{V: ObjectClose},
				{V: Str{"hello"}},
				{V: ArrayOpen},
				{V: Bool{true}},
				{V: Bool{false}},
				{V: ArrayClose},
				{V: Null},
				{V: ArrayClose},
				{V: EOF},
			},
		},
		{
			in: `[{ ]`,
			want: []R{
				{V: ArrayOpen},
				{V: ObjectOpen},
				{E: `unexpected token ]`},
			},
		},
		{
			in: `[[ ]`,
			want: []R{
				{V: ArrayOpen},
				{V: ArrayOpen},
				{V: ArrayClose},
				{E: errEOF},
			},
		},
		{
			in: `[,]`,
			want: []R{
				{V: ArrayOpen},
				{E: `unexpected token ,`},
			},
		},
		{
			in: `[true "hello"]`,
			want: []R{
				{V: ArrayOpen},
				{V: Bool{true}},
				{E: `unexpected token "hello"`},
			},
		},
		{
			in: `[] null`,
			want: []R{
				{V: ArrayOpen},
				{V: ArrayClose},
				{E: `unexpected token null`},
			},
		},
		{
			in: `true []`,
			want: []R{
				{V: Bool{true}},
				{E: `unexpected token [`},
			},
		},

		// JSON objects.
		{
			in: space + `{}` + space,
			want: []R{
				{V: ObjectOpen},
				{V: ObjectClose},
				{V: EOF},
			},
		},
		{
			in: space + `{` + space + `}` + space,
			want: []R{
				{V: ObjectOpen},
				{V: ObjectClose},
				{V: EOF},
			},
		},
		{
			in: space + `{` + space,
			want: []R{
				{V: ObjectOpen},
				{E: errEOF},
			},
		},
		{
			in:   space + `}` + space,
			want: []R{{E: `unexpected token }`}},
		},
		{
			in: `{` + space + `null` + space + `}`,
			want: []R{
				{V: ObjectOpen},
				{E: `unexpected token null`},
			},
		},
		{
			in: `{[]}`,
			want: []R{
				{V: ObjectOpen},
				{E: `(line 1:2): unexpected token [`},
			},
		},
		{
			in: `{,}`,
			want: []R{
				{V: ObjectOpen},
				{E: `unexpected token ,`},
			},
		},
		{
			in: `{"345678"}`,
			want: []R{
				{V: ObjectOpen},
				{E: `(line 1:10): unexpected character }, missing ":" after field name`},
			},
		},
		{
			in: `{` + space + `"hello"` + space + `:` + space + `"world"` + space + `}`,
			want: []R{
				{V: ObjectOpen},
				{V: Name{"hello"}, P: len(space) + 1, RS: `"hello"`},
				{V: Str{"world"}, RS: `"world"`},
				{V: ObjectClose},
				{V: EOF},
			},
		},
		{
			in: `{"hello" "world"}`,
			want: []R{
				{V: ObjectOpen},
				{E: `(line 1:10): unexpected character ", missing ":" after field name`},
			},
		},
		{
			in: `{"hello":`,
			want: []R{
				{V: ObjectOpen},
				{V: Name{"hello"}},
				{E: errEOF},
			},
		},
		{
			in: `{"hello":"world"`,
			want: []R{
				{V: ObjectOpen},
				{V: Name{"hello"}},
				{V: Str{"world"}},
				{E: errEOF},
			},
		},
		{
			in: `{"hello":"world",`,
			want: []R{
				{V: ObjectOpen},
				{V: Name{"hello"}},
				{V: Str{"world"}},
				{E: errEOF},
			},
		},
		{
			in: `{""`,
			want: []R{
				{V: ObjectOpen},
				{E: errEOF},
			},
		},
		{
			in: `{"34":"89",}`,
			want: []R{
				{V: ObjectOpen},
				{V: Name{"34"}, RS: `"34"`},
				{V: Str{"89"}},
				{E: `syntax error (line 1:12): unexpected token }`},
			},
		},
		{
			in: `{
			  "number": 123e2,
			  "bool"  : false,
			  "object": {"string": "world"},
			  "null"  : null,
			  "array" : [1.01, "hello", true],
			  "string": "hello"
			}`,
			want: []R{
				{V: ObjectOpen},

				{V: Name{"number"}},
				{V: I32{12300}},

				{V: Name{"bool"}},
				{V: Bool{false}},

				{V: Name{"object"}},
				{V: ObjectOpen},
				{V: Name{"string"}},
				{V: Str{"world"}},
				{V: ObjectClose},

				{V: Name{"null"}},
				{V: Null},

				{V: Name{"array"}},
				{V: ArrayOpen},
				{V: F32{1.01}},
				{V: Str{"hello"}},
				{V: Bool{true}},
				{V: ArrayClose},

				{V: Name{"string"}},
				{V: Str{"hello"}},

				{V: ObjectClose},
				{V: EOF},
			},
		},
		{
			in: `[
			  {"object": {"number": 47}},
			  ["list"],
			  null
			]`,
			want: []R{
				{V: ArrayOpen},

				{V: ObjectOpen},
				{V: Name{"object"}},
				{V: ObjectOpen},
				{V: Name{"number"}},
				{V: I32{47}},
				{V: ObjectClose},
				{V: ObjectClose},

				{V: ArrayOpen},
				{V: Str{"list"}},
				{V: ArrayClose},

				{V: Null},

				{V: ArrayClose},
				{V: EOF},
			},
		},

		// Tests for line and column info.
		{
			in: `12345678 x`,
			want: []R{
				{V: I64{12345678}},
				{E: `syntax error (line 1:10): invalid value x`},
			},
		},
		{
			in: "\ntrue\n   x",
			want: []R{
				{V: Bool{true}},
				{E: `syntax error (line 3:4): invalid value x`},
			},
		},
		{
			in: `"💩"x`,
			want: []R{
				{V: Str{"💩"}},
				{E: `syntax error (line 1:4): invalid value x`},
			},
		},
		{
			in: "\n\n[\"🔥🔥🔥\"x",
			want: []R{
				{V: ArrayOpen},
				{V: Str{"🔥🔥🔥"}},
				{E: `syntax error (line 3:7): invalid value x`},
			},
		},
		{
			// Multi-rune emojis.
			in: `["👍🏻👍🏿"x`,
			want: []R{
				{V: ArrayOpen},
				{V: Str{"👍🏻👍🏿"}},
				{E: `syntax error (line 1:8): invalid value x`},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run("", func(t *testing.T) {
			dec := json.NewDecoder([]byte(tc.in))
			for i, want := range tc.want {
				peekTok, peekErr := dec.Peek()
				tok, err := dec.Read()
				if err != nil {
					if want.E == "" {
						errorf(t, tc.in, "want#%d: Read() got unexpected error: %v", i, err)
					} else if !strings.Contains(err.Error(), want.E) {
						errorf(t, tc.in, "want#%d: Read() got %q, want %q", i, err, want.E)
					}
					return
				}
				if want.E != "" {
					errorf(t, tc.in, "want#%d: Read() got nil error, want %q", i, want.E)
					return
				}
				checkToken(t, tok, i, want, tc.in)
				if !cmp.Equal(tok, peekTok, cmp.Comparer(json.TokenEquals)) {
					errorf(t, tc.in, "want#%d: Peek() %+v != Read() token %+v", i, peekTok, tok)
				}
				if err != peekErr {
					errorf(t, tc.in, "want#%d: Peek() error %v != Read() error %v", i, err, peekErr)
				}
			}
		})
	}
}

func checkToken(t *testing.T, tok json.Token, idx int, r R, in string) {
	// Validate Token.Pos() if R.P is set.
	if r.P > 0 {
		got := tok.Pos()
		if got != r.P {
			errorf(t, in, "want#%d: Token.Pos() got %v want %v", idx, got, r.P)
		}
	}
	// Validate Token.RawString if R.RS is set.
	if len(r.RS) > 0 {
		got := tok.RawString()
		if got != r.RS {
			errorf(t, in, "want#%d: Token.RawString() got %v want %v", idx, got, r.P)
		}
	}

	// Skip checking for Token details if r.V is not set.
	if r.V == nil {
		return
	}

	if err := r.V.check(tok); err != "" {
		errorf(t, in, "want#%d: %s", idx, err)
	}
	return
}

func errorf(t *testing.T, in string, fmtStr string, args ...interface{}) {
	t.Helper()
	vargs := []interface{}{in}
	for _, arg := range args {
		vargs = append(vargs, arg)
	}
	t.Errorf("input:\n%s\n~end~\n"+fmtStr, vargs...)
}

func TestClone(t *testing.T) {
	input := `{"outer":{"str":"hello", "number": 123}}`
	dec := json.NewDecoder([]byte(input))

	// Clone at the start should produce the same reads as the original.
	clone := dec.Clone()
	compareDecoders(t, dec, clone)

	// Advance to inner object, clone and compare again.
	dec.Read() // Read ObjectOpen.
	dec.Read() // Read Name.
	clone = dec.Clone()
	compareDecoders(t, dec, clone)
}

func compareDecoders(t *testing.T, d1 *json.Decoder, d2 *json.Decoder) {
	for {
		tok1, err1 := d1.Read()
		tok2, err2 := d2.Read()
		if tok1.Kind() != tok2.Kind() {
			t.Errorf("cloned decoder: got Kind %v, want %v", tok2.Kind(), tok1.Kind())
		}
		if tok1.RawString() != tok2.RawString() {
			t.Errorf("cloned decoder: got RawString %v, want %v", tok2.RawString(), tok1.RawString())
		}
		if err1 != err2 {
			t.Errorf("cloned decoder: got error %v, want %v", err2, err1)
		}
		if tok1.Kind() == json.EOF {
			break
		}
	}
}
