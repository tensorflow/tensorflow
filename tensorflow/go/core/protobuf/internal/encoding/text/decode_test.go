// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text_test

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/encoding/text"
	"google.golang.org/protobuf/internal/flags"
)

var eofErr = text.ErrUnexpectedEOF.Error()

type R struct {
	// K is expected Kind of the returned Token object from calling Decoder.Read.
	K text.Kind
	// E is expected error substring from calling Decoder.Read if set.
	E string
	// T contains NT (if K is Name) or ST (if K is Scalar) or nil (others)
	T interface{}
	// P is expected Token.Pos if set > 0.
	P int
	// RS is expected result from Token.RawString() if not empty.
	RS string
}

// NT contains data for checking against a name token.
type NT struct {
	K text.NameKind
	// Sep is true if name token should have separator character, else false.
	Sep bool
	// If K is IdentName or TypeName, invoke corresponding getter and compare against this field.
	S string
	// If K is FieldNumber, invoke getter and compare against this field.
	N int32
}

// ST contains data for checking against a scalar token.
type ST struct {
	// checker that is expected to return OK.
	ok checker
	// checker that is expected to return not OK.
	nok checker
}

// checker provides API for the token wrapper API call types Str, Enum, Bool,
// Uint64, Uint32, Int64, Int32, Float64, Float32.
type checker interface {
	// checkOk checks and expects for token API call to return ok and compare
	// against implementation-stored value. Returns empty string if success,
	// else returns error message describing the error.
	checkOk(text.Token) string
	// checkNok checks and expects for token API call to return not ok. Returns
	// empty string if success, else returns error message describing the error.
	checkNok(text.Token) string
}

type Str struct {
	val string
}

func (s Str) checkOk(tok text.Token) string {
	got, ok := tok.String()
	if !ok {
		return fmt.Sprintf("Token.String() returned not OK for token: %v", tok.RawString())
	}
	if got != s.val {
		return fmt.Sprintf("Token.String() got %q want %q for token: %v", got, s.val, tok.RawString())
	}
	return ""
}

func (s Str) checkNok(tok text.Token) string {
	if _, ok := tok.String(); ok {
		return fmt.Sprintf("Token.String() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Enum struct {
	val string
}

func (e Enum) checkOk(tok text.Token) string {
	got, ok := tok.Enum()
	if !ok {
		return fmt.Sprintf("Token.Enum() returned not OK for token: %v", tok.RawString())
	}
	if got != e.val {
		return fmt.Sprintf("Token.Enum() got %q want %q for token: %v", got, e.val, tok.RawString())
	}
	return ""
}

func (e Enum) checkNok(tok text.Token) string {
	if _, ok := tok.Enum(); ok {
		return fmt.Sprintf("Token.Enum() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Bool struct {
	val bool
}

func (b Bool) checkOk(tok text.Token) string {
	got, ok := tok.Bool()
	if !ok {
		return fmt.Sprintf("Token.Bool() returned not OK for token: %v", tok.RawString())
	}
	if got != b.val {
		return fmt.Sprintf("Token.Bool() got %v want %v for token: %v", got, b.val, tok.RawString())
	}
	return ""
}

func (b Bool) checkNok(tok text.Token) string {
	if _, ok := tok.Bool(); ok {
		return fmt.Sprintf("Token.Bool() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Uint64 struct {
	val uint64
}

func (n Uint64) checkOk(tok text.Token) string {
	got, ok := tok.Uint64()
	if !ok {
		return fmt.Sprintf("Token.Uint64() returned not OK for token: %v", tok.RawString())
	}
	if got != n.val {
		return fmt.Sprintf("Token.Uint64() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Uint64) checkNok(tok text.Token) string {
	if _, ok := tok.Uint64(); ok {
		return fmt.Sprintf("Token.Uint64() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Uint32 struct {
	val uint32
}

func (n Uint32) checkOk(tok text.Token) string {
	got, ok := tok.Uint32()
	if !ok {
		return fmt.Sprintf("Token.Uint32() returned not OK for token: %v", tok.RawString())
	}
	if got != n.val {
		return fmt.Sprintf("Token.Uint32() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Uint32) checkNok(tok text.Token) string {
	if _, ok := tok.Uint32(); ok {
		return fmt.Sprintf("Token.Uint32() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Int64 struct {
	val int64
}

func (n Int64) checkOk(tok text.Token) string {
	got, ok := tok.Int64()
	if !ok {
		return fmt.Sprintf("Token.Int64() returned not OK for token: %v", tok.RawString())
	}
	if got != n.val {
		return fmt.Sprintf("Token.Int64() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Int64) checkNok(tok text.Token) string {
	if _, ok := tok.Int64(); ok {
		return fmt.Sprintf("Token.Int64() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Int32 struct {
	val int32
}

func (n Int32) checkOk(tok text.Token) string {
	got, ok := tok.Int32()
	if !ok {
		return fmt.Sprintf("Token.Int32() returned not OK for token: %v", tok.RawString())
	}
	if got != n.val {
		return fmt.Sprintf("Token.Int32() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Int32) checkNok(tok text.Token) string {
	if _, ok := tok.Int32(); ok {
		return fmt.Sprintf("Token.Int32() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Float64 struct {
	val float64
}

func (n Float64) checkOk(tok text.Token) string {
	got, ok := tok.Float64()
	if !ok {
		return fmt.Sprintf("Token.Float64() returned not OK for token: %v", tok.RawString())
	}
	if math.Float64bits(got) != math.Float64bits(n.val) {
		return fmt.Sprintf("Token.Float64() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Float64) checkNok(tok text.Token) string {
	if _, ok := tok.Float64(); ok {
		return fmt.Sprintf("Token.Float64() returned OK for token: %v", tok.RawString())
	}
	return ""
}

type Float32 struct {
	val float32
}

func (n Float32) checkOk(tok text.Token) string {
	got, ok := tok.Float32()
	if !ok {
		return fmt.Sprintf("Token.Float32() returned not OK for token: %v", tok.RawString())
	}
	if math.Float32bits(got) != math.Float32bits(n.val) {
		return fmt.Sprintf("Token.Float32() got %v want %v for token: %v", got, n.val, tok.RawString())
	}
	return ""
}

func (n Float32) checkNok(tok text.Token) string {
	if _, ok := tok.Float32(); ok {
		return fmt.Sprintf("Token.Float32() returned OK for token: %v", tok.RawString())
	}
	return ""
}

func TestDecoder(t *testing.T) {
	const space = " \n\r\t"
	tests := []struct {
		in string
		// want is a list of expected Tokens returned from calling Decoder.Read.
		// An item makes the test code invoke Decoder.Read and compare against
		// R.K and R.E. If R.K is Name, it compares
		want []R
	}{
		{
			in:   "",
			want: []R{{K: text.EOF}},
		},
		{
			in:   "# comment",
			want: []R{{K: text.EOF}},
		},
		{
			in:   space + "# comment" + space,
			want: []R{{K: text.EOF}},
		},
		{
			in:   space,
			want: []R{{K: text.EOF, P: len(space)}},
		},
		{
			// Calling Read after EOF will keep returning EOF for
			// succeeding Read calls.
			in: space,
			want: []R{
				{K: text.EOF},
				{K: text.EOF},
				{K: text.EOF},
			},
		},
		{
			// NUL is an invalid whitespace since C++ uses C-strings.
			in:   "\x00",
			want: []R{{E: "invalid field name: \x00"}},
		},

		// Field names.
		{
			in: "name",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "name"}, RS: "name"},
				{E: eofErr},
			},
		},
		{
			in: space + "name:" + space,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
				{E: eofErr},
			},
		},
		{
			in: space + "name" + space + ":" + space,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
				{E: eofErr},
			},
		},
		{
			in: "name # comment",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "name"}},
				{E: eofErr},
			},
		},
		{
			// Comments only extend until the newline.
			in: "# comment \nname",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "name"}, P: 11},
			},
		},
		{
			in: "name # comment \n:",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
			},
		},
		{
			in: "name123",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "name123"}},
			},
		},
		{
			in: "name_123",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "name_123"}},
			},
		},
		{
			in: "_123",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, S: "_123"}},
			},
		},
		{
			in:   ":",
			want: []R{{E: "syntax error (line 1:1): invalid field name: :"}},
		},
		{
			in:   "\n\n\n {",
			want: []R{{E: "syntax error (line 4:2): invalid field name: {"}},
		},
		{
			in:   "123name",
			want: []R{{E: "invalid field name: 123name"}},
		},
		{
			in:   `/`,
			want: []R{{E: `invalid field name: /`}},
		},
		{
			in:   `世界`,
			want: []R{{E: `invalid field name: 世`}},
		},
		{
			in:   `1a/b`,
			want: []R{{E: `invalid field name: 1a`}},
		},
		{
			in:   `1c\d`,
			want: []R{{E: `invalid field name: 1c`}},
		},
		{
			in:   "\x84f",
			want: []R{{E: "invalid field name: \x84"}},
		},
		{
			in:   "\uFFFDxxx",
			want: []R{{E: "invalid field name: \uFFFD"}},
		},
		{
			in:   "-a234567890123456789012345678901234567890abc",
			want: []R{{E: "invalid field name: -a2345678901234567890123456789012…"}},
		},
		{
			in: "[type]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "type"}, RS: "[type]"},
			},
		},
		{
			// V1 allows this syntax. C++ does not, however, C++ also fails if
			// field is Any and does not contain '/'.
			in: "[/type]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "/type"}},
			},
		},
		{
			in:   "[.type]",
			want: []R{{E: "invalid type URL/extension field name: [.type]"}},
		},
		{
			in: "[pkg.Foo.extension_field]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "pkg.Foo.extension_field"}},
			},
		},
		{
			in: "[domain.com/type]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "domain.com/type"}},
			},
		},
		{
			in: "[domain.com/pkg.type]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "domain.com/pkg.type"}},
			},
		},
		{
			in: "[sub.domain.com\x2fpath\x2fto\x2fproto.package.name]",
			want: []R{
				{
					K: text.Name,
					T: NT{
						K: text.TypeName,
						S: "sub.domain.com/path/to/proto.package.name",
					},
					RS: "[sub.domain.com\x2fpath\x2fto\x2fproto.package.name]",
				},
			},
		},
		{
			// V2 no longer allows a quoted string for the Any type URL.
			in:   `["domain.com/pkg.type"]`,
			want: []R{{E: `invalid type URL/extension field name: ["`}},
		},
		{
			// V2 no longer allows a quoted string for the Any type URL.
			in:   `['domain.com/pkg.type']`,
			want: []R{{E: `invalid type URL/extension field name: ['`}},
		},
		{
			in:   "[pkg.Foo.extension_field:",
			want: []R{{E: "invalid type URL/extension field name: [pkg.Foo.extension_field:"}},
		},
		{
			// V2 no longer allows whitespace within identifier "word".
			in:   "[proto.packa ge.field]",
			want: []R{{E: "invalid type URL/extension field name: [proto.packa g"}},
		},
		{
			// V2 no longer allows comments within identifier "word".
			in:   "[proto.packa # comment\n ge.field]",
			want: []R{{E: "invalid type URL/extension field name: [proto.packa # comment\n g"}},
		},
		{
			in:   "[proto.package.]",
			want: []R{{E: "invalid type URL/extension field name: [proto.package."}},
		},
		{
			in:   "[proto.package/]",
			want: []R{{E: "invalid type URL/extension field name: [proto.package/"}},
		},
		{
			in: `message_field{[bad@]`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{E: `invalid type URL/extension field name: [bad@`},
			},
		},
		{
			in: `message_field{[invalid//type]`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{E: `invalid type URL/extension field name: [invalid//`},
			},
		},
		{
			in: `message_field{[proto.package.]`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{E: `invalid type URL/extension field name: [proto.package.`},
			},
		},
		{
			in:   "[proto.package",
			want: []R{{E: eofErr}},
		},
		{
			in: "[" + space + "type" + space + "]" + space + ":",
			want: []R{
				{
					K: text.Name,
					T: NT{
						K:   text.TypeName,
						Sep: true,
						S:   "type",
					},
					RS: "[" + space + "type" + space + "]",
				},
			},
		},
		{
			// Whitespaces/comments are only allowed betweeb
			in: "[" + space + "domain" + space + "." + space + "com # comment\n" +
				"/" + "pkg" + space + "." + space + "type" + space + "]",
			want: []R{
				{K: text.Name, T: NT{K: text.TypeName, S: "domain.com/pkg.type"}},
			},
		},
		{
			in: "42",
			want: []R{
				{K: text.Name, T: NT{K: text.FieldNumber, N: 42}},
			},
		},
		{
			in:   "0x42:",
			want: []R{{E: "invalid field number: 0x42"}},
		},
		{
			in:   "042:",
			want: []R{{E: "invalid field number: 042"}},
		},
		{
			in:   "123.456:",
			want: []R{{E: "invalid field number: 123.456"}},
		},
		{
			in:   "-123",
			want: []R{{E: "invalid field number: -123"}},
		},
		{
			in:   "- \t 123.321e6",
			want: []R{{E: "invalid field number: -123.321e6"}},
		},
		{
			in:   "-",
			want: []R{{E: "invalid field name: -"}},
		},
		{
			in:   "- ",
			want: []R{{E: "invalid field name: -"}},
		},
		{
			in:   "- # negative\n 123",
			want: []R{{E: "invalid field number: -123"}},
		},
		{
			// Field number > math.MaxInt32.
			in:   "2147483648:",
			want: []R{{E: "invalid field number: 2147483648"}},
		},

		// String field value. More string parsing specific testing in
		// TestUnmarshalString.
		{
			in: `name: "hello world"`,
			want: []R{
				{K: text.Name},
				{
					K:  text.Scalar,
					T:  ST{ok: Str{"hello world"}, nok: Enum{}},
					RS: `"hello world"`,
				},
				{K: text.EOF},
			},
		},
		{
			in: `name: 'hello'`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
			},
		},
		{
			in: `name: "hello'`,
			want: []R{
				{K: text.Name},
				{E: eofErr},
			},
		},
		{
			in: `name: 'hello`,
			want: []R{
				{K: text.Name},
				{E: eofErr},
			},
		},
		{
			// Field name without separator is ok. prototext package will need
			// to determine that this is not valid for scalar values.
			in: space + `name` + space + `"hello"` + space,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
			},
		},
		{
			in: `name'hello'`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
			},
		},
		{
			in: `name: ` + space + `"hello"` + space + `,`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name` + space + `:` + `"hello"` + space + `;` + space,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"hello" , ,`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar},
				{E: "(line 1:16): invalid field name: ,"},
			},
		},
		{
			in: `name:"hello" , ;`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar},
				{E: "(line 1:16): invalid field name: ;"},
			},
		},
		{
			in: `name:"hello" name:'world'`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"hello", name:"world"`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"hello"; name:"world",`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.EOF},
			},
		},
		{
			in: `foo:"hello"bar:"world"`,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "foo"}},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "bar"}},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.EOF},
			},
		},
		{
			in: `foo:"hello"[bar]:"world"`,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "foo"}},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.Name, T: NT{K: text.TypeName, Sep: true, S: "bar"}},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"foo"` + space + `"bar"` + space + `'qux'`,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
				{K: text.Scalar, T: ST{ok: Str{"foobarqux"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"foo"'bar'"qux"`,
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
				{K: text.Scalar, T: ST{ok: Str{"foobarqux"}}},
				{K: text.EOF},
			},
		},
		{
			in: `name:"foo"` + space + `"bar" # comment` + "\n'qux' # comment",
			want: []R{
				{K: text.Name, T: NT{K: text.IdentName, Sep: true, S: "name"}},
				{K: text.Scalar, T: ST{ok: Str{"foobarqux"}}},
				{K: text.EOF},
			},
		},

		// Lists.
		{
			in: `name: [`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{E: eofErr},
			},
		},
		{
			in: `name: []`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.ListClose},
				{K: text.EOF},
			},
		},
		{
			in: `name []`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.ListClose},
				{K: text.EOF},
			},
		},
		{
			in: `name: [,`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{E: `(line 1:8): invalid scalar value: ,`},
			},
		},
		{
			in: `name: [0`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar},
				{E: eofErr},
			},
		},
		{
			in: `name: [` + space + `"hello"` + space + `]` + space,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}, P: len(space) + 7},
				{K: text.ListClose},
				{K: text.EOF},
			},
		},
		{
			in: `name: ["hello",]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{E: `invalid scalar value: ]`},
			},
		},
		{
			in: `name: ["foo"` + space + `'bar' "qux"]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"foobarqux"}}},
				{K: text.ListClose},
				{K: text.EOF},
			},
		},
		{
			in: `name:` + space + `["foo",` + space + "'bar', # comment\n\n" + `"qux"]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"foo"}}},
				{K: text.Scalar, T: ST{ok: Str{"bar"}}},
				{K: text.Scalar, T: ST{ok: Str{"qux"}}},
				{K: text.ListClose},
				{K: text.EOF},
			},
		},

		{
			// List within list is not allowed.
			in: `name: [[]]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{E: `syntax error (line 1:8): invalid scalar value: [`},
			},
		},
		{
			// List items need to be separated by ,.
			in: `name: ["foo" true]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"foo"}}},
				{E: `syntax error (line 1:14): unexpected character 't'`},
			},
		},
		{
			in: `name: ["foo"; "bar"]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"foo"}}},
				{E: `syntax error (line 1:13): unexpected character ';'`},
			},
		},
		{
			in: `name: ["foo", true, ENUM, 1.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"foo"}}},
				{K: text.Scalar, T: ST{ok: Enum{"true"}}},
				{K: text.Scalar, T: ST{ok: Enum{"ENUM"}}},
				{K: text.Scalar, T: ST{ok: Float32{1.0}}},
				{K: text.ListClose},
			},
		},

		// Boolean literal values.
		{
			in: `name: True`,
			want: []R{
				{K: text.Name},
				{
					K: text.Scalar,
					T: ST{ok: Bool{true}},
				},
				{K: text.EOF},
			},
		},
		{
			in: `name false`,
			want: []R{
				{K: text.Name},
				{
					K: text.Scalar,
					T: ST{ok: Bool{false}},
				},
				{K: text.EOF},
			},
		},
		{
			in: `name: [t, f, True, False, true, false, 1, 0, 0x01, 0x00, 01, 00]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
				{K: text.ListClose},
			},
		},
		{
			// Looks like boolean but not.
			in: `name: [tRUe, falSE, -1, -0, -0x01, -0x00, -01, -00, 0.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.Scalar, T: ST{nok: Bool{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `foo: true[bar] false`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Bool{true}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Bool{false}}},
			},
		},

		// Enum field values.
		{
			in: space + `name: ENUM`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Enum{"ENUM"}}},
			},
		},
		{
			in: space + `name:[TRUE, FALSE, T, F, t, f]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Enum{"TRUE"}}},
				{K: text.Scalar, T: ST{ok: Enum{"FALSE"}}},
				{K: text.Scalar, T: ST{ok: Enum{"T"}}},
				{K: text.Scalar, T: ST{ok: Enum{"F"}}},
				{K: text.Scalar, T: ST{ok: Enum{"t"}}},
				{K: text.Scalar, T: ST{ok: Enum{"f"}}},
				{K: text.ListClose},
			},
		},
		{
			in: `foo: Enum1[bar]:Enum2`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Enum{"Enum1"}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Enum{"Enum2"}}},
			},
		},
		{
			// Invalid enum values.
			in: `name: [-inf, -foo, "string", 42, 1.0, 0x47]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.Scalar, T: ST{nok: Enum{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name: true.`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: true.`},
			},
		},

		// Numeric values.
		{
			in: `nums:42 nums:0x2A nums:052`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Uint64{42}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Uint64{42}}},
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Uint64{42}}},
			},
		},
		{
			in: `nums:[-42, -0x2a, -052]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums:[-42, -0x2a, -052]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Int64{-42}}},
				{K: text.Scalar, T: ST{ok: Int64{-42}}},
				{K: text.Scalar, T: ST{ok: Int64{-42}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0,0x0,00,-9876543210,9876543210,0x0123456789abcdef,-0x0123456789abcdef,01234567,-01234567]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Uint64{0}}},
				{K: text.Scalar, T: ST{ok: Int64{0}}},
				{K: text.Scalar, T: ST{ok: Uint64{0}}},
				{K: text.Scalar, T: ST{ok: Int64{-9876543210}}},
				{K: text.Scalar, T: ST{ok: Uint64{9876543210}}},
				{K: text.Scalar, T: ST{ok: Uint64{0x0123456789abcdef}}},
				{K: text.Scalar, T: ST{ok: Int64{-0x0123456789abcdef}}},
				{K: text.Scalar, T: ST{ok: Uint64{01234567}}},
				{K: text.Scalar, T: ST{ok: Int64{-01234567}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0,0x0,00,-876543210,876543210,0x01234,-0x01234,01234567,-01234567]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Uint32{0}}},
				{K: text.Scalar, T: ST{ok: Int32{0}}},
				{K: text.Scalar, T: ST{ok: Uint32{0}}},
				{K: text.Scalar, T: ST{ok: Int32{-876543210}}},
				{K: text.Scalar, T: ST{ok: Uint32{876543210}}},
				{K: text.Scalar, T: ST{ok: Uint32{0x01234}}},
				{K: text.Scalar, T: ST{ok: Int32{-0x01234}}},
				{K: text.Scalar, T: ST{ok: Uint32{01234567}}},
				{K: text.Scalar, T: ST{ok: Int32{-01234567}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [` +
				fmt.Sprintf("%d", uint64(math.MaxUint64)) + `,` +
				fmt.Sprintf("%d", uint32(math.MaxUint32)) + `,` +
				fmt.Sprintf("%d", int64(math.MaxInt64)) + `,` +
				fmt.Sprintf("%d", int64(math.MinInt64)) + `,` +
				fmt.Sprintf("%d", int32(math.MaxInt32)) + `,` +
				fmt.Sprintf("%d", int32(math.MinInt32)) +
				`]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Uint64{math.MaxUint64}}},
				{K: text.Scalar, T: ST{ok: Uint32{math.MaxUint32}}},
				{K: text.Scalar, T: ST{ok: Int64{math.MaxInt64}}},
				{K: text.Scalar, T: ST{ok: Int64{math.MinInt64}}},
				{K: text.Scalar, T: ST{ok: Int32{math.MaxInt32}}},
				{K: text.Scalar, T: ST{ok: Int32{math.MinInt32}}},
				{K: text.ListClose},
			},
		},
		{
			// Integer exceeds range.
			in: `nums: [` +
				`18446744073709551616,` + // max uint64 + 1
				fmt.Sprintf("%d", uint64(math.MaxUint32+1)) + `,` +
				fmt.Sprintf("%d", uint64(math.MaxInt64+1)) + `,` +
				`-9223372036854775809,` + // min int64 - 1
				fmt.Sprintf("%d", uint64(math.MaxInt32+1)) + `,` +
				fmt.Sprintf("%d", int64(math.MinInt32-1)) + `` +
				`]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0xbeefbeef, 0xbeefbeefbeefbeef]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{
					K: text.Scalar,
					T: func() ST {
						if flags.ProtoLegacy {
							return ST{ok: Int32{-1091584273}}
						}
						return ST{nok: Int32{}}
					}(),
				},
				{
					K: text.Scalar,
					T: func() ST {
						if flags.ProtoLegacy {
							return ST{ok: Int64{-4688318750159552785}}
						}
						return ST{nok: Int64{}}
					}(),
				},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,0f,1f,10f,-0f,-1f,-10f,1.0,0.1e-3,1.5e+5,1e10,.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{0.0}}},
				{K: text.Scalar, T: ST{ok: Float64{0.0}}},
				{K: text.Scalar, T: ST{ok: Float64{1.0}}},
				{K: text.Scalar, T: ST{ok: Float64{10.0}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Copysign(0, -1)}}},
				{K: text.Scalar, T: ST{ok: Float64{-1.0}}},
				{K: text.Scalar, T: ST{ok: Float64{-10.0}}},
				{K: text.Scalar, T: ST{ok: Float64{1.0}}},
				{K: text.Scalar, T: ST{ok: Float64{0.1e-3}}},
				{K: text.Scalar, T: ST{ok: Float64{1.5e+5}}},
				{K: text.Scalar, T: ST{ok: Float64{1.0e+10}}},
				{K: text.Scalar, T: ST{ok: Float64{0.0}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,0f,1f,10f,-0f,-1f,-10f,1.0,0.1e-3,1.5e+5,1e10,.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{0.0}}},
				{K: text.Scalar, T: ST{ok: Float32{0.0}}},
				{K: text.Scalar, T: ST{ok: Float32{1.0}}},
				{K: text.Scalar, T: ST{ok: Float32{10.0}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Copysign(0, -1))}}},
				{K: text.Scalar, T: ST{ok: Float32{-1.0}}},
				{K: text.Scalar, T: ST{ok: Float32{-10.0}}},
				{K: text.Scalar, T: ST{ok: Float32{1.0}}},
				{K: text.Scalar, T: ST{ok: Float32{0.1e-3}}},
				{K: text.Scalar, T: ST{ok: Float32{1.5e+5}}},
				{K: text.Scalar, T: ST{ok: Float32{1.0e+10}}},
				{K: text.Scalar, T: ST{ok: Float32{0.0}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,1f,10F,1e1,1.10]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.Scalar, T: ST{nok: Int64{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,1f,10F,1e1,1.10]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.Scalar, T: ST{nok: Int32{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,1f,10F,1e1,1.10]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [0.,1f,10F,1e1,1.10]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [` +
				fmt.Sprintf("%g", math.MaxFloat32) + `,` +
				fmt.Sprintf("%g", -math.MaxFloat32) + `,` +
				fmt.Sprintf("%g", math.MaxFloat32*2) + `,` +
				fmt.Sprintf("%g", -math.MaxFloat32*2) + `,` +
				`3.59539e+308,` + // math.MaxFloat64 * 2
				`-3.59539e+308,` + // -math.MaxFloat64 * 2
				fmt.Sprintf("%d000", uint64(math.MaxUint64)) +
				`]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.MaxFloat32)}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(-math.MaxFloat32)}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.MaxUint64) * 1000}}},
				{K: text.ListClose},
			},
		},
		{
			in: `nums: [` +
				fmt.Sprintf("%g", math.MaxFloat64) + `,` +
				fmt.Sprintf("%g", -math.MaxFloat64) + `,` +
				`3.59539e+308,` + // math.MaxFloat64 * 2
				`-3.59539e+308,` + // -math.MaxFloat64 * 2
				fmt.Sprintf("%d000", uint64(math.MaxUint64)) +
				`]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{math.MaxFloat64}}},
				{K: text.Scalar, T: ST{ok: Float64{-math.MaxFloat64}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.Scalar, T: ST{ok: Float64{float64(math.MaxUint64) * 1000}}},
				{K: text.ListClose},
			},
		},
		{
			// -0 is only valid for signed types. It is not valid for unsigned types.
			in: `num: [-0, -0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{nok: Uint32{}}},
				{K: text.Scalar, T: ST{nok: Uint64{}}},
				{K: text.ListClose},
			},
		},
		{
			// -0 is only valid for signed types. It is not valid for unsigned types.
			in: `num: [-0, -0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Int32{0}}},
				{K: text.Scalar, T: ST{ok: Int64{0}}},
				{K: text.ListClose},
			},
		},
		{
			// Negative zeros on float64 should preserve sign bit.
			in: `num: [-0, -.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{math.Copysign(0, -1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Copysign(0, -1)}}},
				{K: text.ListClose},
			},
		},
		{
			// Negative zeros on float32 should preserve sign bit.
			in: `num: [-0, -.0]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Copysign(0, -1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Copysign(0, -1))}}},
				{K: text.ListClose},
			},
		},
		{
			in: `num: +0`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: +`},
			},
		},
		{
			in: `num: 01.1234`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: 01.1234`},
			},
		},
		{
			in: `num: 0x`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: 0x`},
			},
		},
		{
			in: `num: 0xX`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: 0xX`},
			},
		},
		{
			in: `num: 0800`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: 0800`},
			},
		},
		{
			in: `num: 1.`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Float32{1.0}}},
			},
		},
		{
			in: `num: -.`,
			want: []R{
				{K: text.Name},
				{E: `invalid scalar value: -.`},
			},
		},

		// Float special literal values, case-insensitive match.
		{
			in: `name:[nan, NaN, Nan, NAN]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{math.NaN()}}},
				{K: text.Scalar, T: ST{ok: Float64{math.NaN()}}},
				{K: text.Scalar, T: ST{ok: Float64{math.NaN()}}},
				{K: text.Scalar, T: ST{ok: Float64{math.NaN()}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name:[inf, INF, infinity, Infinity, INFinity]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(1)}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name:[-inf, -INF, -infinity, -Infinity, -INFinity]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.Scalar, T: ST{ok: Float64{math.Inf(-1)}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name:[nan, NaN, Nan, NAN]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.NaN())}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.NaN())}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.NaN())}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.NaN())}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name:[inf, INF, infinity, Infinity, INFinity]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(1))}}},
				{K: text.ListClose},
			},
		},
		{
			in: `name:[-inf, -INF, -infinity, -Infinity, -INFinity]`,
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.Scalar, T: ST{ok: Float32{float32(math.Inf(-1))}}},
				{K: text.ListClose},
			},
		},
		{
			// C++ permits this, but we currently reject this. It is easy to add
			// if needed.
			in: `name: -nan`,
			want: []R{
				{K: text.Name},
				{K: text.Scalar, T: ST{nok: Float64{}}},
			},
		},
		// Messages.
		{
			in: `m: {}`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{K: text.MessageClose},
				{K: text.EOF},
			},
		},
		{
			in: `m: <>`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{K: text.MessageClose},
				{K: text.EOF},
			},
		},
		{
			in: space + `m {` + space + "\n# comment\n" + `}` + space,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{K: text.MessageClose},
			},
		},
		{
			in: `m { foo: < bar: "hello" > }`,
			want: []R{
				{K: text.Name, RS: "m"},
				{K: text.MessageOpen},

				{K: text.Name, RS: "foo"},
				{K: text.MessageOpen},

				{K: text.Name, RS: "bar"},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},

				{K: text.MessageClose},

				{K: text.MessageClose},
			},
		},
		{
			in: `list [ <s:"hello">, {s:"world"} ]`,
			want: []R{
				{K: text.Name, RS: "list"},
				{K: text.ListOpen},

				{K: text.MessageOpen},
				{K: text.Name, RS: "s"},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},
				{K: text.MessageClose},

				{K: text.MessageOpen},
				{K: text.Name, RS: "s"},
				{K: text.Scalar, T: ST{ok: Str{"world"}}},
				{K: text.MessageClose},

				{K: text.ListClose},
				{K: text.EOF},
			},
		},
		{
			in: `m: { >`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},
				{E: `mismatched close character '>'`},
			},
		},
		{
			in: `m: <s: "hello"}`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},

				{K: text.Name},
				{K: text.Scalar, T: ST{ok: Str{"hello"}}},

				{E: `mismatched close character '}'`},
			},
		},
		{
			in:   `{}`,
			want: []R{{E: `invalid field name: {`}},
		},
		{
			in: `
m: {
  foo: true;
  bar: {
	enum: ENUM
	list: [ < >, { } ] ;
  }
  [qux]: "end"
}
				`,
			want: []R{
				{K: text.Name},
				{K: text.MessageOpen},

				{K: text.Name, RS: "foo"},
				{K: text.Scalar, T: ST{ok: Bool{true}}},

				{K: text.Name, RS: "bar"},
				{K: text.MessageOpen},

				{K: text.Name, RS: "enum"},
				{K: text.Scalar, T: ST{ok: Enum{"ENUM"}}},

				{K: text.Name, RS: "list"},
				{K: text.ListOpen},
				{K: text.MessageOpen},
				{K: text.MessageClose},
				{K: text.MessageOpen},
				{K: text.MessageClose},
				{K: text.ListClose},

				{K: text.MessageClose},

				{K: text.Name, RS: "[qux]"},
				{K: text.Scalar, T: ST{ok: Str{"end"}}},

				{K: text.MessageClose},
				{K: text.EOF},
			},
		},

		// Other syntax errors.
		{
			in: "x: -",
			want: []R{
				{K: text.Name},
				{E: `syntax error (line 1:4): invalid scalar value: -`},
			},
		},
		{
			in: "x:[\"💩\"x",
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"💩"}}, P: 3},
				{E: `syntax error (line 1:7)`},
			},
		},
		{
			in: "x:\n\n[\"🔥🔥🔥\"x",
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"🔥🔥🔥"}}, P: 5},
				{E: `syntax error (line 3:7)`},
			},
		},
		{
			// multi-rune emojis; could be column:8
			in: "x:[\"👍🏻👍🏿\"x",
			want: []R{
				{K: text.Name},
				{K: text.ListOpen},
				{K: text.Scalar, T: ST{ok: Str{"👍🏻👍🏿"}}, P: 3},
				{E: `syntax error (line 1:10)`},
			},
		},
	}

	for _, tc := range tests {
		t.Run("", func(t *testing.T) {
			tc := tc
			in := []byte(tc.in)
			dec := text.NewDecoder(in[:len(in):len(in)])
			for i, want := range tc.want {
				peekTok, peekErr := dec.Peek()
				tok, err := dec.Read()
				if err != nil {
					if want.E == "" {
						errorf(t, tc.in, "Read() got unexpected error: %v", err)
					} else if !strings.Contains(err.Error(), want.E) {
						errorf(t, tc.in, "Read() got %q, want %q", err, want.E)
					}
					return
				}
				if want.E != "" {
					errorf(t, tc.in, "Read() got nil error, want %q", want.E)
					return
				}
				gotK := tok.Kind()
				if gotK != want.K {
					errorf(t, tc.in, "Read() got %v, want %v", gotK, want.K)
					return
				}
				checkToken(t, tok, i, want, tc.in)
				if !cmp.Equal(tok, peekTok, cmp.Comparer(text.TokenEquals)) {
					errorf(t, tc.in, "Peek() %+v != Read() token %+v", peekTok, tok)
				}
				if err != peekErr {
					errorf(t, tc.in, "Peek() error %v != Read() error %v", err, peekErr)
				}
			}
		})
	}
}

func checkToken(t *testing.T, tok text.Token, idx int, r R, in string) {
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

	// Skip checking for Token details if r.T is not set.
	if r.T == nil {
		return
	}

	switch tok.Kind() {
	case text.Name:
		want := r.T.(NT)
		kind := tok.NameKind()
		if kind != want.K {
			errorf(t, in, "want#%d: Token.NameKind() got %v want %v", idx, kind, want.K)
			return
		}
		switch kind {
		case text.IdentName:
			got := tok.IdentName()
			if got != want.S {
				errorf(t, in, "want#%d: Token.IdentName() got %v want %v", idx, got, want.S)
			}
		case text.TypeName:
			got := tok.TypeName()
			if got != want.S {
				errorf(t, in, "want#%d: Token.TypeName() got %v want %v", idx, got, want.S)
			}
		case text.FieldNumber:
			got := tok.FieldNumber()
			if got != want.N {
				errorf(t, in, "want#%d: Token.FieldNumber() got %v want %v", idx, got, want.N)
			}
		}

	case text.Scalar:
		want := r.T.(ST)
		if ok := want.ok; ok != nil {
			if err := ok.checkOk(tok); err != "" {
				errorf(t, in, "want#%d: %s", idx, err)
			}
		}
		if nok := want.nok; nok != nil {
			if err := nok.checkNok(tok); err != "" {
				errorf(t, in, "want#%d: %s", idx, err)
			}
		}
	}
}

func errorf(t *testing.T, in string, fmtStr string, args ...interface{}) {
	t.Helper()
	vargs := []interface{}{in}
	for _, arg := range args {
		vargs = append(vargs, arg)
	}
	t.Errorf("input:\n%s\n~end~\n"+fmtStr, vargs...)
}

func TestUnmarshalString(t *testing.T) {
	tests := []struct {
		in string
		// want is expected string result.
		want string
		// err is expected error substring from calling DecodeString if set.
		err string
	}{
		{
			in: func() string {
				var b []byte
				for i := 0; i < utf8.RuneSelf; i++ {
					switch i {
					case 0, '\\', '\n', '\'': // these must be escaped, so ignore them
					default:
						b = append(b, byte(i))
					}
				}
				return "'" + string(b) + "'"
			}(),
			want: "\x01\x02\x03\x04\x05\x06\a\b\t\v\f\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !\"#$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\u007f",
		},
		{
			in:  "'\xde\xad\xbe\xef'",
			err: `invalid UTF-8 detected`,
		},
		{
			// Valid UTF-8 wire encoding, but sub-optimal encoding.
			in:  "'\xc0\x80'",
			err: "invalid UTF-8 detected",
		},
		{
			// Valid UTF-8 wire encoding, but invalid rune (surrogate pair).
			in:  "'\xed\xa0\x80'",
			err: "invalid UTF-8 detected",
		},
		{
			// Valid UTF-8 wire encoding, but invalid rune (above max rune).
			in:  "'\xf7\xbf\xbf\xbf'",
			err: "invalid UTF-8 detected",
		},
		{
			// Valid UTF-8 wire encoding of the RuneError rune.
			in:   "'\xef\xbf\xbd'",
			want: string(utf8.RuneError),
		},
		{
			in:   "'hello\u1234world'",
			want: "hello\u1234world",
		},
		{
			in:   `'\"\'\\\?\a\b\n\r\t\v\f\1\12\123\xA\xaB\x12\uAb8f\U0010FFFF'`,
			want: "\"'\\?\a\b\n\r\t\v\f\x01\nS\n\xab\x12\uab8f\U0010ffff",
		},
		{
			in:  `str: '\8'`,
			err: `invalid escape code "\\8" in string`,
		},
		{
			in:   `'\1x'`,
			want: "\001x",
		},
		{
			in:   `'\12x'`,
			want: "\012x",
		},
		{
			in:   `'\123x'`,
			want: "\123x",
		},
		{
			in:   `'\1234x'`,
			want: "\1234x",
		},
		{
			in:   `'\1'`,
			want: "\001",
		},
		{
			in:   `'\12'`,
			want: "\012",
		},
		{
			in:   `'\123'`,
			want: "\123",
		},
		{
			in:   `'\1234'`,
			want: "\1234",
		},
		{
			in:   `'\377'`,
			want: "\377",
		},
		{
			// Overflow octal escape.
			in:  `'\400'`,
			err: `invalid octal escape code "\\400" in string`,
		},
		{
			in:   `'\xfx'`,
			want: "\x0fx",
		},
		{
			in:   `'\xffx'`,
			want: "\xffx",
		},
		{
			in:   `'\xfffx'`,
			want: "\xfffx",
		},
		{
			in:   `'\xf'`,
			want: "\x0f",
		},
		{
			in:   `'\xff'`,
			want: "\xff",
		},
		{
			in:   `'\xfff'`,
			want: "\xfff",
		},
		{
			in:  `'\xz'`,
			err: `invalid hex escape code "\\x" in string`,
		},
		{
			in:  `'\uPo'`,
			err: eofErr,
		},
		{
			in:  `'\uPoo'`,
			err: `invalid Unicode escape code "\\uPoo'" in string`,
		},
		{
			in:  `str: '\uPoop'`,
			err: `invalid Unicode escape code "\\uPoop" in string`,
		},
		{
			// Unmatched surrogate pair.
			in:  `str: '\uDEAD'`,
			err: `unexpected EOF`, // trying to reader other half
		},
		{
			// Surrogate pair with invalid other half.
			in:  `str: '\uDEAD\u0000'`,
			err: `invalid Unicode escape code "\\u0000" in string`,
		},
		{
			// Properly matched surrogate pair.
			in:   `'\uD800\uDEAD'`,
			want: "𐊭",
		},
		{
			// Overflow on Unicode rune.
			in:  `'\U00110000'`,
			err: `invalid Unicode escape code "\\U00110000" in string`,
		},
		{
			in:  `'\z'`,
			err: `invalid escape code "\\z" in string`,
		},
		{
			// Strings cannot have NUL literal since C-style strings forbid them.
			in:  "'\x00'",
			err: `invalid character '\x00' in string`,
		},
		{
			// Strings cannot have newline literal. The C++ permits them if an
			// option is specified to allow them. In Go, we always forbid them.
			in:  "'\n'",
			err: `invalid character '\n' in string`,
		},
	}

	for _, tc := range tests {
		t.Run("", func(t *testing.T) {
			got, err := text.UnmarshalString(tc.in)
			if err != nil {
				if tc.err == "" {
					errorf(t, tc.in, "UnmarshalString() got unexpected error: %q", err)
				} else if !strings.Contains(err.Error(), tc.err) {
					errorf(t, tc.in, "UnmarshalString() error got %q, want %q", err, tc.err)
				}
				return
			}
			if tc.err != "" {
				errorf(t, tc.in, "UnmarshalString() got nil error, want %q", tc.err)
				return
			}
			if got != tc.want {
				errorf(t, tc.in, "UnmarshalString()\n[got]\n%s\n[want]\n%s", got, tc.want)
			}
		})
	}
}

// Tests line and column number produced by Decoder.Position.
func TestPosition(t *testing.T) {
	dec := text.NewDecoder([]byte("0123456789\n12345\n789"))

	tests := []struct {
		pos int
		row int
		col int
	}{
		{
			pos: 0,
			row: 1,
			col: 1,
		},
		{
			pos: 10,
			row: 1,
			col: 11,
		},
		{
			pos: 11,
			row: 2,
			col: 1,
		},
		{
			pos: 18,
			row: 3,
			col: 2,
		},
	}

	for _, tc := range tests {
		t.Run("", func(t *testing.T) {
			row, col := dec.Position(tc.pos)
			if row != tc.row || col != tc.col {
				t.Errorf("Position(%d) got (%d,%d) want (%d,%d)", tc.pos, row, col, tc.row, tc.col)
			}
		})
	}
}
