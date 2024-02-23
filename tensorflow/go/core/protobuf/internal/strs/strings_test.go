// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strs

import (
	"strconv"
	"testing"
)

func TestGoCamelCase(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"", ""},
		{"one", "One"},
		{"one_two", "OneTwo"},
		{"_my_field_name_2", "XMyFieldName_2"},
		{"Something_Capped", "Something_Capped"},
		{"my_Name", "My_Name"},
		{"OneTwo", "OneTwo"},
		{"_", "X"},
		{"_a_", "XA_"},
		{"one.two", "OneTwo"},
		{"one.Two", "One_Two"},
		{"one_two.three_four", "OneTwoThreeFour"},
		{"one_two.Three_four", "OneTwo_ThreeFour"},
		{"_one._two", "XOne_XTwo"},
		{"SCREAMING_SNAKE_CASE", "SCREAMING_SNAKE_CASE"},
		{"double__underscore", "Double_Underscore"},
		{"camelCase", "CamelCase"},
		{"go2proto", "Go2Proto"},
		{"世界", "世界"},
		{"x世界", "X世界"},
		{"foo_bar世界", "FooBar世界"},
	}
	for _, tc := range tests {
		if got := GoCamelCase(tc.in); got != tc.want {
			t.Errorf("GoCamelCase(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestGoSanitized(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"", "_"},
		{"boo", "boo"},
		{"Boo", "Boo"},
		{"ßoo", "ßoo"},
		{"default", "_default"},
		{"hello", "hello"},
		{"hello-world!!", "hello_world__"},
		{"hello-\xde\xad\xbe\xef\x00", "hello_____"},
		{"hello 世界", "hello_世界"},
		{"世界", "世界"},
	}
	for _, tc := range tests {
		if got := GoSanitized(tc.in); got != tc.want {
			t.Errorf("GoSanitized(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestName(t *testing.T) {
	tests := []struct {
		in                string
		inEnumPrefix      string
		wantMapEntry      string
		wantEnumValue     string
		wantTrimValue     string
		wantJSONCamelCase string
		wantJSONSnakeCase string
	}{{
		in:                "abc",
		inEnumPrefix:      "",
		wantMapEntry:      "AbcEntry",
		wantEnumValue:     "Abc",
		wantTrimValue:     "abc",
		wantJSONCamelCase: "abc",
		wantJSONSnakeCase: "abc",
	}, {
		in:                "foo_baR_",
		inEnumPrefix:      "foo_bar",
		wantMapEntry:      "FooBaREntry",
		wantEnumValue:     "FooBar",
		wantTrimValue:     "foo_baR_",
		wantJSONCamelCase: "fooBaR",
		wantJSONSnakeCase: "foo_ba_r_",
	}, {
		in:                "snake_caseCamelCase",
		inEnumPrefix:      "snakecasecamel",
		wantMapEntry:      "SnakeCaseCamelCaseEntry",
		wantEnumValue:     "SnakeCasecamelcase",
		wantTrimValue:     "Case",
		wantJSONCamelCase: "snakeCaseCamelCase",
		wantJSONSnakeCase: "snake_case_camel_case",
	}, {
		in:                "FiZz_BuZz",
		inEnumPrefix:      "fizz",
		wantMapEntry:      "FiZzBuZzEntry",
		wantEnumValue:     "FizzBuzz",
		wantTrimValue:     "BuZz",
		wantJSONCamelCase: "FiZzBuZz",
		wantJSONSnakeCase: "_fi_zz__bu_zz",
	}}

	for _, tt := range tests {
		if got := MapEntryName(tt.in); got != tt.wantMapEntry {
			t.Errorf("MapEntryName(%q) = %q, want %q", tt.in, got, tt.wantMapEntry)
		}
		if got := EnumValueName(tt.in); got != tt.wantEnumValue {
			t.Errorf("EnumValueName(%q) = %q, want %q", tt.in, got, tt.wantEnumValue)
		}
		if got := TrimEnumPrefix(tt.in, tt.inEnumPrefix); got != tt.wantTrimValue {
			t.Errorf("ErimEnumPrefix(%q, %q) = %q, want %q", tt.in, tt.inEnumPrefix, got, tt.wantTrimValue)
		}
		if got := JSONCamelCase(tt.in); got != tt.wantJSONCamelCase {
			t.Errorf("JSONCamelCase(%q) = %q, want %q", tt.in, got, tt.wantJSONCamelCase)
		}
		if got := JSONSnakeCase(tt.in); got != tt.wantJSONSnakeCase {
			t.Errorf("JSONSnakeCase(%q) = %q, want %q", tt.in, got, tt.wantJSONSnakeCase)
		}
	}
}

var (
	srcString = "1234"
	srcBytes  = []byte(srcString)
	dst       uint64
)

func BenchmarkCast(b *testing.B) {
	b.Run("Ideal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			dst, _ = strconv.ParseUint(srcString, 0, 64)
		}
		if dst != 1234 {
			b.Errorf("got %d, want %s", dst, srcString)
		}
	})
	b.Run("Copy", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			dst, _ = strconv.ParseUint(string(srcBytes), 0, 64)
		}
		if dst != 1234 {
			b.Errorf("got %d, want %s", dst, srcString)
		}
	})
	b.Run("Cast", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			dst, _ = strconv.ParseUint(UnsafeString(srcBytes), 0, 64)
		}
		if dst != 1234 {
			b.Errorf("got %d, want %s", dst, srcString)
		}
	})
}
