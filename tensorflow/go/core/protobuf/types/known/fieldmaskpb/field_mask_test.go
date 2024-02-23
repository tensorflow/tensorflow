// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fieldmaskpb_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	fmpb "google.golang.org/protobuf/types/known/fieldmaskpb"
)

func TestAppend(t *testing.T) {
	tests := []struct {
		inMessage proto.Message
		inPaths   []string
		wantPaths []string
		wantError error
	}{{
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{},
		wantPaths: []string{},
	}, {
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{"paths", "paths"},
		wantPaths: []string{"paths", "paths"},
	}, {
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{"paths", "<INVALID>", "paths"},
		wantPaths: []string{"paths"},
		wantError: cmpopts.AnyError,
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "OptionalGroup.optional_nested_message", "map_uint32_uint32", "map_string_nested_message.corecursive", "oneof_bool"},
		wantPaths: []string{"optional_int32", "OptionalGroup.optional_nested_message", "map_uint32_uint32"},
		wantError: cmpopts.AnyError,
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_nested_message", "optional_nested_message.corecursive", "optional_nested_message.corecursive.optional_nested_message", "optional_nested_message.corecursive.optional_nested_message.corecursive"},
		wantPaths: []string{"optional_nested_message", "optional_nested_message.corecursive", "optional_nested_message.corecursive.optional_nested_message", "optional_nested_message.corecursive.optional_nested_message.corecursive"},
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "optional_nested_message.corecursive.optional_int64", "optional_nested_message.corecursive.<INVALID>", "optional_int64"},
		wantPaths: []string{"optional_int32", "optional_nested_message.corecursive.optional_int64"},
		wantError: cmpopts.AnyError,
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "optional_nested_message.corecursive.oneof_uint32", "optional_nested_message.oneof_field", "optional_int64"},
		wantPaths: []string{"optional_int32", "optional_nested_message.corecursive.oneof_uint32"},
		wantError: cmpopts.AnyError,
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var mask fmpb.FieldMask
			gotError := mask.Append(tt.inMessage, tt.inPaths...)
			gotPaths := mask.GetPaths()
			if diff := cmp.Diff(tt.wantPaths, gotPaths, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Append() paths mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantError, gotError, cmpopts.EquateErrors()); diff != "" {
				t.Errorf("Append() error mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCombine(t *testing.T) {
	tests := []struct {
		in            [][]string
		wantUnion     []string
		wantIntersect []string
	}{{
		in: [][]string{
			{},
			{},
		},
		wantUnion:     []string{},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a"},
			{},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a"},
			{"a"},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{"a"},
	}, {
		in: [][]string{
			{"a"},
			{"b"},
			{"c"},
		},
		wantUnion:     []string{"a", "b", "c"},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a", "b"},
			{"b.b"},
			{"b"},
			{"b", "a.A"},
			{"b", "c", "c.a", "c.b"},
		},
		wantUnion:     []string{"a", "b", "c"},
		wantIntersect: []string{"b.b"},
	}, {
		in: [][]string{
			{"a.b", "a.c.d"},
			{"a"},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{"a.b", "a.c.d"},
	}, {
		in: [][]string{
			{},
			{"a.b", "a.c", "d"},
		},
		wantUnion:     []string{"a.b", "a.c", "d"},
		wantIntersect: []string{},
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var masks []*fmpb.FieldMask
			for _, paths := range tt.in {
				masks = append(masks, &fmpb.FieldMask{Paths: paths})
			}

			union := fmpb.Union(masks[0], masks[1], masks[2:]...)
			gotUnion := union.GetPaths()
			if diff := cmp.Diff(tt.wantUnion, gotUnion, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Union() mismatch (-want +got):\n%s", diff)
			}

			intersect := fmpb.Intersect(masks[0], masks[1], masks[2:]...)
			gotIntersect := intersect.GetPaths()
			if diff := cmp.Diff(tt.wantIntersect, gotIntersect, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Intersect() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		in   []string
		want []string
	}{{
		in:   []string{},
		want: []string{},
	}, {
		in:   []string{"a"},
		want: []string{"a"},
	}, {
		in:   []string{"foo", "foo.bar", "foo.baz"},
		want: []string{"foo"},
	}, {
		in:   []string{"foo.bar", "foo.baz"},
		want: []string{"foo.bar", "foo.baz"},
	}, {
		in:   []string{"", "a.", ".b", "a.b", ".", "", "a.", ".b", "a.b", "."},
		want: []string{"", "a.", "a.b"},
	}, {
		in:   []string{"e.a", "e.b", "e.c", "e.d", "e.f", "e.g", "e.b.a", "e$c", "e.b.c"},
		want: []string{"e.a", "e.b", "e.c", "e.d", "e.f", "e.g", "e$c"},
	}, {
		in:   []string{"a", "aa", "aaa", "a$", "AAA", "aA.a", "a.a", "a", "aa", "aaa", "a$", "AAA", "aA.a"},
		want: []string{"AAA", "a", "aA.a", "aa", "aaa", "a$"},
	}, {
		in:   []string{"a.b", "aa.bb.cc", ".", "a$b", "aa", "a.", "a", "b.c.d", ".a", "", "a$", "a$", "a.b", "a", "a.bb", ""},
		want: []string{"", "a", "aa", "a$", "a$b", "b.c.d"},
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			mask := &fmpb.FieldMask{
				Paths: append([]string(nil), tt.in...),
			}
			mask.Normalize()
			got := mask.GetPaths()
			if diff := cmp.Diff(tt.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Normalize() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestIsValid(t *testing.T) {
	tests := []struct {
		message proto.Message
		paths   []string
		want    bool
	}{{
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{"no_such_field"},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{""},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths: []string{
			"optional_int32",
			"optional_int32",
			"optional_int64",
			"optional_uint32",
			"optional_uint64",
			"optional_sint32",
			"optional_sint64",
			"optional_fixed32",
			"optional_fixed64",
			"optional_sfixed32",
			"optional_sfixed64",
			"optional_float",
			"optional_double",
			"optional_bool",
			"optional_string",
			"optional_bytes",
			"OptionalGroup",
			"optional_nested_message",
			"optional_foreign_message",
			"optional_import_message",
			"optional_nested_enum",
			"optional_foreign_enum",
			"optional_import_enum",
			"repeated_int32",
			"repeated_int64",
			"repeated_uint32",
			"repeated_uint64",
			"repeated_sint32",
			"repeated_sint64",
			"repeated_fixed32",
			"repeated_fixed64",
			"repeated_sfixed32",
			"repeated_sfixed64",
			"repeated_float",
			"repeated_double",
			"repeated_bool",
			"repeated_string",
			"repeated_bytes",
			"RepeatedGroup",
			"repeated_nested_message",
			"repeated_foreign_message",
			"repeated_importmessage",
			"repeated_nested_enum",
			"repeated_foreign_enum",
			"repeated_importenum",
			"map_int32_int32",
			"map_int64_int64",
			"map_uint32_uint32",
			"map_uint64_uint64",
			"map_sint32_sint32",
			"map_sint64_sint64",
			"map_fixed32_fixed32",
			"map_fixed64_fixed64",
			"map_sfixed32_sfixed32",
			"map_sfixed64_sfixed64",
			"map_int32_float",
			"map_int32_double",
			"map_bool_bool",
			"map_string_string",
			"map_string_bytes",
			"map_string_nested_message",
			"map_string_nested_enum",
			"oneof_uint32",
			"oneof_nested_message",
			"oneof_string",
			"oneof_bytes",
			"oneof_bool",
			"oneof_uint64",
			"oneof_float",
			"oneof_double",
			"oneof_enum",
			"OneofGroup",
		},
		want: true,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths: []string{
			"optional_nested_message.a",
			"optional_nested_message.corecursive",
			"optional_nested_message.corecursive.optional_int32",
			"optional_nested_message.corecursive.optional_nested_message.corecursive.optional_nested_message.a",
			"OptionalGroup.a",
			"OptionalGroup.optional_nested_message",
			"OptionalGroup.optional_nested_message.corecursive",
			"oneof_nested_message.a",
			"oneof_nested_message.corecursive",
		},
		want: true,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{"repeated_nested_message.a"},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{"repeated_nested_message[0]"},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{"repeated_nested_message[0].a"},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{"map_string_nested_message.a"},
		want:    false,
	}, {
		message: (*testpb.TestAllTypes)(nil),
		paths:   []string{`map_string_nested_message["key"]`},
		want:    false,
	}, {
		message: (*testpb.TestAllExtensions)(nil),
		paths:   []string{"nested_string_extension"},
		want:    false,
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			mask := &fmpb.FieldMask{Paths: tt.paths}
			got := mask.IsValid(tt.message)
			if got != tt.want {
				t.Errorf("IsValid() returns %v want %v", got, tt.want)
			}
		})
	}
}
