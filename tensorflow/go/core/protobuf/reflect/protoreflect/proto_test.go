// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import "testing"

func TestNameIsValid(t *testing.T) {
	tests := []struct {
		in   Name
		want bool
	}{
		{"", false},
		{"a", true},
		{".", false},
		{"_", true}, // odd, but permitted by protoc
		{".foo", false},
		{"foo.", false},
		{"foo", true},
		{"one1_two2_three3", true},
		{"1one", false},
	}

	for _, tt := range tests {
		if got := tt.in.IsValid(); got != tt.want {
			t.Errorf("Name(%q).IsValid() = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestFullNameIsValid(t *testing.T) {
	tests := []struct {
		in   FullName
		want bool
	}{
		{"", false},
		{"a", true},
		{"a.b", true},
		{"a.b.c", true},
		{".", false},
		{"_._._", true}, // odd, but permitted by protoc
		{".foo", false},
		{"foo.", false},
		{"foo", true},
		{"one1_two2_three3", true},
		{"one1.two2.three3", true},
		{".one1.two2.three3", false},
		{"one1.two2.three3.", false},
		{"foo.1one", false},
	}

	for _, tt := range tests {
		if got := tt.in.IsValid(); got != tt.want {
			t.Errorf("Name(%q).IsValid() = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestNameAppend(t *testing.T) {
	tests := []FullName{
		"",
		"a",
		"a.b",
		"a.b.c",
		"one1.two2.three3",
	}

	for _, tt := range tests {
		if got := tt.Parent().Append(tt.Name()); got != tt {
			t.Errorf("FullName.Parent().Append(FullName.Name()) = %q, want %q", got, tt)
		}
	}
}

var sink bool

func BenchmarkFullNameIsValid(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = FullName("google.protobuf.Any").IsValid()
	}
}
