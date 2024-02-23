// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import "testing"

func TestSourcePathString(t *testing.T) {
	tests := []struct {
		in   SourcePath
		want string
	}{
		{nil, ""},
		{SourcePath{}, ""},
		{SourcePath{0}, ".0"},
		{SourcePath{1}, ".name"},
		{SourcePath{1, 1}, ".name.1"},
		{SourcePath{1, 1, -2, 3}, ".name.1.-2.3"},
		{SourcePath{3}, ".dependency"},
		{SourcePath{3, 0}, ".dependency[0]"},
		{SourcePath{3, -1}, ".dependency.-1"},
		{SourcePath{3, 1, 2}, ".dependency[1].2"},
		{SourcePath{4}, ".message_type"},
		{SourcePath{4, 0}, ".message_type[0]"},
		{SourcePath{4, -1}, ".message_type.-1"},
		{SourcePath{4, 1, 0}, ".message_type[1].0"},
		{SourcePath{4, 1, 1}, ".message_type[1].name"},
	}
	for _, tt := range tests {
		if got := tt.in.String(); got != tt.want {
			t.Errorf("SourcePath(%d).String() = %v, want %v", tt.in, got, tt.want)
		}
	}
}
