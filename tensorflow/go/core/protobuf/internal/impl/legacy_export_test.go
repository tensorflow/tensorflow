// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"bytes"
	"compress/gzip"
	"io/ioutil"
	"math"
	"strings"
	"testing"
)

func TestCompressGZIP(t *testing.T) {
	tests := []string{
		"",
		"a",
		"ab",
		"abc",
		strings.Repeat("a", math.MaxUint16-1),
		strings.Repeat("b", math.MaxUint16),
		strings.Repeat("c", math.MaxUint16+1),
		strings.Repeat("abcdefghijklmnopqrstuvwxyz", math.MaxUint16-13),
	}
	for _, want := range tests {
		rb := bytes.NewReader(Export{}.CompressGZIP([]byte(want)))
		zr, err := gzip.NewReader(rb)
		if err != nil {
			t.Errorf("unexpected gzip.NewReader error: %v", err)
		}
		b, err := ioutil.ReadAll(zr)
		if err != nil {
			t.Errorf("unexpected ioutil.ReadAll error: %v", err)
		}
		if got := string(b); got != want {
			t.Errorf("output mismatch: got %q, want %q", got, want)
		}
	}
}
