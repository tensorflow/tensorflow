// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fuzztest contains a common fuzzer test.
package fuzztest

import (
	"flag"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

var corpus = flag.String("corpus", "corpus", "directory containing the fuzzer corpus")

// Test executes a fuzz function for every entry in the corpus.
func Test(t *testing.T, fuzz func(b []byte) int) {
	dir, err := os.Open(*corpus)
	if err != nil {
		t.Fatal(err)
	}
	infos, err := dir.Readdir(0)
	if err != nil {
		t.Fatal(err)

	}
	var names []string
	for _, info := range infos {
		names = append(names, info.Name())
	}
	sort.Strings(names)
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			b, err := ioutil.ReadFile(filepath.Join(*corpus, name))
			if err != nil {
				t.Fatal(err)
			}
			b = b[:len(b):len(b)] // set cap to len
			fuzz(b)
		})
	}
}
