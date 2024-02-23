// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package detrand

import "testing"

func Benchmark(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		binaryHash()
	}
}
