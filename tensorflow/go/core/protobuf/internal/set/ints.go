// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package set provides simple set data structures for uint64s.
package set

import "math/bits"

// int64s represents a set of integers within the range of 0..63.
type int64s uint64

func (bs *int64s) Len() int {
	return bits.OnesCount64(uint64(*bs))
}
func (bs *int64s) Has(n uint64) bool {
	return uint64(*bs)&(uint64(1)<<n) > 0
}
func (bs *int64s) Set(n uint64) {
	*(*uint64)(bs) |= uint64(1) << n
}
func (bs *int64s) Clear(n uint64) {
	*(*uint64)(bs) &^= uint64(1) << n
}

// Ints represents a set of integers within the range of 0..math.MaxUint64.
type Ints struct {
	lo int64s
	hi map[uint64]struct{}
}

func (bs *Ints) Len() int {
	return bs.lo.Len() + len(bs.hi)
}
func (bs *Ints) Has(n uint64) bool {
	if n < 64 {
		return bs.lo.Has(n)
	}
	_, ok := bs.hi[n]
	return ok
}
func (bs *Ints) Set(n uint64) {
	if n < 64 {
		bs.lo.Set(n)
		return
	}
	if bs.hi == nil {
		bs.hi = make(map[uint64]struct{})
	}
	bs.hi[n] = struct{}{}
}
func (bs *Ints) Clear(n uint64) {
	if n < 64 {
		bs.lo.Clear(n)
		return
	}
	delete(bs.hi, n)
}
