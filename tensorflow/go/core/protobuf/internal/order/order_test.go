// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package order

import (
	"math/rand"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type fieldDesc struct {
	index      int
	name       protoreflect.FullName
	number     protoreflect.FieldNumber
	extension  bool
	oneofIndex int // non-zero means within oneof; negative means synthetic
	protoreflect.FieldDescriptor
}

func (d fieldDesc) Index() int                       { return d.index }
func (d fieldDesc) Name() protoreflect.Name          { return d.name.Name() }
func (d fieldDesc) FullName() protoreflect.FullName  { return d.name }
func (d fieldDesc) Number() protoreflect.FieldNumber { return d.number }
func (d fieldDesc) IsExtension() bool                { return d.extension }
func (d fieldDesc) ContainingOneof() protoreflect.OneofDescriptor {
	switch {
	case d.oneofIndex < 0:
		return oneofDesc{index: -d.oneofIndex, synthetic: true}
	case d.oneofIndex > 0:
		return oneofDesc{index: +d.oneofIndex, synthetic: false}
	default:
		return nil
	}
}

type oneofDesc struct {
	index     int
	synthetic bool
	protoreflect.OneofDescriptor
}

func (d oneofDesc) Index() int        { return d.index }
func (d oneofDesc) IsSynthetic() bool { return d.synthetic }

func TestFieldOrder(t *testing.T) {
	tests := []struct {
		label  string
		order  FieldOrder
		fields []fieldDesc
	}{{
		label: "LegacyFieldOrder",
		order: LegacyFieldOrder,
		fields: []fieldDesc{
			// Extension fields sorted first by field number.
			{number: 2, extension: true},
			{number: 4, extension: true},
			{number: 100, extension: true},
			{number: 120, extension: true},

			// Non-extension fields that are not within a oneof
			// sorted next by field number.
			{number: 1},
			{number: 5, oneofIndex: -10}, // synthetic oneof
			{number: 10},
			{number: 11, oneofIndex: -9}, // synthetic oneof
			{number: 12},

			// Non-synthetic oneofs sorted last by index.
			{number: 13, oneofIndex: 4},
			{number: 3, oneofIndex: 5},
			{number: 9, oneofIndex: 5},
			{number: 7, oneofIndex: 8},
		},
	}, {
		label: "NumberFieldOrder",
		order: NumberFieldOrder,
		fields: []fieldDesc{
			{number: 1, index: 5, name: "c"},
			{number: 2, index: 2, name: "b"},
			{number: 3, index: 3, name: "d"},
			{number: 5, index: 1, name: "a"},
			{number: 7, index: 7, name: "e"},
		},
	}, {
		label: "IndexNameFieldOrder",
		order: IndexNameFieldOrder,
		fields: []fieldDesc{
			// Non-extension fields sorted first by index.
			{index: 0, number: 5, name: "c"},
			{index: 2, number: 2, name: "a"},
			{index: 4, number: 4, name: "b"},
			{index: 7, number: 6, name: "d"},

			// Extension fields sorted last by full name.
			{index: 3, number: 1, name: "d.a", extension: true},
			{index: 5, number: 3, name: "e", extension: true},
			{index: 1, number: 7, name: "g", extension: true},
		},
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			want := tt.fields
			got := append([]fieldDesc(nil), want...)
			for i, j := range rand.Perm(len(got)) {
				got[i], got[j] = got[j], got[i]
			}
			sort.Slice(got, func(i, j int) bool {
				return tt.order(got[i], got[j])
			})
			if diff := cmp.Diff(want, got,
				cmp.Comparer(func(x, y fieldDesc) bool { return x == y }),
			); diff != "" {
				t.Errorf("order mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestKeyOrder(t *testing.T) {
	tests := []struct {
		label string
		order KeyOrder
		keys  []interface{}
	}{{
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{false, true},
	}, {
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{int32(-100), int32(-99), int32(-10), int32(-9), int32(-1), int32(0), int32(+1), int32(+9), int32(+10), int32(+99), int32(+100)},
	}, {
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{int64(-100), int64(-99), int64(-10), int64(-9), int64(-1), int64(0), int64(+1), int64(+9), int64(+10), int64(+99), int64(+100)},
	}, {
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{uint32(0), uint32(1), uint32(9), uint32(10), uint32(99), uint32(100)},
	}, {
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{uint64(0), uint64(1), uint64(9), uint64(10), uint64(99), uint64(100)},
	}, {
		label: "GenericKeyOrder",
		order: GenericKeyOrder,
		keys:  []interface{}{"", "a", "aa", "ab", "ba", "bb", "\u0080", "\u0080\u0081", "\u0082\u0080"},
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			var got, want []protoreflect.MapKey
			for _, v := range tt.keys {
				want = append(want, protoreflect.ValueOf(v).MapKey())
			}
			got = append(got, want...)
			for i, j := range rand.Perm(len(got)) {
				got[i], got[j] = got[j], got[i]
			}
			sort.Slice(got, func(i, j int) bool {
				return tt.order(got[i], got[j])
			})
			if diff := cmp.Diff(want, got, cmp.Transformer("", protoreflect.MapKey.Interface)); diff != "" {
				t.Errorf("order mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
