// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/protobuild"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/testing/protopack"
	"google.golang.org/protobuf/types/dynamicpb"

	legacypb "google.golang.org/protobuf/internal/testprotos/legacy"
	testpb "google.golang.org/protobuf/internal/testprotos/test"
	test3pb "google.golang.org/protobuf/internal/testprotos/test3"
)

type testMerge struct {
	desc  string
	dst   protobuild.Message
	src   protobuild.Message
	want  protobuild.Message // if dst and want are nil, want = src
	types []proto.Message
}

var testMerges = []testMerge{{
	desc: "clone a large message",
	src: protobuild.Message{
		"optional_int32":       1001,
		"optional_int64":       1002,
		"optional_uint32":      1003,
		"optional_uint64":      1004,
		"optional_sint32":      1005,
		"optional_sint64":      1006,
		"optional_fixed32":     1007,
		"optional_fixed64":     1008,
		"optional_sfixed32":    1009,
		"optional_sfixed64":    1010,
		"optional_float":       1011.5,
		"optional_double":      1012.5,
		"optional_bool":        true,
		"optional_string":      "string",
		"optional_bytes":       []byte("bytes"),
		"optional_nested_enum": 1,
		"optional_nested_message": protobuild.Message{
			"a": 100,
		},
		"repeated_int32":       []int32{1001, 2001},
		"repeated_int64":       []int64{1002, 2002},
		"repeated_uint32":      []uint32{1003, 2003},
		"repeated_uint64":      []uint64{1004, 2004},
		"repeated_sint32":      []int32{1005, 2005},
		"repeated_sint64":      []int64{1006, 2006},
		"repeated_fixed32":     []uint32{1007, 2007},
		"repeated_fixed64":     []uint64{1008, 2008},
		"repeated_sfixed32":    []int32{1009, 2009},
		"repeated_sfixed64":    []int64{1010, 2010},
		"repeated_float":       []float32{1011.5, 2011.5},
		"repeated_double":      []float64{1012.5, 2012.5},
		"repeated_bool":        []bool{true, false},
		"repeated_string":      []string{"foo", "bar"},
		"repeated_bytes":       []string{"FOO", "BAR"},
		"repeated_nested_enum": []string{"FOO", "BAR"},
		"repeated_nested_message": []protobuild.Message{
			{"a": 200},
			{"a": 300},
		},
	},
}, {
	desc: "clone maps",
	src: protobuild.Message{
		"map_int32_int32":       map[int32]int32{1056: 1156, 2056: 2156},
		"map_int64_int64":       map[int64]int64{1057: 1157, 2057: 2157},
		"map_uint32_uint32":     map[uint32]uint32{1058: 1158, 2058: 2158},
		"map_uint64_uint64":     map[uint64]uint64{1059: 1159, 2059: 2159},
		"map_sint32_sint32":     map[int32]int32{1060: 1160, 2060: 2160},
		"map_sint64_sint64":     map[int64]int64{1061: 1161, 2061: 2161},
		"map_fixed32_fixed32":   map[uint32]uint32{1062: 1162, 2062: 2162},
		"map_fixed64_fixed64":   map[uint64]uint64{1063: 1163, 2063: 2163},
		"map_sfixed32_sfixed32": map[int32]int32{1064: 1164, 2064: 2164},
		"map_sfixed64_sfixed64": map[int64]int64{1065: 1165, 2065: 2165},
		"map_int32_float":       map[int32]float32{1066: 1166.5, 2066: 2166.5},
		"map_int32_double":      map[int32]float64{1067: 1167.5, 2067: 2167.5},
		"map_bool_bool":         map[bool]bool{true: false, false: true},
		"map_string_string":     map[string]string{"69.1.key": "69.1.val", "69.2.key": "69.2.val"},
		"map_string_bytes":      map[string][]byte{"70.1.key": []byte("70.1.val"), "70.2.key": []byte("70.2.val")},
		"map_string_nested_message": map[string]protobuild.Message{
			"71.1.key": {"a": 1171},
			"71.2.key": {"a": 2171},
		},
		"map_string_nested_enum": map[string]string{"73.1.key": "FOO", "73.2.key": "BAR"},
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof uint32",
	src: protobuild.Message{
		"oneof_uint32": 1111,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof string",
	src: protobuild.Message{
		"oneof_string": "string",
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof bytes",
	src: protobuild.Message{
		"oneof_bytes": "bytes",
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof bool",
	src: protobuild.Message{
		"oneof_bool": true,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof uint64",
	src: protobuild.Message{
		"oneof_uint64": 100,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof float",
	src: protobuild.Message{
		"oneof_float": 100,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof double",
	src: protobuild.Message{
		"oneof_double": 1111,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof enum",
	src: protobuild.Message{
		"oneof_enum": 1,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof message",
	src: protobuild.Message{
		"oneof_nested_message": protobuild.Message{
			"a": 1,
		},
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "clone oneof group",
	src: protobuild.Message{
		"oneofgroup": protobuild.Message{
			"a": 1,
		},
	},
	types: []proto.Message{&testpb.TestAllTypes{}},
}, {
	desc: "merge bytes",
	dst: protobuild.Message{
		"optional_bytes":   []byte{1, 2, 3},
		"repeated_bytes":   [][]byte{{1, 2}, {3, 4}},
		"map_string_bytes": map[string][]byte{"alpha": {1, 2, 3}},
	},
	src: protobuild.Message{
		"optional_bytes":   []byte{4, 5, 6},
		"repeated_bytes":   [][]byte{{5, 6}, {7, 8}},
		"map_string_bytes": map[string][]byte{"alpha": {4, 5, 6}, "bravo": {1, 2, 3}},
	},
	want: protobuild.Message{
		"optional_bytes":   []byte{4, 5, 6},
		"repeated_bytes":   [][]byte{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
		"map_string_bytes": map[string][]byte{"alpha": {4, 5, 6}, "bravo": {1, 2, 3}},
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "merge singular fields",
	dst: protobuild.Message{
		"optional_int32":       1,
		"optional_int64":       1,
		"optional_uint32":      1,
		"optional_uint64":      1,
		"optional_sint32":      1,
		"optional_sint64":      1,
		"optional_fixed32":     1,
		"optional_fixed64":     1,
		"optional_sfixed32":    1,
		"optional_sfixed64":    1,
		"optional_float":       1,
		"optional_double":      1,
		"optional_bool":        false,
		"optional_string":      "1",
		"optional_bytes":       "1",
		"optional_nested_enum": 1,
		"optional_nested_message": protobuild.Message{
			"a": 1,
			"corecursive": protobuild.Message{
				"optional_int64": 1,
			},
		},
	},
	src: protobuild.Message{
		"optional_int32":       2,
		"optional_int64":       2,
		"optional_uint32":      2,
		"optional_uint64":      2,
		"optional_sint32":      2,
		"optional_sint64":      2,
		"optional_fixed32":     2,
		"optional_fixed64":     2,
		"optional_sfixed32":    2,
		"optional_sfixed64":    2,
		"optional_float":       2,
		"optional_double":      2,
		"optional_bool":        true,
		"optional_string":      "2",
		"optional_bytes":       "2",
		"optional_nested_enum": 2,
		"optional_nested_message": protobuild.Message{
			"a": 2,
			"corecursive": protobuild.Message{
				"optional_int64": 2,
			},
		},
	},
	want: protobuild.Message{
		"optional_int32":       2,
		"optional_int64":       2,
		"optional_uint32":      2,
		"optional_uint64":      2,
		"optional_sint32":      2,
		"optional_sint64":      2,
		"optional_fixed32":     2,
		"optional_fixed64":     2,
		"optional_sfixed32":    2,
		"optional_sfixed64":    2,
		"optional_float":       2,
		"optional_double":      2,
		"optional_bool":        true,
		"optional_string":      "2",
		"optional_bytes":       "2",
		"optional_nested_enum": 2,
		"optional_nested_message": protobuild.Message{
			"a": 2,
			"corecursive": protobuild.Message{
				"optional_int64": 2,
			},
		},
	},
}, {
	desc: "no merge of empty singular fields",
	dst: protobuild.Message{
		"optional_int32":       1,
		"optional_int64":       1,
		"optional_uint32":      1,
		"optional_uint64":      1,
		"optional_sint32":      1,
		"optional_sint64":      1,
		"optional_fixed32":     1,
		"optional_fixed64":     1,
		"optional_sfixed32":    1,
		"optional_sfixed64":    1,
		"optional_float":       1,
		"optional_double":      1,
		"optional_bool":        false,
		"optional_string":      "1",
		"optional_bytes":       "1",
		"optional_nested_enum": 1,
		"optional_nested_message": protobuild.Message{
			"a": 1,
			"corecursive": protobuild.Message{
				"optional_int64": 1,
			},
		},
	},
	src: protobuild.Message{
		"optional_nested_message": protobuild.Message{
			"a": 1,
			"corecursive": protobuild.Message{
				"optional_int32": 2,
			},
		},
	},
	want: protobuild.Message{
		"optional_int32":       1,
		"optional_int64":       1,
		"optional_uint32":      1,
		"optional_uint64":      1,
		"optional_sint32":      1,
		"optional_sint64":      1,
		"optional_fixed32":     1,
		"optional_fixed64":     1,
		"optional_sfixed32":    1,
		"optional_sfixed64":    1,
		"optional_float":       1,
		"optional_double":      1,
		"optional_bool":        false,
		"optional_string":      "1",
		"optional_bytes":       "1",
		"optional_nested_enum": 1,
		"optional_nested_message": protobuild.Message{
			"a": 1,
			"corecursive": protobuild.Message{
				"optional_int32": 2,
				"optional_int64": 1,
			},
		},
	},
}, {
	desc: "merge list fields",
	dst: protobuild.Message{
		"repeated_int32":       []int32{1, 2, 3},
		"repeated_int64":       []int64{1, 2, 3},
		"repeated_uint32":      []uint32{1, 2, 3},
		"repeated_uint64":      []uint64{1, 2, 3},
		"repeated_sint32":      []int32{1, 2, 3},
		"repeated_sint64":      []int64{1, 2, 3},
		"repeated_fixed32":     []uint32{1, 2, 3},
		"repeated_fixed64":     []uint64{1, 2, 3},
		"repeated_sfixed32":    []int32{1, 2, 3},
		"repeated_sfixed64":    []int64{1, 2, 3},
		"repeated_float":       []float32{1, 2, 3},
		"repeated_double":      []float64{1, 2, 3},
		"repeated_bool":        []bool{true},
		"repeated_string":      []string{"a", "b", "c"},
		"repeated_bytes":       []string{"a", "b", "c"},
		"repeated_nested_enum": []int{1, 2, 3},
		"repeated_nested_message": []protobuild.Message{
			{"a": 100},
			{"a": 200},
		},
	},
	src: protobuild.Message{
		"repeated_int32":       []int32{4, 5, 6},
		"repeated_int64":       []int64{4, 5, 6},
		"repeated_uint32":      []uint32{4, 5, 6},
		"repeated_uint64":      []uint64{4, 5, 6},
		"repeated_sint32":      []int32{4, 5, 6},
		"repeated_sint64":      []int64{4, 5, 6},
		"repeated_fixed32":     []uint32{4, 5, 6},
		"repeated_fixed64":     []uint64{4, 5, 6},
		"repeated_sfixed32":    []int32{4, 5, 6},
		"repeated_sfixed64":    []int64{4, 5, 6},
		"repeated_float":       []float32{4, 5, 6},
		"repeated_double":      []float64{4, 5, 6},
		"repeated_bool":        []bool{false},
		"repeated_string":      []string{"d", "e", "f"},
		"repeated_bytes":       []string{"d", "e", "f"},
		"repeated_nested_enum": []int{4, 5, 6},
		"repeated_nested_message": []protobuild.Message{
			{"a": 300},
			{"a": 400},
		},
	},
	want: protobuild.Message{
		"repeated_int32":       []int32{1, 2, 3, 4, 5, 6},
		"repeated_int64":       []int64{1, 2, 3, 4, 5, 6},
		"repeated_uint32":      []uint32{1, 2, 3, 4, 5, 6},
		"repeated_uint64":      []uint64{1, 2, 3, 4, 5, 6},
		"repeated_sint32":      []int32{1, 2, 3, 4, 5, 6},
		"repeated_sint64":      []int64{1, 2, 3, 4, 5, 6},
		"repeated_fixed32":     []uint32{1, 2, 3, 4, 5, 6},
		"repeated_fixed64":     []uint64{1, 2, 3, 4, 5, 6},
		"repeated_sfixed32":    []int32{1, 2, 3, 4, 5, 6},
		"repeated_sfixed64":    []int64{1, 2, 3, 4, 5, 6},
		"repeated_float":       []float32{1, 2, 3, 4, 5, 6},
		"repeated_double":      []float64{1, 2, 3, 4, 5, 6},
		"repeated_bool":        []bool{true, false},
		"repeated_string":      []string{"a", "b", "c", "d", "e", "f"},
		"repeated_bytes":       []string{"a", "b", "c", "d", "e", "f"},
		"repeated_nested_enum": []int{1, 2, 3, 4, 5, 6},
		"repeated_nested_message": []protobuild.Message{
			{"a": 100},
			{"a": 200},
			{"a": 300},
			{"a": 400},
		},
	},
}, {
	desc: "merge map fields",
	dst: protobuild.Message{
		"map_int32_int32":       map[int]int{1: 1, 3: 1},
		"map_int64_int64":       map[int]int{1: 1, 3: 1},
		"map_uint32_uint32":     map[int]int{1: 1, 3: 1},
		"map_uint64_uint64":     map[int]int{1: 1, 3: 1},
		"map_sint32_sint32":     map[int]int{1: 1, 3: 1},
		"map_sint64_sint64":     map[int]int{1: 1, 3: 1},
		"map_fixed32_fixed32":   map[int]int{1: 1, 3: 1},
		"map_fixed64_fixed64":   map[int]int{1: 1, 3: 1},
		"map_sfixed32_sfixed32": map[int]int{1: 1, 3: 1},
		"map_sfixed64_sfixed64": map[int]int{1: 1, 3: 1},
		"map_int32_float":       map[int]int{1: 1, 3: 1},
		"map_int32_double":      map[int]int{1: 1, 3: 1},
		"map_bool_bool":         map[bool]bool{true: true},
		"map_string_string":     map[string]string{"a": "1", "ab": "1"},
		"map_string_bytes":      map[string]string{"a": "1", "ab": "1"},
		"map_string_nested_message": map[string]protobuild.Message{
			"a": {"a": 1},
			"ab": {
				"a": 1,
				"corecursive": protobuild.Message{
					"map_int32_int32": map[int]int{1: 1, 3: 1},
				},
			},
		},
		"map_string_nested_enum": map[string]int{"a": 1, "ab": 1},
	},
	src: protobuild.Message{
		"map_int32_int32":       map[int]int{2: 2, 3: 2},
		"map_int64_int64":       map[int]int{2: 2, 3: 2},
		"map_uint32_uint32":     map[int]int{2: 2, 3: 2},
		"map_uint64_uint64":     map[int]int{2: 2, 3: 2},
		"map_sint32_sint32":     map[int]int{2: 2, 3: 2},
		"map_sint64_sint64":     map[int]int{2: 2, 3: 2},
		"map_fixed32_fixed32":   map[int]int{2: 2, 3: 2},
		"map_fixed64_fixed64":   map[int]int{2: 2, 3: 2},
		"map_sfixed32_sfixed32": map[int]int{2: 2, 3: 2},
		"map_sfixed64_sfixed64": map[int]int{2: 2, 3: 2},
		"map_int32_float":       map[int]int{2: 2, 3: 2},
		"map_int32_double":      map[int]int{2: 2, 3: 2},
		"map_bool_bool":         map[bool]bool{false: false},
		"map_string_string":     map[string]string{"b": "2", "ab": "2"},
		"map_string_bytes":      map[string]string{"b": "2", "ab": "2"},
		"map_string_nested_message": map[string]protobuild.Message{
			"b": {"a": 2},
			"ab": {
				"a": 2,
				"corecursive": protobuild.Message{
					"map_int32_int32": map[int]int{2: 2, 3: 2},
				},
			},
		},
		"map_string_nested_enum": map[string]int{"b": 2, "ab": 2},
	},
	want: protobuild.Message{
		"map_int32_int32":       map[int]int{1: 1, 2: 2, 3: 2},
		"map_int64_int64":       map[int]int{1: 1, 2: 2, 3: 2},
		"map_uint32_uint32":     map[int]int{1: 1, 2: 2, 3: 2},
		"map_uint64_uint64":     map[int]int{1: 1, 2: 2, 3: 2},
		"map_sint32_sint32":     map[int]int{1: 1, 2: 2, 3: 2},
		"map_sint64_sint64":     map[int]int{1: 1, 2: 2, 3: 2},
		"map_fixed32_fixed32":   map[int]int{1: 1, 2: 2, 3: 2},
		"map_fixed64_fixed64":   map[int]int{1: 1, 2: 2, 3: 2},
		"map_sfixed32_sfixed32": map[int]int{1: 1, 2: 2, 3: 2},
		"map_sfixed64_sfixed64": map[int]int{1: 1, 2: 2, 3: 2},
		"map_int32_float":       map[int]int{1: 1, 2: 2, 3: 2},
		"map_int32_double":      map[int]int{1: 1, 2: 2, 3: 2},
		"map_bool_bool":         map[bool]bool{true: true, false: false},
		"map_string_string":     map[string]string{"a": "1", "b": "2", "ab": "2"},
		"map_string_bytes":      map[string]string{"a": "1", "b": "2", "ab": "2"},
		"map_string_nested_message": map[string]protobuild.Message{
			"a": {"a": 1},
			"b": {"a": 2},
			"ab": {
				"a": 2,
				"corecursive": protobuild.Message{
					// The map item "ab" was entirely replaced, so
					// this does not contain 1:1 from dst.
					"map_int32_int32": map[int]int{2: 2, 3: 2},
				},
			},
		},
		"map_string_nested_enum": map[string]int{"a": 1, "b": 2, "ab": 2},
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "merge oneof message fields",
	dst: protobuild.Message{
		"oneof_nested_message": protobuild.Message{
			"a": 100,
		},
	},
	src: protobuild.Message{
		"oneof_nested_message": protobuild.Message{
			"corecursive": protobuild.Message{
				"optional_int64": 1000,
			},
		},
	},
	want: protobuild.Message{
		"oneof_nested_message": protobuild.Message{
			"a": 100,
			"corecursive": protobuild.Message{
				"optional_int64": 1000,
			},
		},
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "merge oneof scalar fields",
	dst: protobuild.Message{
		"oneof_uint32": 100,
	},
	src: protobuild.Message{
		"oneof_float": 3.14152,
	},
	want: protobuild.Message{
		"oneof_float": 3.14152,
	},
	types: []proto.Message{&testpb.TestAllTypes{}, &test3pb.TestAllTypes{}},
}, {
	desc: "merge unknown fields",
	dst: protobuild.Message{
		protobuild.Unknown: protopack.Message{
			protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
		}.Marshal(),
	},
	src: protobuild.Message{
		protobuild.Unknown: protopack.Message{
			protopack.Tag{Number: 500000, Type: protopack.VarintType}, protopack.Svarint(-50),
		}.Marshal(),
	},
	want: protobuild.Message{
		protobuild.Unknown: protopack.Message{
			protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
			protopack.Tag{Number: 500000, Type: protopack.VarintType}, protopack.Svarint(-50),
		}.Marshal(),
	},
}, {
	desc: "clone legacy message",
	src: protobuild.Message{"f1": protobuild.Message{
		"optional_int32":        1,
		"optional_int64":        1,
		"optional_uint32":       1,
		"optional_uint64":       1,
		"optional_sint32":       1,
		"optional_sint64":       1,
		"optional_fixed32":      1,
		"optional_fixed64":      1,
		"optional_sfixed32":     1,
		"optional_sfixed64":     1,
		"optional_float":        1,
		"optional_double":       1,
		"optional_bool":         true,
		"optional_string":       "string",
		"optional_bytes":        "bytes",
		"optional_sibling_enum": 1,
		"optional_sibling_message": protobuild.Message{
			"f1": "value",
		},
		"repeated_int32":        []int32{1},
		"repeated_int64":        []int64{1},
		"repeated_uint32":       []uint32{1},
		"repeated_uint64":       []uint64{1},
		"repeated_sint32":       []int32{1},
		"repeated_sint64":       []int64{1},
		"repeated_fixed32":      []uint32{1},
		"repeated_fixed64":      []uint64{1},
		"repeated_sfixed32":     []int32{1},
		"repeated_sfixed64":     []int64{1},
		"repeated_float":        []float32{1},
		"repeated_double":       []float64{1},
		"repeated_bool":         []bool{true},
		"repeated_string":       []string{"string"},
		"repeated_bytes":        []string{"bytes"},
		"repeated_sibling_enum": []int{1},
		"repeated_sibling_message": []protobuild.Message{
			{"f1": "1"},
		},
		"map_bool_int32":    map[bool]int{true: 1},
		"map_bool_int64":    map[bool]int{true: 1},
		"map_bool_uint32":   map[bool]int{true: 1},
		"map_bool_uint64":   map[bool]int{true: 1},
		"map_bool_sint32":   map[bool]int{true: 1},
		"map_bool_sint64":   map[bool]int{true: 1},
		"map_bool_fixed32":  map[bool]int{true: 1},
		"map_bool_fixed64":  map[bool]int{true: 1},
		"map_bool_sfixed32": map[bool]int{true: 1},
		"map_bool_sfixed64": map[bool]int{true: 1},
		"map_bool_float":    map[bool]int{true: 1},
		"map_bool_double":   map[bool]int{true: 1},
		"map_bool_bool":     map[bool]bool{true: false},
		"map_bool_string":   map[bool]string{true: "1"},
		"map_bool_bytes":    map[bool]string{true: "1"},
		"map_bool_sibling_message": map[bool]protobuild.Message{
			true: {"f1": "1"},
		},
		"map_bool_sibling_enum": map[bool]int{true: 1},
		"oneof_sibling_message": protobuild.Message{
			"f1": "1",
		},
	}},
	types: []proto.Message{&legacypb.Legacy{}},
}}

func TestMerge(t *testing.T) {
	for _, tt := range testMerges {
		for _, mt := range templateMessages(tt.types...) {
			t.Run(fmt.Sprintf("%s (%v)", tt.desc, mt.Descriptor().FullName()), func(t *testing.T) {
				dst := mt.New().Interface()
				tt.dst.Build(dst.ProtoReflect())

				src := mt.New().Interface()
				tt.src.Build(src.ProtoReflect())

				want := mt.New().Interface()
				if tt.dst == nil && tt.want == nil {
					tt.src.Build(want.ProtoReflect())
				} else {
					tt.want.Build(want.ProtoReflect())
				}

				// Merge should be semantically equivalent to unmarshaling the
				// encoded form of src into the current dst.
				b1, err := proto.MarshalOptions{AllowPartial: true}.Marshal(dst)
				if err != nil {
					t.Fatalf("Marshal(dst) error: %v", err)
				}
				b2, err := proto.MarshalOptions{AllowPartial: true}.Marshal(src)
				if err != nil {
					t.Fatalf("Marshal(src) error: %v", err)
				}
				unmarshaled := dst.ProtoReflect().New().Interface()
				err = proto.UnmarshalOptions{AllowPartial: true}.Unmarshal(append(b1, b2...), unmarshaled)
				if err != nil {
					t.Fatalf("Unmarshal() error: %v", err)
				}
				if !proto.Equal(unmarshaled, want) {
					t.Fatalf("Unmarshal(Marshal(dst)+Marshal(src)) mismatch:\n got %v\nwant %v\ndiff (-want,+got):\n%v", unmarshaled, want, cmp.Diff(want, unmarshaled, protocmp.Transform()))
				}

				// Test heterogeneous MessageTypes by merging into a
				// dynamic message.
				ddst := dynamicpb.NewMessage(mt.Descriptor())
				tt.dst.Build(ddst.ProtoReflect())
				proto.Merge(ddst, src)
				if !proto.Equal(ddst, want) {
					t.Fatalf("Merge() into dynamic message mismatch:\n got %v\nwant %v\ndiff (-want,+got):\n%v", ddst, want, cmp.Diff(want, ddst, protocmp.Transform()))
				}

				proto.Merge(dst, src)
				if !proto.Equal(dst, want) {
					t.Fatalf("Merge() mismatch:\n got %v\nwant %v\ndiff (-want,+got):\n%v", dst, want, cmp.Diff(want, dst, protocmp.Transform()))
				}
				mutateValue(protoreflect.ValueOfMessage(src.ProtoReflect()))
				if !proto.Equal(dst, want) {
					t.Fatalf("mutation observed after modifying source:\n got %v\nwant %v\ndiff (-want,+got):\n%v", dst, want, cmp.Diff(want, dst, protocmp.Transform()))
				}
			})
		}
	}
}

func TestMergeFromNil(t *testing.T) {
	dst := &testpb.TestAllTypes{}
	proto.Merge(dst, (*testpb.TestAllTypes)(nil))
	if !proto.Equal(dst, &testpb.TestAllTypes{}) {
		t.Errorf("destination should be empty after merging from nil message; got:\n%v", prototext.Format(dst))
	}
}

// TestMergeAberrant tests inputs that are beyond the protobuf data model.
// Just because there is a test for the current behavior does not mean that
// this will behave the same way in the future.
func TestMergeAberrant(t *testing.T) {
	tests := []struct {
		label string
		dst   proto.Message
		src   proto.Message
		check func(proto.Message) bool
	}{{
		label: "Proto2EmptyBytes",
		dst:   &testpb.TestAllTypes{OptionalBytes: nil},
		src:   &testpb.TestAllTypes{OptionalBytes: []byte{}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).OptionalBytes != nil
		},
	}, {
		label: "Proto3EmptyBytes",
		dst:   &test3pb.TestAllTypes{SingularBytes: nil},
		src:   &test3pb.TestAllTypes{SingularBytes: []byte{}},
		check: func(m proto.Message) bool {
			return m.(*test3pb.TestAllTypes).SingularBytes == nil
		},
	}, {
		label: "EmptyList",
		dst:   &testpb.TestAllTypes{RepeatedInt32: nil},
		src:   &testpb.TestAllTypes{RepeatedInt32: []int32{}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).RepeatedInt32 == nil
		},
	}, {
		label: "ListWithNilBytes",
		dst:   &testpb.TestAllTypes{RepeatedBytes: nil},
		src:   &testpb.TestAllTypes{RepeatedBytes: [][]byte{nil}},
		check: func(m proto.Message) bool {
			return reflect.DeepEqual(m.(*testpb.TestAllTypes).RepeatedBytes, [][]byte{{}})
		},
	}, {
		label: "ListWithEmptyBytes",
		dst:   &testpb.TestAllTypes{RepeatedBytes: nil},
		src:   &testpb.TestAllTypes{RepeatedBytes: [][]byte{{}}},
		check: func(m proto.Message) bool {
			return reflect.DeepEqual(m.(*testpb.TestAllTypes).RepeatedBytes, [][]byte{{}})
		},
	}, {
		label: "ListWithNilMessage",
		dst:   &testpb.TestAllTypes{RepeatedNestedMessage: nil},
		src:   &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{nil}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).RepeatedNestedMessage[0] != nil
		},
	}, {
		label: "EmptyMap",
		dst:   &testpb.TestAllTypes{MapStringString: nil},
		src:   &testpb.TestAllTypes{MapStringString: map[string]string{}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).MapStringString == nil
		},
	}, {
		label: "MapWithNilBytes",
		dst:   &testpb.TestAllTypes{MapStringBytes: nil},
		src:   &testpb.TestAllTypes{MapStringBytes: map[string][]byte{"k": nil}},
		check: func(m proto.Message) bool {
			return reflect.DeepEqual(m.(*testpb.TestAllTypes).MapStringBytes, map[string][]byte{"k": {}})
		},
	}, {
		label: "MapWithEmptyBytes",
		dst:   &testpb.TestAllTypes{MapStringBytes: nil},
		src:   &testpb.TestAllTypes{MapStringBytes: map[string][]byte{"k": {}}},
		check: func(m proto.Message) bool {
			return reflect.DeepEqual(m.(*testpb.TestAllTypes).MapStringBytes, map[string][]byte{"k": {}})
		},
	}, {
		label: "MapWithNilMessage",
		dst:   &testpb.TestAllTypes{MapStringNestedMessage: nil},
		src:   &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": nil}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).MapStringNestedMessage["k"] != nil
		},
	}, {
		label: "OneofWithTypedNilWrapper",
		dst:   &testpb.TestAllTypes{OneofField: nil},
		src:   &testpb.TestAllTypes{OneofField: (*testpb.TestAllTypes_OneofNestedMessage)(nil)},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).OneofField == nil
		},
	}, {
		label: "OneofWithNilMessage",
		dst:   &testpb.TestAllTypes{OneofField: nil},
		src:   &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofNestedMessage{OneofNestedMessage: nil}},
		check: func(m proto.Message) bool {
			return m.(*testpb.TestAllTypes).OneofField.(*testpb.TestAllTypes_OneofNestedMessage).OneofNestedMessage != nil
		},
		// TODO: extension, nil message
		// TODO: repeated extension, nil
		// TODO: extension bytes
		// TODO: repeated extension, nil message
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			var pass bool
			func() {
				defer func() { recover() }()
				proto.Merge(tt.dst, tt.src)
				pass = tt.check(tt.dst)
			}()
			if !pass {
				t.Error("check failed")
			}
		})
	}
}

func TestMergeRace(t *testing.T) {
	dst := new(testpb.TestAllTypes)
	srcs := []*testpb.TestAllTypes{
		{OptionalInt32: proto.Int32(1)},
		{OptionalString: proto.String("hello")},
		{RepeatedInt32: []int32{2, 3, 4}},
		{RepeatedString: []string{"goodbye"}},
		{MapStringString: map[string]string{"key": "value"}},
		{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
			A: proto.Int32(5),
		}},
		func() *testpb.TestAllTypes {
			m := new(testpb.TestAllTypes)
			m.ProtoReflect().SetUnknown(protopack.Message{
				protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
			}.Marshal())
			return m
		}(),
	}

	// It should be safe to concurrently merge non-overlapping fields.
	var wg sync.WaitGroup
	defer wg.Wait()
	for _, src := range srcs {
		wg.Add(1)
		go func(src proto.Message) {
			defer wg.Done()
			proto.Merge(dst, src)
		}(src)
	}
}

func TestMergeSelf(t *testing.T) {
	got := &testpb.TestAllTypes{
		OptionalInt32:   proto.Int32(1),
		OptionalString:  proto.String("hello"),
		RepeatedInt32:   []int32{2, 3, 4},
		RepeatedString:  []string{"goodbye"},
		MapStringString: map[string]string{"key": "value"},
		OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
			A: proto.Int32(5),
		},
	}
	got.ProtoReflect().SetUnknown(protopack.Message{
		protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
	}.Marshal())
	proto.Merge(got, got)

	// The main impact of merging to self is that repeated fields and
	// unknown fields are doubled.
	want := &testpb.TestAllTypes{
		OptionalInt32:   proto.Int32(1),
		OptionalString:  proto.String("hello"),
		RepeatedInt32:   []int32{2, 3, 4, 2, 3, 4},
		RepeatedString:  []string{"goodbye", "goodbye"},
		MapStringString: map[string]string{"key": "value"},
		OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
			A: proto.Int32(5),
		},
	}
	want.ProtoReflect().SetUnknown(protopack.Message{
		protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
		protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Svarint(-5),
	}.Marshal())

	if !proto.Equal(got, want) {
		t.Errorf("Equal mismatch:\ngot  %v\nwant %v", got, want)
	}
}

func TestClone(t *testing.T) {
	want := &testpb.TestAllTypes{
		OptionalInt32: proto.Int32(1),
	}
	got := proto.Clone(want).(*testpb.TestAllTypes)
	if !proto.Equal(got, want) {
		t.Errorf("Clone(src) != src:\n got %v\nwant %v", got, want)
	}
}

// mutateValue changes a Value, returning a new value.
//
// For scalar values, it returns a value different from the input.
// For Message, List, and Map values, it mutates the input and returns it.
func mutateValue(v protoreflect.Value) protoreflect.Value {
	switch v := v.Interface().(type) {
	case bool:
		return protoreflect.ValueOfBool(!v)
	case protoreflect.EnumNumber:
		return protoreflect.ValueOfEnum(v + 1)
	case int32:
		return protoreflect.ValueOfInt32(v + 1)
	case int64:
		return protoreflect.ValueOfInt64(v + 1)
	case uint32:
		return protoreflect.ValueOfUint32(v + 1)
	case uint64:
		return protoreflect.ValueOfUint64(v + 1)
	case float32:
		return protoreflect.ValueOfFloat32(v + 1)
	case float64:
		return protoreflect.ValueOfFloat64(v + 1)
	case []byte:
		for i := range v {
			v[i]++
		}
		return protoreflect.ValueOfBytes(v)
	case string:
		return protoreflect.ValueOfString("_" + v)
	case protoreflect.Message:
		v.Range(func(fd protoreflect.FieldDescriptor, val protoreflect.Value) bool {
			v.Set(fd, mutateValue(val))
			return true
		})
		return protoreflect.ValueOfMessage(v)
	case protoreflect.List:
		for i := 0; i < v.Len(); i++ {
			v.Set(i, mutateValue(v.Get(i)))
		}
		return protoreflect.ValueOfList(v)
	case protoreflect.Map:
		v.Range(func(mk protoreflect.MapKey, mv protoreflect.Value) bool {
			v.Set(mk, mutateValue(mv))
			return true
		})
		return protoreflect.ValueOfMap(v)
	default:
		panic(fmt.Sprintf("unknown value type %T", v))
	}
}
