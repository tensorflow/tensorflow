// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Program generate-corpus generates a seed corpus for the fuzzers.
//
// This command is not run automatically because its output is not stable.
// It's present in source control mainly as documentation of where the seed
// corpus came from.
package main

import (
	"crypto/sha1"
	"fmt"
	"io/ioutil"
	"log"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	fuzzpb "google.golang.org/protobuf/internal/testprotos/fuzz"
	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

var messages = []proto.Message{
	&fuzzpb.Fuzz{
		TestAllTypes: &testpb.TestAllTypes{
			OptionalInt32:      proto.Int32(1001),
			OptionalInt64:      proto.Int64(1002),
			OptionalUint32:     proto.Uint32(1003),
			OptionalUint64:     proto.Uint64(1004),
			OptionalSint32:     proto.Int32(1005),
			OptionalSint64:     proto.Int64(1006),
			OptionalFixed32:    proto.Uint32(1007),
			OptionalFixed64:    proto.Uint64(1008),
			OptionalSfixed32:   proto.Int32(1009),
			OptionalSfixed64:   proto.Int64(1010),
			OptionalFloat:      proto.Float32(1011.5),
			OptionalDouble:     proto.Float64(1012.5),
			OptionalBool:       proto.Bool(true),
			OptionalString:     proto.String("string"),
			OptionalBytes:      []byte("bytes"),
			OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum(),
			Optionalgroup: &testpb.TestAllTypes_OptionalGroup{
				A: proto.Int32(1017),
			},
			OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
				A: proto.Int32(42),
				Corecursive: &testpb.TestAllTypes{
					OptionalInt32: proto.Int32(43),
				},
			},
			RepeatedInt32:    []int32{1001, 2001},
			RepeatedInt64:    []int64{1002, 2002},
			RepeatedUint32:   []uint32{1003, 2003},
			RepeatedUint64:   []uint64{1004, 2004},
			RepeatedSint32:   []int32{1005, 2005},
			RepeatedSint64:   []int64{1006, 2006},
			RepeatedFixed32:  []uint32{1007, 2007},
			RepeatedFixed64:  []uint64{1008, 2008},
			RepeatedSfixed32: []int32{1009, 2009},
			RepeatedSfixed64: []int64{1010, 2010},
			RepeatedFloat:    []float32{1011.5, 2011.5},
			RepeatedDouble:   []float64{1012.5, 2012.5},
			RepeatedBool:     []bool{true, false},
			RepeatedString:   []string{"foo", "bar"},
			RepeatedBytes:    [][]byte{[]byte("FOO"), []byte("BAR")},
			RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{
				testpb.TestAllTypes_FOO,
				testpb.TestAllTypes_BAR,
			},
			RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				nil,
				{A: proto.Int32(2)},
			},
			Repeatedgroup: []*testpb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1017)},
				nil,
				{A: proto.Int32(2017)},
			},
			MapInt32Int32:       map[int32]int32{1056: 1156, 2056: 2156},
			MapInt64Int64:       map[int64]int64{1057: 1157, 2057: 2157},
			MapUint32Uint32:     map[uint32]uint32{1058: 1158, 2058: 2158},
			MapUint64Uint64:     map[uint64]uint64{1059: 1159, 2059: 2159},
			MapSint32Sint32:     map[int32]int32{1060: 1160, 2060: 2160},
			MapSint64Sint64:     map[int64]int64{1061: 1161, 2061: 2161},
			MapFixed32Fixed32:   map[uint32]uint32{1062: 1162, 2062: 2162},
			MapFixed64Fixed64:   map[uint64]uint64{1063: 1163, 2063: 2163},
			MapSfixed32Sfixed32: map[int32]int32{1064: 1164, 2064: 2164},
			MapSfixed64Sfixed64: map[int64]int64{1065: 1165, 2065: 2165},
			MapInt32Float:       map[int32]float32{1066: 1166.5, 2066: 2166.5},
			MapInt32Double:      map[int32]float64{1067: 1167.5, 2067: 2167.5},
			MapBoolBool:         map[bool]bool{true: false, false: true},
			MapStringString:     map[string]string{"69.1.key": "69.1.val", "69.2.key": "69.2.val"},
			MapStringBytes:      map[string][]byte{"70.1.key": []byte("70.1.val"), "70.2.key": []byte("70.2.val")},
			MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{
				"71.1.key": {A: proto.Int32(1171)},
				"71.2.key": {A: proto.Int32(2171)},
			},
			MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{
				"73.1.key": testpb.TestAllTypes_FOO,
				"73.2.key": testpb.TestAllTypes_BAR,
			},
			OneofField: &testpb.TestAllTypes_OneofUint32{1111},
		},
	},
}

func main() {
	for _, m := range messages {
		wire, err := proto.Marshal(m)
		if err != nil {
			log.Fatal(err)
		}
		if err := ioutil.WriteFile(fmt.Sprintf("internal/fuzz/wirefuzz/corpus/%x", sha1.Sum(wire)), wire, 0777); err != nil {
			log.Fatal(err)
		}

		text, err := prototext.Marshal(m)
		if err != nil {
			log.Fatal(err)
		}
		if err := ioutil.WriteFile(fmt.Sprintf("internal/fuzz/textfuzz/corpus/%x", sha1.Sum(text)), text, 0777); err != nil {
			log.Fatal(err)
		}

		json, err := protojson.Marshal(m)
		if err != nil {
			log.Fatal(err)
		}
		if err := ioutil.WriteFile(fmt.Sprintf("internal/fuzz/jsonfuzz/corpus/%x", sha1.Sum(json)), json, 0777); err != nil {
			log.Fatal(err)
		}
	}
}
