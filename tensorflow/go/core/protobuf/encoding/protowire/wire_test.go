// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protowire

import (
	"bytes"
	"encoding/hex"
	"io"
	"math"
	"strings"
	"testing"
)

type (
	testOps struct {
		// appendOps is a sequence of append operations, each appending to
		// the output of the previous append operation.
		appendOps []appendOp

		// wantRaw (if not nil) is the bytes that the appendOps should produce.
		wantRaw []byte

		// consumeOps are a sequence of consume operations, each consuming the
		// remaining output after the previous consume operation.
		// The first consume operation starts with the output of appendOps.
		consumeOps []consumeOp
	}

	// appendOp represents an Append operation.
	appendOp  = interface{}
	appendTag struct {
		inNum  Number
		inType Type
	}
	appendVarint struct {
		inVal uint64
	}
	appendFixed32 struct {
		inVal uint32
	}
	appendFixed64 struct {
		inVal uint64
	}
	appendBytes struct {
		inVal []byte
	}
	appendGroup struct {
		inNum Number
		inVal []byte
	}
	appendRaw []byte

	// consumeOp represents an Consume operation.
	consumeOp    = interface{}
	consumeField struct {
		wantNum  Number
		wantType Type
		wantCnt  int
		wantErr  error
	}
	consumeFieldValue struct {
		inNum   Number
		inType  Type
		wantCnt int
		wantErr error
	}
	consumeTag struct {
		wantNum  Number
		wantType Type
		wantCnt  int
		wantErr  error
	}
	consumeVarint struct {
		wantVal uint64
		wantCnt int
		wantErr error
	}
	consumeFixed32 struct {
		wantVal uint32
		wantCnt int
		wantErr error
	}
	consumeFixed64 struct {
		wantVal uint64
		wantCnt int
		wantErr error
	}
	consumeBytes struct {
		wantVal []byte
		wantCnt int
		wantErr error
	}
	consumeGroup struct {
		inNum   Number
		wantVal []byte
		wantCnt int
		wantErr error
	}

	ops []interface{}
)

// dhex decodes a hex-string and returns the bytes and panics if s is invalid.
func dhex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func TestTag(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeTag{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendTag{inNum: 0, inType: Fixed32Type}},
		wantRaw:    dhex("05"),
		consumeOps: ops{consumeTag{wantErr: errFieldNumber}},
	}, {
		appendOps:  ops{appendTag{inNum: 1, inType: Fixed32Type}},
		wantRaw:    dhex("0d"),
		consumeOps: ops{consumeTag{wantNum: 1, wantType: Fixed32Type, wantCnt: 1}},
	}, {
		appendOps:  ops{appendTag{inNum: FirstReservedNumber, inType: BytesType}},
		wantRaw:    dhex("c2a309"),
		consumeOps: ops{consumeTag{wantNum: FirstReservedNumber, wantType: BytesType, wantCnt: 3}},
	}, {
		appendOps:  ops{appendTag{inNum: LastReservedNumber, inType: StartGroupType}},
		wantRaw:    dhex("fbe109"),
		consumeOps: ops{consumeTag{wantNum: LastReservedNumber, wantType: StartGroupType, wantCnt: 3}},
	}, {
		appendOps:  ops{appendTag{inNum: MaxValidNumber, inType: VarintType}},
		wantRaw:    dhex("f8ffffff0f"),
		consumeOps: ops{consumeTag{wantNum: MaxValidNumber, wantType: VarintType, wantCnt: 5}},
	}, {
		appendOps:  ops{appendVarint{inVal: ((math.MaxInt32+1)<<3 | uint64(VarintType))}},
		wantRaw:    dhex("8080808040"),
		consumeOps: ops{consumeTag{wantErr: errFieldNumber}},
	}})
}

func TestVarint(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("80"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("8080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("80808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("8080808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("808080808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("80808080808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("8080808080808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("808080808080808080"))},
		consumeOps: ops{consumeVarint{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("80808080808080808080"))},
		consumeOps: ops{consumeVarint{wantErr: errOverflow}},
	}, {
		// Test varints at various boundaries where the length changes.
		appendOps:  ops{appendVarint{inVal: 0x0}},
		wantRaw:    dhex("00"),
		consumeOps: ops{consumeVarint{wantVal: 0, wantCnt: 1}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x1}},
		wantRaw:    dhex("01"),
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 1}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7f}},
		wantRaw:    dhex("7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x7f, wantCnt: 1}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7f + 1}},
		wantRaw:    dhex("8001"),
		consumeOps: ops{consumeVarint{wantVal: 0x7f + 1, wantCnt: 2}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x3fff}},
		wantRaw:    dhex("ff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x3fff, wantCnt: 2}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x3fff + 1}},
		wantRaw:    dhex("808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x3fff + 1, wantCnt: 3}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x1fffff}},
		wantRaw:    dhex("ffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x1fffff, wantCnt: 3}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x1fffff + 1}},
		wantRaw:    dhex("80808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x1fffff + 1, wantCnt: 4}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0xfffffff}},
		wantRaw:    dhex("ffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0xfffffff, wantCnt: 4}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0xfffffff + 1}},
		wantRaw:    dhex("8080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0xfffffff + 1, wantCnt: 5}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7ffffffff}},
		wantRaw:    dhex("ffffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x7ffffffff, wantCnt: 5}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7ffffffff + 1}},
		wantRaw:    dhex("808080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x7ffffffff + 1, wantCnt: 6}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x3ffffffffff}},
		wantRaw:    dhex("ffffffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x3ffffffffff, wantCnt: 6}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x3ffffffffff + 1}},
		wantRaw:    dhex("80808080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x3ffffffffff + 1, wantCnt: 7}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x1ffffffffffff}},
		wantRaw:    dhex("ffffffffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x1ffffffffffff, wantCnt: 7}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x1ffffffffffff + 1}},
		wantRaw:    dhex("8080808080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x1ffffffffffff + 1, wantCnt: 8}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0xffffffffffffff}},
		wantRaw:    dhex("ffffffffffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0xffffffffffffff, wantCnt: 8}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0xffffffffffffff + 1}},
		wantRaw:    dhex("808080808080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0xffffffffffffff + 1, wantCnt: 9}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7fffffffffffffff}},
		wantRaw:    dhex("ffffffffffffffff7f"),
		consumeOps: ops{consumeVarint{wantVal: 0x7fffffffffffffff, wantCnt: 9}},
	}, {
		appendOps:  ops{appendVarint{inVal: 0x7fffffffffffffff + 1}},
		wantRaw:    dhex("80808080808080808001"),
		consumeOps: ops{consumeVarint{wantVal: 0x7fffffffffffffff + 1, wantCnt: 10}},
	}, {
		appendOps:  ops{appendVarint{inVal: math.MaxUint64}},
		wantRaw:    dhex("ffffffffffffffffff01"),
		consumeOps: ops{consumeVarint{wantVal: math.MaxUint64, wantCnt: 10}},
	}, {
		appendOps:  ops{appendRaw(dhex("ffffffffffffffffff02"))},
		consumeOps: ops{consumeVarint{wantErr: errOverflow}},
	}, {
		// Test denormalized varints; where the encoding, while valid, is
		// larger than necessary.
		appendOps:  ops{appendRaw(dhex("01"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 1}},
	}, {
		appendOps:  ops{appendRaw(dhex("8100"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 2}},
	}, {
		appendOps:  ops{appendRaw(dhex("818000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 3}},
	}, {
		appendOps:  ops{appendRaw(dhex("81808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 4}},
	}, {
		appendOps:  ops{appendRaw(dhex("8180808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 5}},
	}, {
		appendOps:  ops{appendRaw(dhex("818080808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 6}},
	}, {
		appendOps:  ops{appendRaw(dhex("81808080808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 7}},
	}, {
		appendOps:  ops{appendRaw(dhex("8180808080808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 8}},
	}, {
		appendOps:  ops{appendRaw(dhex("818080808080808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 9}},
	}, {
		appendOps:  ops{appendRaw(dhex("81808080808080808000"))},
		consumeOps: ops{consumeVarint{wantVal: 1, wantCnt: 10}},
	}, {
		appendOps:  ops{appendRaw(dhex("8180808080808080808000"))},
		consumeOps: ops{consumeVarint{wantErr: errOverflow}},
	}})
}

func TestFixed32(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeFixed32{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("000000"))},
		consumeOps: ops{consumeFixed32{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendFixed32{0}},
		wantRaw:    dhex("00000000"),
		consumeOps: ops{consumeFixed32{wantVal: 0, wantCnt: 4}},
	}, {
		appendOps:  ops{appendFixed32{math.MaxUint32}},
		wantRaw:    dhex("ffffffff"),
		consumeOps: ops{consumeFixed32{wantVal: math.MaxUint32, wantCnt: 4}},
	}, {
		appendOps:  ops{appendFixed32{0xf0e1d2c3}},
		wantRaw:    dhex("c3d2e1f0"),
		consumeOps: ops{consumeFixed32{wantVal: 0xf0e1d2c3, wantCnt: 4}},
	}})
}

func TestFixed64(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeFixed64{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("00000000000000"))},
		consumeOps: ops{consumeFixed64{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendFixed64{0}},
		wantRaw:    dhex("0000000000000000"),
		consumeOps: ops{consumeFixed64{wantVal: 0, wantCnt: 8}},
	}, {
		appendOps:  ops{appendFixed64{math.MaxUint64}},
		wantRaw:    dhex("ffffffffffffffff"),
		consumeOps: ops{consumeFixed64{wantVal: math.MaxUint64, wantCnt: 8}},
	}, {
		appendOps:  ops{appendFixed64{0xf0e1d2c3b4a59687}},
		wantRaw:    dhex("8796a5b4c3d2e1f0"),
		consumeOps: ops{consumeFixed64{wantVal: 0xf0e1d2c3b4a59687, wantCnt: 8}},
	}})
}

func TestBytes(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeBytes{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendRaw(dhex("01"))},
		consumeOps: ops{consumeBytes{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendVarint{0}, appendRaw("")},
		wantRaw:    dhex("00"),
		consumeOps: ops{consumeBytes{wantVal: dhex(""), wantCnt: 1}},
	}, {
		appendOps:  ops{appendBytes{[]byte("hello")}},
		wantRaw:    []byte("\x05hello"),
		consumeOps: ops{consumeBytes{wantVal: []byte("hello"), wantCnt: 6}},
	}, {
		appendOps:  ops{appendBytes{[]byte(strings.Repeat("hello", 50))}},
		wantRaw:    []byte("\xfa\x01" + strings.Repeat("hello", 50)),
		consumeOps: ops{consumeBytes{wantVal: []byte(strings.Repeat("hello", 50)), wantCnt: 252}},
	}, {
		appendOps:  ops{appendRaw("\x85\x80\x00hello")},
		consumeOps: ops{consumeBytes{wantVal: []byte("hello"), wantCnt: 8}},
	}, {
		appendOps:  ops{appendRaw("\x85\x80\x00hell")},
		consumeOps: ops{consumeBytes{wantErr: io.ErrUnexpectedEOF}},
	}})
}

func TestGroup(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeGroup{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps:  ops{appendTag{inNum: 0, inType: StartGroupType}},
		consumeOps: ops{consumeGroup{inNum: 1, wantErr: errFieldNumber}},
	}, {
		appendOps:  ops{appendTag{inNum: 2, inType: EndGroupType}},
		consumeOps: ops{consumeGroup{inNum: 1, wantErr: errEndGroup}},
	}, {
		appendOps:  ops{appendTag{inNum: 1, inType: EndGroupType}},
		consumeOps: ops{consumeGroup{inNum: 1, wantCnt: 1}},
	}, {
		appendOps: ops{
			appendTag{inNum: 5, inType: Fixed32Type},
			appendFixed32{0xf0e1d2c3},
			appendTag{inNum: 5, inType: EndGroupType},
		},
		wantRaw:    dhex("2dc3d2e1f02c"),
		consumeOps: ops{consumeGroup{inNum: 5, wantVal: dhex("2dc3d2e1f0"), wantCnt: 6}},
	}, {
		appendOps: ops{
			appendTag{inNum: 5, inType: Fixed32Type},
			appendFixed32{0xf0e1d2c3},
			appendRaw(dhex("ac808000")),
		},
		consumeOps: ops{consumeGroup{inNum: 5, wantVal: dhex("2dc3d2e1f0"), wantCnt: 9}},
	}})
}

func TestField(t *testing.T) {
	runTests(t, []testOps{{
		appendOps:  ops{appendRaw(dhex(""))},
		consumeOps: ops{consumeField{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps: ops{
			appendTag{inNum: 5000, inType: StartGroupType},
			appendTag{inNum: 1, inType: VarintType},
			appendVarint{123456789},
			appendTag{inNum: 12, inType: Fixed32Type},
			appendFixed32{123456789},
			appendTag{inNum: 123, inType: Fixed64Type},
			appendFixed64{123456789},
			appendTag{inNum: 1234, inType: BytesType},
			appendBytes{[]byte("hello")},
			appendTag{inNum: 12345, inType: StartGroupType},
			appendTag{inNum: 11, inType: VarintType},
			appendVarint{123456789},
			appendTag{inNum: 1212, inType: Fixed32Type},
			appendFixed32{123456789},
			appendTag{inNum: 123123, inType: Fixed64Type},
			appendFixed64{123456789},
			appendTag{inNum: 12341234, inType: BytesType},
			appendBytes{[]byte("goodbye")},
			appendTag{inNum: 12345, inType: EndGroupType},
			appendTag{inNum: 5000, inType: EndGroupType},
		},
		wantRaw: dhex("c3b80208959aef3a6515cd5b07d90715cd5b0700000000924d0568656c6c6fcb830658959aef3ae54b15cd5b07998f3c15cd5b070000000092ff892f07676f6f64627965cc8306c4b802"),
		consumeOps: ops{
			consumeTag{wantNum: 5000, wantType: StartGroupType, wantCnt: 3},
			consumeTag{wantNum: 1, wantType: VarintType, wantCnt: 1},
			consumeVarint{wantVal: 123456789, wantCnt: 4},
			consumeTag{wantNum: 12, wantType: Fixed32Type, wantCnt: 1},
			consumeFixed32{wantVal: 123456789, wantCnt: 4},
			consumeTag{wantNum: 123, wantType: Fixed64Type, wantCnt: 2},
			consumeFixed64{wantVal: 123456789, wantCnt: 8},
			consumeTag{wantNum: 1234, wantType: BytesType, wantCnt: 2},
			consumeBytes{wantVal: []byte("hello"), wantCnt: 6},
			consumeTag{wantNum: 12345, wantType: StartGroupType, wantCnt: 3},
			consumeTag{wantNum: 11, wantType: VarintType, wantCnt: 1},
			consumeVarint{wantVal: 123456789, wantCnt: 4},
			consumeTag{wantNum: 1212, wantType: Fixed32Type, wantCnt: 2},
			consumeFixed32{wantVal: 123456789, wantCnt: 4},
			consumeTag{wantNum: 123123, wantType: Fixed64Type, wantCnt: 3},
			consumeFixed64{wantVal: 123456789, wantCnt: 8},
			consumeTag{wantNum: 12341234, wantType: BytesType, wantCnt: 4},
			consumeBytes{wantVal: []byte("goodbye"), wantCnt: 8},
			consumeTag{wantNum: 12345, wantType: EndGroupType, wantCnt: 3},
			consumeTag{wantNum: 5000, wantType: EndGroupType, wantCnt: 3},
		},
	}, {
		appendOps:  ops{appendRaw(dhex("c3b80208959aef3a6515cd5b07d90715cd5b0700000000924d0568656c6c6fcb830658959aef3ae54b15cd5b07998f3c15cd5b070000000092ff892f07676f6f64627965cc8306c4b802"))},
		consumeOps: ops{consumeField{wantNum: 5000, wantType: StartGroupType, wantCnt: 74}},
	}, {
		appendOps:  ops{appendTag{inNum: 5, inType: EndGroupType}},
		wantRaw:    dhex("2c"),
		consumeOps: ops{consumeField{wantErr: errEndGroup}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendTag{inNum: 22, inType: StartGroupType},
			appendTag{inNum: 333, inType: StartGroupType},
			appendTag{inNum: 4444, inType: StartGroupType},
			appendTag{inNum: 4444, inType: EndGroupType},
			appendTag{inNum: 333, inType: EndGroupType},
			appendTag{inNum: 22, inType: EndGroupType},
			appendTag{inNum: 1, inType: EndGroupType},
		},
		wantRaw:    dhex("0bb301eb14e39502e49502ec14b4010c"),
		consumeOps: ops{consumeField{wantNum: 1, wantType: StartGroupType, wantCnt: 16}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendGroup{inNum: 1, inVal: dhex("b301eb14e39502e49502ec14b401")},
		},
		wantRaw:    dhex("0b" + "b301eb14e39502e49502ec14b401" + "0c"),
		consumeOps: ops{consumeField{wantNum: 1, wantType: StartGroupType, wantCnt: 16}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendTag{inNum: 22, inType: StartGroupType},
			appendTag{inNum: 333, inType: StartGroupType},
			appendTag{inNum: 4444, inType: StartGroupType},
			appendTag{inNum: 333, inType: EndGroupType},
			appendTag{inNum: 22, inType: EndGroupType},
			appendTag{inNum: 1, inType: EndGroupType},
		},
		consumeOps: ops{consumeField{wantErr: errEndGroup}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendTag{inNum: 22, inType: StartGroupType},
			appendTag{inNum: 333, inType: StartGroupType},
			appendTag{inNum: 4444, inType: StartGroupType},
			appendTag{inNum: 4444, inType: EndGroupType},
			appendTag{inNum: 333, inType: EndGroupType},
			appendTag{inNum: 22, inType: EndGroupType},
		},
		consumeOps: ops{consumeField{wantErr: io.ErrUnexpectedEOF}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendTag{inNum: 22, inType: StartGroupType},
			appendTag{inNum: 333, inType: StartGroupType},
			appendTag{inNum: 4444, inType: StartGroupType},
			appendTag{inNum: 0, inType: VarintType},
			appendTag{inNum: 4444, inType: EndGroupType},
			appendTag{inNum: 333, inType: EndGroupType},
			appendTag{inNum: 22, inType: EndGroupType},
			appendTag{inNum: 1, inType: EndGroupType},
		},
		consumeOps: ops{consumeField{wantErr: errFieldNumber}},
	}, {
		appendOps: ops{
			appendTag{inNum: 1, inType: StartGroupType},
			appendTag{inNum: 22, inType: StartGroupType},
			appendTag{inNum: 333, inType: StartGroupType},
			appendTag{inNum: 4444, inType: StartGroupType},
			appendTag{inNum: 1, inType: 6},
			appendTag{inNum: 4444, inType: EndGroupType},
			appendTag{inNum: 333, inType: EndGroupType},
			appendTag{inNum: 22, inType: EndGroupType},
			appendTag{inNum: 1, inType: EndGroupType},
		},
		consumeOps: ops{consumeField{wantErr: errReserved}},
	}})
}

func runTests(t *testing.T, tests []testOps) {
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var b []byte
			for _, op := range tt.appendOps {
				b0 := b
				switch op := op.(type) {
				case appendTag:
					b = AppendTag(b, op.inNum, op.inType)
				case appendVarint:
					b = AppendVarint(b, op.inVal)
				case appendFixed32:
					b = AppendFixed32(b, op.inVal)
				case appendFixed64:
					b = AppendFixed64(b, op.inVal)
				case appendBytes:
					b = AppendBytes(b, op.inVal)
				case appendGroup:
					b = AppendGroup(b, op.inNum, op.inVal)
				case appendRaw:
					b = append(b, op...)
				}

				check := func(label string, want int) {
					t.Helper()
					if got := len(b) - len(b0); got != want {
						t.Errorf("len(Append%v) and Size%v mismatch: got %v, want %v", label, label, got, want)
					}
				}
				switch op := op.(type) {
				case appendTag:
					check("Tag", SizeTag(op.inNum))
				case appendVarint:
					check("Varint", SizeVarint(op.inVal))
				case appendFixed32:
					check("Fixed32", SizeFixed32())
				case appendFixed64:
					check("Fixed64", SizeFixed64())
				case appendBytes:
					check("Bytes", SizeBytes(len(op.inVal)))
				case appendGroup:
					check("Group", SizeGroup(op.inNum, len(op.inVal)))
				}
			}

			if tt.wantRaw != nil && !bytes.Equal(b, tt.wantRaw) {
				t.Errorf("raw output mismatch:\ngot  %x\nwant %x", b, tt.wantRaw)
			}

			for _, op := range tt.consumeOps {
				check := func(label string, gotCnt, wantCnt int, wantErr error) {
					t.Helper()
					gotErr := ParseError(gotCnt)
					if gotCnt < 0 {
						gotCnt = 0
					}
					if gotCnt != wantCnt {
						t.Errorf("Consume%v(): consumed %d bytes, want %d bytes consumed", label, gotCnt, wantCnt)
					}
					if gotErr != wantErr {
						t.Errorf("Consume%v(): got %v error, want %v error", label, gotErr, wantErr)
					}
					b = b[gotCnt:]
				}
				switch op := op.(type) {
				case consumeField:
					gotNum, gotType, n := ConsumeField(b)
					if gotNum != op.wantNum || gotType != op.wantType {
						t.Errorf("ConsumeField() = (%d, %v), want (%d, %v)", gotNum, gotType, op.wantNum, op.wantType)
					}
					check("Field", n, op.wantCnt, op.wantErr)
				case consumeFieldValue:
					n := ConsumeFieldValue(op.inNum, op.inType, b)
					check("FieldValue", n, op.wantCnt, op.wantErr)
				case consumeTag:
					gotNum, gotType, n := ConsumeTag(b)
					if gotNum != op.wantNum || gotType != op.wantType {
						t.Errorf("ConsumeTag() = (%d, %v), want (%d, %v)", gotNum, gotType, op.wantNum, op.wantType)
					}
					check("Tag", n, op.wantCnt, op.wantErr)
				case consumeVarint:
					gotVal, n := ConsumeVarint(b)
					if gotVal != op.wantVal {
						t.Errorf("ConsumeVarint() = %d, want %d", gotVal, op.wantVal)
					}
					check("Varint", n, op.wantCnt, op.wantErr)
				case consumeFixed32:
					gotVal, n := ConsumeFixed32(b)
					if gotVal != op.wantVal {
						t.Errorf("ConsumeFixed32() = %d, want %d", gotVal, op.wantVal)
					}
					check("Fixed32", n, op.wantCnt, op.wantErr)
				case consumeFixed64:
					gotVal, n := ConsumeFixed64(b)
					if gotVal != op.wantVal {
						t.Errorf("ConsumeFixed64() = %d, want %d", gotVal, op.wantVal)
					}
					check("Fixed64", n, op.wantCnt, op.wantErr)
				case consumeBytes:
					gotVal, n := ConsumeBytes(b)
					if !bytes.Equal(gotVal, op.wantVal) {
						t.Errorf("ConsumeBytes() = %x, want %x", gotVal, op.wantVal)
					}
					check("Bytes", n, op.wantCnt, op.wantErr)
				case consumeGroup:
					gotVal, n := ConsumeGroup(op.inNum, b)
					if !bytes.Equal(gotVal, op.wantVal) {
						t.Errorf("ConsumeGroup() = %x, want %x", gotVal, op.wantVal)
					}
					check("Group", n, op.wantCnt, op.wantErr)
				}
			}
		})
	}
}

func TestZigZag(t *testing.T) {
	tests := []struct {
		dec int64
		enc uint64
	}{
		{math.MinInt64 + 0, math.MaxUint64 - 0},
		{math.MinInt64 + 1, math.MaxUint64 - 2},
		{math.MinInt64 + 2, math.MaxUint64 - 4},
		{-3, 5},
		{-2, 3},
		{-1, 1},
		{0, 0},
		{+1, 2},
		{+2, 4},
		{+3, 6},
		{math.MaxInt64 - 2, math.MaxUint64 - 5},
		{math.MaxInt64 - 1, math.MaxUint64 - 3},
		{math.MaxInt64 - 0, math.MaxUint64 - 1},
	}

	for _, tt := range tests {
		if enc := EncodeZigZag(tt.dec); enc != tt.enc {
			t.Errorf("EncodeZigZag(%d) = %d, want %d", tt.dec, enc, tt.enc)
		}
		if dec := DecodeZigZag(tt.enc); dec != tt.dec {
			t.Errorf("DecodeZigZag(%d) = %d, want %d", tt.enc, dec, tt.dec)
		}
	}
}
