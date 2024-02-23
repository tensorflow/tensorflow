// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import (
	"bytes"
	"math"
	"reflect"
	"testing"
)

var (
	fakeMessage = new(struct{ Message })
	fakeList    = new(struct{ List })
	fakeMap     = new(struct{ Map })
)

func TestValue(t *testing.T) {

	tests := []struct {
		in   Value
		want interface{}
	}{
		{in: Value{}},
		{in: ValueOf(nil)},
		{in: ValueOf(true), want: true},
		{in: ValueOf(int32(math.MaxInt32)), want: int32(math.MaxInt32)},
		{in: ValueOf(int64(math.MaxInt64)), want: int64(math.MaxInt64)},
		{in: ValueOf(uint32(math.MaxUint32)), want: uint32(math.MaxUint32)},
		{in: ValueOf(uint64(math.MaxUint64)), want: uint64(math.MaxUint64)},
		{in: ValueOf(float32(math.MaxFloat32)), want: float32(math.MaxFloat32)},
		{in: ValueOf(float64(math.MaxFloat64)), want: float64(math.MaxFloat64)},
		{in: ValueOf(string("hello")), want: string("hello")},
		{in: ValueOf([]byte("hello")), want: []byte("hello")},
		{in: ValueOf(fakeMessage), want: fakeMessage},
		{in: ValueOf(fakeList), want: fakeList},
		{in: ValueOf(fakeMap), want: fakeMap},
	}

	for _, tt := range tests {
		got := tt.in.Interface()
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Value(%v).Interface() = %v, want %v", tt.in, got, tt.want)
		}

		if got := tt.in.IsValid(); got != (tt.want != nil) {
			t.Errorf("Value(%v).IsValid() = %v, want %v", tt.in, got, tt.want != nil)
		}
		switch want := tt.want.(type) {
		case int32:
			if got := tt.in.Int(); got != int64(want) {
				t.Errorf("Value(%v).Int() = %v, want %v", tt.in, got, tt.want)
			}
		case int64:
			if got := tt.in.Int(); got != int64(want) {
				t.Errorf("Value(%v).Int() = %v, want %v", tt.in, got, tt.want)
			}
		case uint32:
			if got := tt.in.Uint(); got != uint64(want) {
				t.Errorf("Value(%v).Uint() = %v, want %v", tt.in, got, tt.want)
			}
		case uint64:
			if got := tt.in.Uint(); got != uint64(want) {
				t.Errorf("Value(%v).Uint() = %v, want %v", tt.in, got, tt.want)
			}
		case float32:
			if got := tt.in.Float(); got != float64(want) {
				t.Errorf("Value(%v).Float() = %v, want %v", tt.in, got, tt.want)
			}
		case float64:
			if got := tt.in.Float(); got != float64(want) {
				t.Errorf("Value(%v).Float() = %v, want %v", tt.in, got, tt.want)
			}
		case string:
			if got := tt.in.String(); got != string(want) {
				t.Errorf("Value(%v).String() = %v, want %v", tt.in, got, tt.want)
			}
		case []byte:
			if got := tt.in.Bytes(); !bytes.Equal(got, want) {
				t.Errorf("Value(%v).Bytes() = %v, want %v", tt.in, got, tt.want)
			}
		case EnumNumber:
			if got := tt.in.Enum(); got != want {
				t.Errorf("Value(%v).Enum() = %v, want %v", tt.in, got, tt.want)
			}
		case Message:
			if got := tt.in.Message(); got != want {
				t.Errorf("Value(%v).Message() = %v, want %v", tt.in, got, tt.want)
			}
		case List:
			if got := tt.in.List(); got != want {
				t.Errorf("Value(%v).List() = %v, want %v", tt.in, got, tt.want)
			}
		case Map:
			if got := tt.in.Map(); got != want {
				t.Errorf("Value(%v).Map() = %v, want %v", tt.in, got, tt.want)
			}
		}
	}
}

func TestValueEqual(t *testing.T) {
	tests := []struct {
		x, y Value
		want bool
	}{
		{Value{}, Value{}, true},
		{Value{}, ValueOfBool(true), false},
		{ValueOfBool(true), ValueOfBool(true), true},
		{ValueOfBool(true), ValueOfBool(false), false},
		{ValueOfBool(false), ValueOfInt32(0), false},
		{ValueOfInt32(0), ValueOfInt32(0), true},
		{ValueOfInt32(0), ValueOfInt32(1), false},
		{ValueOfInt32(0), ValueOfInt64(0), false},
		{ValueOfInt64(123), ValueOfInt64(123), true},
		{ValueOfFloat64(0), ValueOfFloat64(0), true},
		{ValueOfFloat64(math.NaN()), ValueOfFloat64(math.NaN()), true},
		{ValueOfFloat64(math.NaN()), ValueOfFloat64(0), false},
		{ValueOfFloat64(math.Inf(1)), ValueOfFloat64(math.Inf(1)), true},
		{ValueOfFloat64(math.Inf(-1)), ValueOfFloat64(math.Inf(1)), false},
		{ValueOfBytes(nil), ValueOfBytes(nil), true},
		{ValueOfBytes(nil), ValueOfBytes([]byte{}), true},
		{ValueOfBytes(nil), ValueOfBytes([]byte{1}), false},
		{ValueOfEnum(0), ValueOfEnum(0), true},
		{ValueOfEnum(0), ValueOfEnum(1), false},
		{ValueOfBool(false), ValueOfMessage(fakeMessage), false},
		{ValueOfMessage(fakeMessage), ValueOfList(fakeList), false},
		{ValueOfList(fakeList), ValueOfMap(fakeMap), false},
		{ValueOfMap(fakeMap), ValueOfMessage(fakeMessage), false},

		// Composite types are not tested here.
		// See proto.TestEqual.
	}

	for _, tt := range tests {
		got := tt.x.Equal(tt.y)
		if got != tt.want {
			t.Errorf("(%v).Equal(%v) = %v, want %v", tt.x, tt.y, got, tt.want)
		}
	}
}

func BenchmarkValue(b *testing.B) {
	const testdata = "The quick brown fox jumped over the lazy dog."
	var sink1 string
	var sink2 Value
	var sink3 interface{}

	// Baseline measures the time to store a string into a native variable.
	b.Run("Baseline", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sink1 = testdata[:len(testdata)%(i+1)]
		}
	})

	// Inline measures the time to store a string into a Value,
	// assuming that the compiler could inline the ValueOf function call.
	b.Run("Inline", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sink2 = valueOfString(testdata[:len(testdata)%(i+1)])
		}
	})

	// Value measures the time to store a string into a Value using the general
	// ValueOf function call. This should be identical to Inline.
	//
	// NOTE: As of Go1.11, this is not as efficient as Inline due to the lack
	// of some compiler optimizations:
	//	https://golang.org/issue/22310
	//	https://golang.org/issue/25189
	b.Run("Value", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sink2 = ValueOf(string(testdata[:len(testdata)%(i+1)]))
		}
	})

	// Interface measures the time to store a string into an interface.
	b.Run("Interface", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sink3 = string(testdata[:len(testdata)%(i+1)])
		}
	})

	_, _, _ = sink1, sink2, sink3
}
