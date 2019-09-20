package flatbuffers

import (
	"unsafe"
)

const (
	// See http://golang.org/ref/spec#Numeric_types

	// SizeUint8 is the byte size of a uint8.
	SizeUint8 = 1
	// SizeUint16 is the byte size of a uint16.
	SizeUint16 = 2
	// SizeUint32 is the byte size of a uint32.
	SizeUint32 = 4
	// SizeUint64 is the byte size of a uint64.
	SizeUint64 = 8

	// SizeInt8 is the byte size of a int8.
	SizeInt8 = 1
	// SizeInt16 is the byte size of a int16.
	SizeInt16 = 2
	// SizeInt32 is the byte size of a int32.
	SizeInt32 = 4
	// SizeInt64 is the byte size of a int64.
	SizeInt64 = 8

	// SizeFloat32 is the byte size of a float32.
	SizeFloat32 = 4
	// SizeFloat64 is the byte size of a float64.
	SizeFloat64 = 8

	// SizeByte is the byte size of a byte.
	// The `byte` type is aliased (by Go definition) to uint8.
	SizeByte = 1

	// SizeBool is the byte size of a bool.
	// The `bool` type is aliased (by flatbuffers convention) to uint8.
	SizeBool = 1

	// SizeSOffsetT is the byte size of an SOffsetT.
	// The `SOffsetT` type is aliased (by flatbuffers convention) to int32.
	SizeSOffsetT = 4
	// SizeUOffsetT is the byte size of an UOffsetT.
	// The `UOffsetT` type is aliased (by flatbuffers convention) to uint32.
	SizeUOffsetT = 4
	// SizeVOffsetT is the byte size of an VOffsetT.
	// The `VOffsetT` type is aliased (by flatbuffers convention) to uint16.
	SizeVOffsetT = 2
)

// byteSliceToString converts a []byte to string without a heap allocation.
func byteSliceToString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}
