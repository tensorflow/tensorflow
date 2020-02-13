/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

// #include <stdlib.h>
// #include <string.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math/bits"
	"reflect"
	"runtime"
	"unsafe"
)

// DataType holds the type for a scalar value.  E.g., one slot in a tensor.
type DataType C.TF_DataType

// Types of scalar values in the TensorFlow type system.
const (
	Float      DataType = C.TF_FLOAT
	Double     DataType = C.TF_DOUBLE
	Int32      DataType = C.TF_INT32
	Uint32     DataType = C.TF_UINT32
	Uint8      DataType = C.TF_UINT8
	Int16      DataType = C.TF_INT16
	Int8       DataType = C.TF_INT8
	String     DataType = C.TF_STRING
	Complex64  DataType = C.TF_COMPLEX64
	Complex    DataType = C.TF_COMPLEX
	Int64      DataType = C.TF_INT64
	Uint64     DataType = C.TF_UINT64
	Bool       DataType = C.TF_BOOL
	Qint8      DataType = C.TF_QINT8
	Quint8     DataType = C.TF_QUINT8
	Qint32     DataType = C.TF_QINT32
	Bfloat16   DataType = C.TF_BFLOAT16
	Qint16     DataType = C.TF_QINT16
	Quint16    DataType = C.TF_QUINT16
	Uint16     DataType = C.TF_UINT16
	Complex128 DataType = C.TF_COMPLEX128
	Half       DataType = C.TF_HALF
)

// Tensor holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	c     *C.TF_Tensor
	shape []int64
}

// NewTensor converts from a Go value to a Tensor. Valid values are scalars,
// slices, and arrays. Every element of a slice must have the same length so
// that the resulting Tensor has a valid shape.
func NewTensor(value interface{}) (*Tensor, error) {
	val := reflect.ValueOf(value)
	shape, dataType, err := shapeAndDataTypeOf(val)
	if err != nil {
		return nil, err
	}
	nflattened := numElements(shape)
	nbytes := typeOf(dataType, nil).Size() * uintptr(nflattened)
	if dataType == String {
		// TF_STRING tensors are encoded as an array of 8-byte offsets
		// followed by string data. See c_api.h.
		nbytes = uintptr(nflattened*8 + int64(byteSizeOfEncodedStrings(val)))
	}
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := &Tensor{
		c:     C.TF_AllocateTensor(C.TF_DataType(dataType), shapePtr, C.int(len(shape)), C.size_t(nbytes)),
		shape: shape,
	}
	runtime.SetFinalizer(t, (*Tensor).finalize)
	raw := tensorData(t.c)
	buf := bytes.NewBuffer(raw[:0:len(raw)])
	if dataType != String {
		if isAllArray(val.Type()) {
			// We have arrays all the way down, or just primitive types. We can
			// just copy the memory in as it is all contiguous.
			if err := copyPtr(buf, unpackEFace(value).data, int(val.Type().Size())); err != nil {
				return nil, err
			}
		} else {
			// When there are slices involved the memory for each leaf slice may
			// not be contiguous with the others or in the order we might
			// expect, so we need to work our way down to each slice of
			// primitives and copy them individually
			if err := encodeTensorWithSlices(buf, val, shape); err != nil {
				return nil, err
			}
		}

		if uintptr(buf.Len()) != nbytes {
			return nil, bug("NewTensor incorrectly calculated the size of a tensor with type %v and shape %v as %v bytes instead of %v", dataType, shape, nbytes, buf.Len())
		}
	} else {
		e := stringEncoder{offsets: buf, data: raw[nflattened*8:], status: newStatus()}
		if err := e.encode(reflect.ValueOf(value), shape); err != nil {
			return nil, err
		}
		if int64(buf.Len()) != nflattened*8 {
			return nil, bug("invalid offset encoding for TF_STRING tensor with shape %v (got %v, want %v)", shape, buf.Len(), nflattened*8)
		}
	}
	return t, nil
}

// isAllArray returns true if type is a primitive type or an array of primitive
// types or an array of ... etc.. When this is true the data we want is
// contiguous in RAM.
func isAllArray(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.Slice:
		return false
	case reflect.Array:
		return isAllArray(typ.Elem())
	default:
		// We know the type is slices/arrays of slices/arrays of primitive types.
		return true
	}
}

// eface defines what an interface type actually is: a pointer to type
// information about the encapsulated type and a pointer to the encapsulated
// value.
type eface struct {
	rtype unsafe.Pointer
	data  unsafe.Pointer
}

// unpackEFace gives us an effient way to get us a pointer to the value carried
// in an interface. If you wrap a pointer type in an interface then the pointer
// is directly stored in the interface struct. If you wrap a value type in an
// interface then the compiler copies the value into a newly allocated piece of
// memory and stores a pointer to that memory in the interface. So we're
// guaranteed to get a pointer. Go reflection doesn't expose the pointer to
// value types straightforwardly as it doesn't want you to think you have a
// reference to the original value. But we just want a pointer to make it
// efficient to read the value, so cheating like this should be safe and
// reasonable.
func unpackEFace(obj interface{}) *eface {
	return (*eface)(unsafe.Pointer(&obj))
}

// ReadTensor constructs a Tensor with the provided type and shape from the
// serialized tensor contents in r.
//
// See also WriteContentsTo.
func ReadTensor(dataType DataType, shape []int64, r io.Reader) (*Tensor, error) {
	if err := isTensorSerializable(dataType); err != nil {
		return nil, err
	}
	nbytes := typeOf(dataType, nil).Size() * uintptr(numElements(shape))
	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}
	t := &Tensor{
		c:     C.TF_AllocateTensor(C.TF_DataType(dataType), shapePtr, C.int(len(shape)), C.size_t(nbytes)),
		shape: shape,
	}
	runtime.SetFinalizer(t, (*Tensor).finalize)
	raw := tensorData(t.c)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	return t, nil
}

// newTensorFromC takes ownership of c and returns the owning Tensor.
func newTensorFromC(c *C.TF_Tensor) *Tensor {
	var shape []int64
	if ndims := int(C.TF_NumDims(c)); ndims > 0 {
		shape = make([]int64, ndims)
	}
	for i := range shape {
		shape[i] = int64(C.TF_Dim(c, C.int(i)))
	}
	t := &Tensor{c: c, shape: shape}
	runtime.SetFinalizer(t, (*Tensor).finalize)
	return t
}

func (t *Tensor) finalize() { C.TF_DeleteTensor(t.c) }

// DataType returns the scalar datatype of the Tensor.
func (t *Tensor) DataType() DataType { return DataType(C.TF_TensorType(t.c)) }

// Shape returns the shape of the Tensor.
func (t *Tensor) Shape() []int64 { return t.shape }

// Value converts the Tensor to a Go value. For now, not all Tensor types are
// supported, and this function may panic if it encounters an unsupported
// DataType.
//
// The type of the output depends on the Tensor type and dimensions.
// For example:
// Tensor(int64, 0): int64
// Tensor(float64, 3): [][][]float64
func (t *Tensor) Value() interface{} {
	raw := tensorData(t.c)
	shape := t.Shape()
	dt := t.DataType()
	return decodeTensor(raw, shape, dt).Interface()
}

func decodeTensor(raw []byte, shape []int64, dt DataType) reflect.Value {
	// Create a 1-dimensional slice of the base large enough for the data and
	// copy the data in.
	n := int(numElements(shape))

	var (
		slice reflect.Value
		typ   reflect.Type
	)
	if dt == String {
		strs, err := decodeOneDimString(raw, n)
		if err != nil {
			panic(bug("unable to decode string with shape %v: %v", shape, err))
		}
		slice = reflect.ValueOf(strs)
		typ = slice.Type()
	} else {
		typ = typeForDataType(dt)
		l := n * int(typ.Size())
		typ = reflect.SliceOf(typ)
		slice = reflect.MakeSlice(typ, n, n)
		h := sliceHeader{
			Data: unsafe.Pointer(slice.Pointer()),
			Len:  l,
			Cap:  l,
		}
		baseBytes := *(*[]byte)(unsafe.Pointer(&h))
		copy(baseBytes, raw)
	}

	// Now we have the data in place in the base slice we can add the
	// dimensions. We want to walk backwards through the shape. If the shape is
	// length 1 or 0 then we're already done.
	if len(shape) == 0 {
		return slice.Index(0)
	}
	if len(shape) == 1 {
		return slice
	}
	// We have a special case if the tensor has no data. Our backing slice is
	// empty, but we still want to create slices following the shape. In this
	// case only the final part of the shape will be 0 and we want to recalculate
	// n at this point ignoring that 0.
	// For example if our shape is 3 * 2 * 0 then n will be zero, but we still
	// want 6 zero length slices to group as follows.
	// {{} {}} {{} {}} {{} {}}
	if n == 0 {
		n = int(numElements(shape[:len(shape)-1]))
	}
	for i := len(shape) - 2; i >= 0; i-- {
		underlyingSize := typ.Elem().Size()
		typ = reflect.SliceOf(typ)
		subsliceLen := int(shape[i+1])
		if subsliceLen != 0 {
			n = n / subsliceLen
		}
		// Just using reflection it is difficult to avoid unnecessary
		// allocations while setting up the sub-slices as the Slice function on
		// a slice Value allocates. So we end up doing pointer arithmetic!
		// Pointer() on a slice gives us access to the data backing the slice.
		// We insert slice headers directly into this data.
		data := slice.Pointer()
		nextSlice := reflect.MakeSlice(typ, n, n)
		nextData := nextSlice.Pointer()
		const sliceSize = unsafe.Sizeof(sliceHeader{})
		for j := 0; j < n; j++ {
			// This is equivalent to h := slice[j*subsliceLen: (j+1)*subsliceLen]
			h := sliceHeader{
				Data: unsafe.Pointer(data + (uintptr(j*subsliceLen) * underlyingSize)),
				Len:  subsliceLen,
				Cap:  subsliceLen,
			}

			// This is equivalent to nSlice[j] = h
			*(*sliceHeader)(unsafe.Pointer(nextData + (uintptr(j) * sliceSize))) = h
		}

		slice = nextSlice
	}
	return slice
}

// decodeOneDimString decodes a string tensor into a one-dimensional []string.
func decodeOneDimString(raw []byte, nStrings int) ([]string, error) {
	// Start by making an array of all the strings
	strs := make([]string, nStrings)
	// The first nStrings * 8 bytes of raw are offsets into the second half of
	// the raw data. This second half is where the strings are encoded.
	offsets := (*(*[]int64)(unsafe.Pointer(&raw)))[:nStrings]

	// Reset raw after the offsets. Now the offsets will work relative to raw
	raw = raw[nStrings*8:]
	// Next we work out the final length of the string data so we can copy the
	// good data out of raw (which is owned by the C tensor and won't be safe
	// to access if the tensor is freed)
	r := bytes.NewReader(raw)
	var totalLength int
	for _, offset := range offsets {
		// At each offset we should find a varint length of a string.
		// Errors here should mean the tensor is corrupt.
		if _, err := r.Seek(offset, io.SeekStart); err != nil {
			return nil, err
		}
		l, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}
		totalLength += int(l)
	}

	// Lets allocate a big buffer to carry our string data.
	stringData := make([]byte, 0, totalLength)
	// Now copy the string data across into our new buffer, keeping track of the
	// location of each string in the strs slice.
	var cursor int
	for i, offset := range offsets {
		// At each offset we should find a varint length. Read it
		if _, err := r.Seek(offset, io.SeekStart); err != nil {
			return nil, err
		}
		l, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}

		// Then copy the actual string into our large buffer
		target := stringData[cursor : cursor+int(l)]
		if _, err := r.Read(target); err != nil {
			return nil, err
		}
		// Track where this string data is.
		strs[i] = *(*string)(unsafe.Pointer(&target))
		cursor += int(l)
	}

	// So now we have a big slice of strings
	return strs, nil
}

// WriteContentsTo writes the serialized contents of t to w.
//
// Returns the number of bytes written. See ReadTensor for
// reconstructing a Tensor from the serialized form.
//
// WARNING: WriteContentsTo is not comprehensive and will fail
// if t.DataType() is non-numeric (e.g., String). See
// https://github.com/tensorflow/tensorflow/issues/6003.
func (t *Tensor) WriteContentsTo(w io.Writer) (int64, error) {
	if err := isTensorSerializable(t.DataType()); err != nil {
		return 0, err
	}
	return io.Copy(w, bytes.NewReader(tensorData(t.c)))
}

func tensorData(c *C.TF_Tensor) []byte {
	// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	cbytes := C.TF_TensorData(c)
	if cbytes == nil {
		return nil
	}
	length := int(C.TF_TensorByteSize(c))
	var slice []byte
	if unsafe.Sizeof(unsafe.Pointer(nil)) == 8 {
		slice = (*[1<<50 - 1]byte)(unsafe.Pointer(cbytes))[:length:length]
	} else {
		slice = (*[1 << 30]byte)(unsafe.Pointer(cbytes))[:length:length]
	}
	return slice
}

var types = []struct {
	typ      reflect.Type
	dataType C.TF_DataType
}{
	{reflect.TypeOf(float32(0)), C.TF_FLOAT},
	{reflect.TypeOf(float64(0)), C.TF_DOUBLE},
	{reflect.TypeOf(int32(0)), C.TF_INT32},
	{reflect.TypeOf(uint32(0)), C.TF_UINT32},
	{reflect.TypeOf(uint8(0)), C.TF_UINT8},
	{reflect.TypeOf(int16(0)), C.TF_INT16},
	{reflect.TypeOf(int8(0)), C.TF_INT8},
	{reflect.TypeOf(""), C.TF_STRING},
	{reflect.TypeOf(complex(float32(0), float32(0))), C.TF_COMPLEX64},
	{reflect.TypeOf(int64(0)), C.TF_INT64},
	{reflect.TypeOf(uint64(0)), C.TF_UINT64},
	{reflect.TypeOf(false), C.TF_BOOL},
	{reflect.TypeOf(uint16(0)), C.TF_UINT16},
	{reflect.TypeOf(complex(float64(0), float64(0))), C.TF_COMPLEX128},
	// TODO(apassos): support DT_RESOURCE representation in go.
	// TODO(keveman): support DT_VARIANT representation in go.
}

// shapeAndDataTypeOf returns the data type and shape of the Tensor
// corresponding to a Go type.
func shapeAndDataTypeOf(val reflect.Value) (shape []int64, dt DataType, err error) {
	typ := val.Type()
	for typ.Kind() == reflect.Array || typ.Kind() == reflect.Slice {
		shape = append(shape, int64(val.Len()))
		if val.Len() > 0 {
			// In order to check tensor structure properly in general case we need to iterate over all slices of the tensor to check sizes match
			// Since we already going to iterate over all elements in encodeTensor() let's
			// 1) do the actual check in encodeTensor() to save some cpu cycles here
			// 2) assume the shape is represented by lengths of elements with zero index in each dimension
			val = val.Index(0)
		}
		typ = typ.Elem()
	}
	for _, t := range types {
		if typ.Kind() == t.typ.Kind() {
			return shape, DataType(t.dataType), nil
		}
	}
	return shape, dt, fmt.Errorf("unsupported type %v", typ)
}

func typeForDataType(dt DataType) reflect.Type {
	for _, t := range types {
		if dt == DataType(t.dataType) {
			return t.typ
		}
	}
	panic(bug("DataType %v is not supported (see https://www.tensorflow.org/code/tensorflow/core/framework/types.proto)", dt))
}

// typeOf converts from a DataType and Shape to the equivalent Go type.
func typeOf(dt DataType, shape []int64) reflect.Type {
	ret := typeForDataType(dt)
	for range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret
}

func numElements(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// byteSizeOfEncodedStrings returns the size of the encoded strings in val.
// val MUST be a string, or a container (array/slice etc.) of strings.
// Tensorflow encodes strings as the varint encoded length followed by the
// string bytes. We could call into the C library to do this but cgo has a heavy
// overhead. So we just do that calculation in Go
func byteSizeOfEncodedStrings(val reflect.Value) int {
	if val.Kind() == reflect.String {
		return sizeVarUint(uint64(val.Len())) + val.Len()
	}
	if val.Kind() != reflect.Slice && val.Kind() != reflect.Array {
		panic(fmt.Sprintf("unexpected type %s", val.Type()))
	}
	// Otherwise must be an array or slice.
	var size int
	for i := 0; i < val.Len(); i++ {
		size += byteSizeOfEncodedStrings(val.Index(i))
	}
	return size
}

// sizeVarUint determines how many bytes it would take to encode the int v as
// an unsigned varint
func sizeVarUint(v uint64) int {
	if v < 0x80 {
		return 1
	}
	bits := bits.Len64(v)
	return (bits + 6) / 7
}

// encodeTensorWithSlices writes v to the specified buffer using the format specified in
// c_api.h. Use stringEncoder for String tensors.
func encodeTensorWithSlices(w *bytes.Buffer, v reflect.Value, shape []int64) error {
	// If current dimension is a slice, verify that it has the expected size
	// Go's type system makes that guarantee for arrays.
	if v.Kind() == reflect.Slice {
		expected := int(shape[0])
		if v.Len() != expected {
			return fmt.Errorf("mismatched slice lengths: %d and %d", v.Len(), expected)
		}
	} else if v.Kind() != reflect.Array {
		return fmt.Errorf("unsupported type %v", v.Type())
	}

	// Once we have just a single dimension we can just copy the data
	if len(shape) == 1 && v.Len() > 0 {
		elt := v.Index(0)
		if !elt.CanAddr() {
			panic("cannot take address")
		}
		ptr := unsafe.Pointer(elt.Addr().Pointer())
		return copyPtr(w, ptr, v.Len()*int(elt.Type().Size()))
	}

	subShape := shape[1:]
	for i := 0; i < v.Len(); i++ {
		err := encodeTensorWithSlices(w, v.Index(i), subShape)
		if err != nil {
			return err
		}
	}

	return nil
}

// sliceHeader is a safer version of reflect.SliceHeader. Using unsafe.Pointer
// for Data reduces potential issues with the GC. The reflect package uses a
// similar struct internally.
type sliceHeader struct {
	Data unsafe.Pointer
	Len  int
	Cap  int
}

// copyPtr copies the backing data for a slice or array directly into w. Note
// we don't need to worry about byte ordering because we want the natural byte
// order for the machine we're running on.
func copyPtr(w *bytes.Buffer, ptr unsafe.Pointer, l int) error {
	h := sliceHeader{
		Data: ptr,
		Len:  l,
		Cap:  l,
	}
	// Convert our slice header into a []byte so we can call w.Write
	b := *(*[]byte)(unsafe.Pointer(&h))
	_, err := w.Write(b)
	return err
}

type stringEncoder struct {
	offsets *bytes.Buffer
	data    []byte
	offset  uint64
	status  *status
}

func (e *stringEncoder) encode(v reflect.Value, shape []int64) error {
	if v.Kind() == reflect.String {
		if err := copyPtr(e.offsets, unsafe.Pointer(&e.offset), int(unsafe.Sizeof(e.offset))); err != nil {
			return err
		}
		// A string is encoded as the varint length followed by the string bytes.
		// We do this in Go to avoid the considerable overhead of a cgo call into
		// the tensorflow library
		s := v.String()
		n := binary.PutUvarint(e.data[e.offset:], uint64(len(s)))
		e.offset += uint64(n)
		n = copy(e.data[e.offset:], s)
		e.offset += uint64(n)
		return nil
	}

	if v.Kind() == reflect.Slice {
		expected := int(shape[0])
		if v.Len() != expected {
			return fmt.Errorf("mismatched slice lengths: %d and %d", v.Len(), expected)
		}
	}

	subShape := shape[1:]
	for i := 0; i < v.Len(); i++ {
		if err := e.encode(v.Index(i), subShape); err != nil {
			return err
		}
	}
	return nil
}

func bug(format string, args ...interface{}) error {
	return fmt.Errorf("BUG: Please report at https://github.com/tensorflow/tensorflow/issues with the note: Go TensorFlow %v: %v", Version(), fmt.Sprintf(format, args...))
}

func isTensorSerializable(dataType DataType) error {
	// For numeric types, the serialized Tensor matches the in-memory
	// representation.  See the implementation of Tensor::AsProtoContent in
	// https://www.tensorflow.org/code/tensorflow/core/framework/tensor.cc
	//
	// The more appropriate way to be in sync with Tensor::AsProtoContent
	// would be to have the TensorFlow C library export functions for
	// serialization and deserialization of Tensors.  Till then capitalize
	// on knowledge of the implementation for numeric types.
	switch dataType {
	case Float, Double, Int32, Uint8, Int16, Int8, Complex, Int64, Bool, Quint8, Qint32, Bfloat16, Qint16, Quint16, Uint16, Complex128, Half:
		return nil
	default:
		return fmt.Errorf("serialization of tensors with the DataType %d is not yet supported, see https://github.com/tensorflow/tensorflow/issues/6003", dataType)
	}
}
