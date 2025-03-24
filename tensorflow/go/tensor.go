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

/*
#include <stdlib.h>
#include <string.h>
#include "tensorflow/c/c_api.h"

void toNewTString(GoString gstr, TF_TString *tstr) {
    TF_TString_Init(tstr);
    TF_TString_Copy(tstr, _GoStringPtr(gstr), _GoStringLen(gstr));
}
*/
import "C"

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/bits"
	"reflect"
	"runtime"
	"unsafe"
)

// DataType holds the type for a scalar value.  E.g., one slot in a tensor.
type DataType C.TF_DataType

// Types of scalar values in the TensorFlow type system.
const (
	Float             DataType = C.TF_FLOAT
	Double            DataType = C.TF_DOUBLE
	Int32             DataType = C.TF_INT32
	Uint32            DataType = C.TF_UINT32
	Uint8             DataType = C.TF_UINT8
	Int16             DataType = C.TF_INT16
	Int8              DataType = C.TF_INT8
	String            DataType = C.TF_STRING
	Complex64         DataType = C.TF_COMPLEX64
	Complex           DataType = C.TF_COMPLEX
	Int64             DataType = C.TF_INT64
	Uint64            DataType = C.TF_UINT64
	Bool              DataType = C.TF_BOOL
	Qint8             DataType = C.TF_QINT8
	Quint8            DataType = C.TF_QUINT8
	Qint32            DataType = C.TF_QINT32
	Bfloat16          DataType = C.TF_BFLOAT16
	Qint16            DataType = C.TF_QINT16
	Quint16           DataType = C.TF_QUINT16
	Uint16            DataType = C.TF_UINT16
	Complex128        DataType = C.TF_COMPLEX128
	Half              DataType = C.TF_HALF
	Float8e5m2        DataType = C.TF_FLOAT8_E5M2
	Float8e4m3fn      DataType = C.TF_FLOAT8_E4M3FN
	Float8e4m3fnuz    DataType = C.TF_FLOAT8_E4M3FNUZ
	Float8e4m3b11fnuz DataType = C.TF_FLOAT8_E4M3B11FNUZ
	Float8e5m2fnuz    DataType = C.TF_FLOAT8_E5M2FNUZ
	Int4              DataType = C.TF_INT4
	Uint4             DataType = C.TF_UINT4
)

// Tensor holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	c     *C.TF_Tensor
	shape []int64
}

// NewTensor converts from a Go value to a Tensor. Valid values are scalars,
// slices, and arrays. Every element of a slice must have the same length so
// that the resulting Tensor has a valid shape.
func NewTensor(value any) (*Tensor, error) {
	if value == nil {
		return nil, fmt.Errorf("cannot create tensor from nil value")
	}

	val := reflect.ValueOf(value)
	shape, dataType, err := shapeAndDataTypeOf(val)
	if err != nil {
		return nil, err
	}

	nflattened := numElements(shape)
	if nflattened < 0 {
		return nil, fmt.Errorf("invalid tensor shape: %v (overflow or negative dimension)", shape)
	}

	var nbytes uintptr
	if dataType == String {
		nbytes = uintptr(nflattened) * C.sizeof_TF_TString
		if uintptr(nflattened) > math.MaxUintptr/C.sizeof_TF_TString {
			return nil, fmt.Errorf("tensor too large")
		}
	} else {
		typeSize := TypeOf(dataType, nil).Size()
		if typeSize == 0 {
			return nil, fmt.Errorf("invalid type size for data type %v", dataType)
		}
		if uintptr(nflattened) > math.MaxUintptr/typeSize {
			return nil, fmt.Errorf("tensor too large")
		}
		nbytes = typeSize * uintptr(nflattened)
	}

	var shapePtr *C.int64_t
	if len(shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	t := &Tensor{
		c:     C.TF_AllocateTensor(C.TF_DataType(dataType), shapePtr, C.int(len(shape)), C.size_t(nbytes)),
		shape: shape,
	}
	if t.c == nil {
		return nil, fmt.Errorf("failed to allocate tensor")
	}

	raw := tensorData(t.c)
	if raw == nil {
		C.TF_DeleteTensor(t.c)
		return nil, fmt.Errorf("failed to get tensor data")
	}

	runtime.SetFinalizer(t, (*Tensor).finalize)

	buf := bytes.NewBuffer(raw[:0:len(raw)])

	if isAllArray(val.Type()) {
		if _, err := copyPtr(buf, unpackEFace(value).data, int(val.Type().Size())); err != nil {
			C.TF_DeleteTensor(t.c)
			return nil, err
		}
	} else {
		if _, err := encodeTensorWithSlices(buf, val, shape); err != nil {
			C.TF_DeleteTensor(t.c)
			return nil, err
		}
	}

	if uintptr(buf.Len()) != nbytes {
		C.TF_DeleteTensor(t.c)
		return nil, bug("NewTensor incorrectly calculated the size of a tensor with type %v and shape %v as %v bytes instead of %v", dataType, shape, nbytes, buf.Len())
	}
	return t, nil
}

// isAllArray returns true if type is a primitive type or an array of primitive
// types or an array of ... etc.. When this is true the data we want is
// contiguous in RAM.
func isAllArray(typ reflect.Type) bool {
	switch typ.Kind() {
	case reflect.String:
		return false
	case reflect.Slice:
		return false
	case reflect.Array:
		return isAllArray(typ.Elem())
	default:
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
// in an interface.
func unpackEFace(obj any) *eface {
	return (*eface)(unsafe.Pointer(&obj))
}

// ReadTensor constructs a Tensor with the provided type and shape from the
// serialized tensor contents in r.
func ReadTensor(dataType DataType, shape []int64, r io.Reader) (*Tensor, error) {
	if err := isTensorSerializable(dataType); err != nil {
		return nil, err
	}

	nflattened := numElements(shape)
	if nflattened < 0 {
		return nil, fmt.Errorf("invalid tensor shape: %v (overflow or negative dimension)", shape)
	}

	var shapePtr *C.int64_t
	if len(shape) > 0 {
		for _, dim := range shape {
			if dim < 0 {
				return nil, fmt.Errorf("all shape dimensions should be non-negative: %v", shape)
			}
		}
		shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))
	}

	var nbytes uintptr
	if dataType == String {
		nbytes = uintptr(nflattened) * C.sizeof_TF_TString
		if uintptr(nflattened) > math.MaxUintptr/C.sizeof_TF_TString {
			return nil, fmt.Errorf("tensor too large")
		}
	} else {
		typeSize := TypeOf(dataType, nil).Size()
		if typeSize == 0 {
			return nil, fmt.Errorf("invalid type size for data type %v", dataType)
		}
		if uintptr(nflattened) > math.MaxUintptr/typeSize {
			return nil, fmt.Errorf("tensor too large")
		}
		nbytes = typeSize * uintptr(nflattened)
	}

	t := &Tensor{
		c:     C.TF_AllocateTensor(C.TF_DataType(dataType), shapePtr, C.int(len(shape)), C.size_t(nbytes)),
		shape: shape,
	}
	if t.c == nil {
		return nil, fmt.Errorf("failed to allocate tensor")
	}

	runtime.SetFinalizer(t, (*Tensor).finalize)
	raw := tensorData(t.c)
	if raw == nil {
		C.TF_DeleteTensor(t.c)
		return nil, fmt.Errorf("failed to get tensor data")
	}

	if _, err := io.ReadFull(r, raw); err != nil {
		C.TF_DeleteTensor(t.c)
		return nil, err
	}
	return t, nil
}

// newTensorFromC takes ownership of c and returns the owning Tensor.
func newTensorFromC(c *C.TF_Tensor) *Tensor {
	if c == nil {
		return nil
	}

	var shape []int64
	if ndims := int(C.TF_NumDims(c)); ndims > 0 {
		shape = make([]int64, ndims)
		for i := range shape {
			dim := C.TF_Dim(c, C.int(i))
			if dim < 0 {
				C.TF_DeleteTensor(c)
				return nil
			}
			shape[i] = int64(dim)
		}
	}

	t := &Tensor{c: c, shape: shape}
	runtime.SetFinalizer(t, func(t *Tensor) {
		if t.c != nil {
			C.TF_DeleteTensor(t.c)
			t.c = nil
		}
	})
	return t
}

func (t *Tensor) finalize() {
	if t.c != nil {
		C.TF_DeleteTensor(t.c)
		t.c = nil
	}
}

// DataType returns the scalar datatype of the Tensor.
func (t *Tensor) DataType() DataType {
	if t == nil || t.c == nil {
		return 0
	}
	return DataType(C.TF_TensorType(t.c))
}

// Shape returns the shape of the Tensor.
func (t *Tensor) Shape() []int64 {
	if t == nil {
		return nil
	}
	return t.shape
}

// Reshape updates tensor's shape in place if possible or returns an error.
func (t *Tensor) Reshape(newShape []int64) error {
	if t == nil || t.c == nil {
		return fmt.Errorf("nil tensor")
	}

	oldShapeSize := numElements(t.shape)
	if oldShapeSize < 0 {
		return fmt.Errorf("invalid current shape: %v", t.shape)
	}

	newShapeSize := numElements(newShape)
	if newShapeSize < 0 {
		return fmt.Errorf("invalid new shape: %v", newShape)
	}

	if oldShapeSize != newShapeSize {
		return fmt.Errorf("unable to convert shape %v (num_elements: %d) into shape %v (num_elements: %d)", t.shape, oldShapeSize, newShape, newShapeSize)
	}

	if len(newShape) == 0 {
		return nil
	}

	var shapePtr *C.int64_t
	shapePtr = (*C.int64_t)(unsafe.Pointer(&newShape[0]))

	status := newStatus()
	C.TF_TensorBitcastFrom(t.c, C.TF_TensorType(t.c), t.c, shapePtr, C.int(len(newShape)), status.c)

	if err := status.Err(); err != nil {
		return err
	}
	t.shape = newShape
	return nil
}

// Value converts the Tensor to a Go value.
func (t *Tensor) Value() any {
	if t == nil || t.c == nil {
		return nil
	}

	raw := tensorData(t.c)
	if raw == nil {
		return nil
	}

	shape := t.Shape()
	dt := t.DataType()
	return decodeTensor(raw, shape, dt).Interface()
}

func decodeTensor(raw []byte, shape []int64, dt DataType) reflect.Value {
	n := int(numElements(shape))
	if n < 0 {
		panic("invalid shape")
	}

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
		if l < 0 {
			panic("size overflow")
		}
		typ = reflect.SliceOf(typ)
		slice = reflect.MakeSlice(typ, n, n)
		baseBytes := ([]byte)(unsafe.Pointer(&sliceHeader{
			Data: unsafe.Pointer(slice.Pointer()),
			Len:  l,
			Cap:  l,
		}))
		copy(baseBytes, raw)
	}

	if len(shape) == 0 {
		return slice.Index(0)
	}
	if len(shape) == 1 {
		return slice
	}

	if n == 0 {
		n = int(numElements(shape[:len(shape)-1]))
		if n < 0 {
			panic("invalid shape")
		}
	}

	for i := len(shape) - 2; i >= 0; i-- {
		underlyingSize := typ.Elem().Size()
		typ = reflect.SliceOf(typ)
		subsliceLen := int(shape[i+1])
		if subsliceLen != 0 {
			n = n / subsliceLen
		}

		data := unsafe.Pointer(slice.Pointer())
		nextSlice := reflect.MakeSlice(typ, n, n)

		for j := 0; j < n; j++ {
			setSliceInSlice(nextSlice, j, sliceHeader{
				Data: unsafe.Pointer(uintptr(data) + (uintptr(j*subsliceLen) * underlyingSize),
				Len:  subsliceLen,
				Cap:  subsliceLen,
			})
		}

		slice = nextSlice
	}
	return slice
}

// setSliceInSlice sets slice[index] = content.
func setSliceInSlice(slice reflect.Value, index int, content sliceHeader) {
	const sliceSize = unsafe.Sizeof(sliceHeader{})
	*(*sliceHeader)(unsafe.Pointer(uintptr(unsafe.Pointer(slice.Pointer())) + (uintptr(index) * sliceSize))) = content
}

// decodeOneDimString decodes a string tensor into a one-dimensional []string.
func decodeOneDimString(raw []byte, nStrings int) ([]string, error) {
	if len(raw) < nStrings*C.sizeof_TF_TString {
		return nil, fmt.Errorf("buffer too small for string tensor")
	}

	strs := make([]string, nStrings)
	tstrs := (([]C.TF_TString)(unsafe.Pointer(&sliceHeader{
		Data: unsafe.Pointer(&raw[0]),
		Len:  nStrings,
		Cap:  nStrings,
	})))[:nStrings]

	for i, tstr := range tstrs {
		dst := C.TF_TString_GetDataPointer(&tstr)
		dstLen := C.TF_TString_GetSize(&tstr)
		if dstLen < 0 {
			return nil, fmt.Errorf("invalid string length at index %d", i)
		}
		strs[i] = C.GoStringN(dst, C.int(dstLen))
	}

	return strs, nil
}

// WriteContentsTo writes the serialized contents of t to w.
func (t *Tensor) WriteContentsTo(w io.Writer) (int64, error) {
	if t == nil || t.c == nil {
		return 0, fmt.Errorf("nil tensor")
	}
	if err := isTensorSerializable(t.DataType()); err != nil {
		return 0, err
	}
	raw := tensorData(t.c)
	if raw == nil {
		return 0, fmt.Errorf("failed to get tensor data")
	}
	return io.Copy(w, bytes.NewReader(raw))
}

func tensorData(c *C.TF_Tensor) []byte {
	if c == nil {
		return nil
	}
	cbytes := C.TF_TensorData(c)
	if cbytes == nil {
		return nil
	}
	length := int(C.TF_TensorByteSize(c))
	if length <= 0 {
		return nil
	}

	var slice []byte
	if unsafe.Sizeof(unsafe.Pointer(nil)) == 8 {
		const maxLen = 1 << 50
		if length > maxLen {
			return nil
		}
		slice = (*[maxLen]byte)(unsafe.Pointer(cbytes))[:length:length]
	} else {
		const maxLen = 1 << 30
		if length > maxLen {
			return nil
		}
		slice = (*[maxLen]byte)(unsafe.Pointer(cbytes))[:length:length]
	}
	return slice
}

var types = []struct {
	typ      reflect.Type
	dataType C.TF_DataType
}{
