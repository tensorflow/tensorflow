// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

// #include <string.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"unsafe"
)

// DataType holds the type for a scalar value.  E.g., one slot in a tensor.
// The values here are identical to corresponding values in types.proto.
type DataType C.TF_DataType

// Tensor holds a multi-dimensional array of elements of a single data type.
type Tensor struct {
	// We create TF_Tensor on demand rather than keep a handle to C.TF_Tensor
	// because many functions, such as Session.Run() and Operations take
	// ownership of the C.TF_Tensor. Translating on-demand provides for a safe
	// API.
	//
	// A memcpy is required because cgo rules prohibit us from maintaining
	// a pointer to Go memory.
	// call: https://golang.org/cmd/cgo/
	buf   *bytes.Buffer
	dt    DataType
	shape []int64
}

// NewTensor converts from a Go value to a Tensor. Valid values are scalars,
// slices, and arrays. Every element of a slice must have the same length so
// that the resulting Tensor has a valid shape.
func NewTensor(value interface{}) (*Tensor, error) {
	val := reflect.ValueOf(value)
	dims, dataType, err := dimsAndDataTypeOf(val.Type())
	if err != nil {
		return nil, err
	}
	t := &Tensor{buf: bytes.NewBuffer(nil), dt: dataType, shape: make([]int64, dims)}
	if err = encodeTensor(t.buf, t.shape, val); err != nil {
		return nil, err
	}
	return t, nil
}

// newTensorFromC converts from a C.TF_Tensor to a Tensor.
func newTensorFromC(ct *C.TF_Tensor) *Tensor {
	t := &Tensor{dt: DataType(C.TF_TensorType(ct))}
	numDims := int(C.TF_NumDims(ct))
	for i := 0; i < numDims; i++ {
		t.shape = append(t.shape, int64(C.TF_Dim(ct, C.int(i))))
	}
	b := make([]byte, int(C.TF_TensorByteSize(ct)))
	if len(b) > 0 {
		C.memcpy(unsafe.Pointer(&b[0]), C.TF_TensorData(ct), C.size_t(len(b)))
	}
	t.buf = bytes.NewBuffer(b)
	return t
}

// DataType returns the scalar datatype of the Tensor.
func (t *Tensor) DataType() DataType {
	return t.dt
}

// Shape returns the shape of the Tensor.
func (t *Tensor) Shape() []int64 {
	return t.shape
}

// Value converts the Tensor to a Go value. For now, not all Tensor types are
// supported, and this function may panic if it encounters an unsupported
// DataType.
//
// The type of the output depends on the Tensor type and dimensions.
// For example:
// Tensor(int64, 0): int64
// Tensor(float64, 3): [][][]float64
func (t *Tensor) Value() interface{} {
	typ, err := typeOf(t.DataType(), t.Shape())
	if err != nil {
		panic(err)
	}
	val := reflect.New(typ)
	if err := decodeTensor(t.buf, t.Shape(), typ, val); err != nil {
		panic(err)
	}
	return reflect.Indirect(val).Interface()
}

// c converts the Tensor to a *C.TF_Tensor. Callers must take ownership of
// the *C.TF_Tensor, either by passing ownership to the C API or explicitly
// calling C.TF_DeleteTensor() on it.
func (t *Tensor) c() *C.TF_Tensor {
	var shapePtr *C.int64_t
	if len(t.shape) > 0 {
		shapePtr = (*C.int64_t)(unsafe.Pointer(&t.shape[0]))
	}
	tensor := C.TF_AllocateTensor(C.TF_DataType(t.dt), shapePtr, C.int(len(t.shape)), C.size_t(t.buf.Len()))
	if t.buf.Len() > 0 {
		slice := t.buf.Bytes() // https://github.com/golang/go/issues/14210
		C.memcpy(C.TF_TensorData(tensor), unsafe.Pointer(&slice[0]), C.size_t(t.buf.Len()))
	}
	return tensor
}

// deleteCTensor only exists to delete C.TF_Tensors in tests. go test doesn't
// support cgo.
func deleteCTensor(ct *C.TF_Tensor) {
	C.TF_DeleteTensor(ct)
}

var types = []struct {
	typ      reflect.Type
	dataType C.TF_DataType
}{
	{reflect.TypeOf(float32(0)), C.TF_FLOAT},
	{reflect.TypeOf(float64(0)), C.TF_DOUBLE},
	{reflect.TypeOf(int32(0)), C.TF_INT32},
	{reflect.TypeOf(uint8(0)), C.TF_UINT8},
	{reflect.TypeOf(int16(0)), C.TF_INT16},
	{reflect.TypeOf(int8(0)), C.TF_INT8},
	{reflect.TypeOf(""), C.TF_STRING},
	{reflect.TypeOf(complex(float32(0), float32(0))), C.TF_COMPLEX64},
	{reflect.TypeOf(int64(0)), C.TF_INT64},
	{reflect.TypeOf(false), C.TF_BOOL},
	{reflect.TypeOf(uint16(0)), C.TF_UINT16},
	{reflect.TypeOf(complex(float64(0), float64(0))), C.TF_COMPLEX128},
}

// dimsAndDataTypeOf returns the data type and dimensions of a Go type for use
// when encoding. We fetch them separately from encoding to support 0-D tensors.
func dimsAndDataTypeOf(typ reflect.Type) (int, DataType, error) {
	dims := 0
	elem := typ
	for ; elem.Kind() == reflect.Array || elem.Kind() == reflect.Slice; elem = elem.Elem() {
		dims++
	}
	for _, t := range types {
		if elem.Kind() == t.typ.Kind() {
			return dims, DataType(t.dataType), nil
		}
	}
	return 0, DataType(0), fmt.Errorf("unsupported type %v", typ)
}

// typeOf converts from a DataType and Shape to the equivalent Go type.
func typeOf(dt DataType, shape []int64) (reflect.Type, error) {
	var ret reflect.Type
	for _, t := range types {
		if dt == DataType(t.dataType) {
			ret = t.typ
			break
		}
	}
	if ret == nil {
		return nil, fmt.Errorf("DataType %v unsupported", dt)
	}
	for _ = range shape {
		ret = reflect.SliceOf(ret)
	}
	return ret, nil
}

// encodeTensor writes v to the specified buffer using the format specified in
// c_api.h
func encodeTensor(buf *bytes.Buffer, shape []int64, v reflect.Value) error {
	switch v.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Write(buf, nativeEndian, v.Interface()); err != nil {
			return err
		}

	case reflect.Array, reflect.Slice:
		// If slice elements are slices, verify that all of them have the same size.
		// Go's type system makes that guarantee for arrays.
		if v.Len() > 0 && v.Type().Elem().Kind() == reflect.Slice {
			expected := v.Index(0).Len()
			for i := 1; i < v.Len(); i++ {
				if v.Index(i).Len() != expected {
					return fmt.Errorf("mismatched slice lengths: %d and %d", v.Index(i).Len(), expected)
				}
			}
		}

		shape[0] = int64(v.Len())
		for i := 0; i < v.Len(); i++ {
			err := encodeTensor(buf, shape[1:], v.Index(i))
			if err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", v.Type())
	}
	return nil
}

// decodeTensor decodes the Tensor from the buffer to ptr using the format
// specified in c_api.h
func decodeTensor(buf *bytes.Buffer, shape []int64, typ reflect.Type, ptr reflect.Value) error {
	switch typ.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		if err := binary.Read(buf, nativeEndian, ptr.Interface()); err != nil {
			return err
		}

	case reflect.Slice:
		val := reflect.Indirect(ptr)
		val.Set(reflect.MakeSlice(typ, int(shape[0]), int(shape[0])))
		for i := 0; i < val.Len(); i++ {
			if err := decodeTensor(buf, shape[1:], typ.Elem(), val.Index(i).Addr()); err != nil {
				return err
			}
		}

	default:
		return fmt.Errorf("unsupported type %v", typ)
	}
	return nil
}

// nativeEndian is the byte order for the local platform. Used to send back and
// forth Tensors with the C API. We test for endianness at runtime because
// some architectures can be booted into different endian modes.
var nativeEndian binary.ByteOrder

func init() {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		nativeEndian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		nativeEndian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
}
