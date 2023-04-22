/*
Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
// #include "tensorflow/c/c_api.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// makeCShape converts a shape specified in C.int64_t into a Shape.
func makeCShape(shape []C.int64_t) Shape {
	s := Shape{dims: make([]int64, len(shape))}
	for i, n := range shape {
		s.dims[i] = int64(n)
	}
	return s
}

// Attr returns the value of an attribute on op. It returns an error if the
// attribute does not exist.
func (op *Operation) Attr(name string) (interface{}, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	status := newStatus()
	meta := C.TF_OperationGetAttrMetadata(op.c, cname, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	if meta.is_list == 1 {
		return listAttribute(op, cname, meta)
	}
	return scalarAttribute(op, cname, meta)
}

func listAttribute(op *Operation, cname *C.char, meta C.TF_AttrMetadata) (interface{}, error) {
	status := newStatus()

	switch meta._type {
	case C.TF_ATTR_STRING:
		if meta.list_size == 0 {
			return []string(nil), nil
		}
		values := make([]unsafe.Pointer, meta.list_size)
		lengths := make([]C.size_t, meta.list_size)
		// Add one element in case total_size is zero.
		storage := make([]C.char, meta.total_size+1)
		C.TF_OperationGetAttrStringList(op.c, cname, &values[0], &lengths[0], C.int(meta.list_size), unsafe.Pointer(&storage[0]), C.size_t(meta.total_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		list := make([]string, meta.list_size)
		for i, val := range values {
			length := lengths[i]
			list[i] = C.GoStringN((*C.char)(val), C.int(length))
		}
		return list, nil

	case C.TF_ATTR_INT:
		if meta.list_size == 0 {
			return []int64(nil), nil
		}
		list := make([]C.int64_t, meta.list_size)
		C.TF_OperationGetAttrIntList(op.c, cname, &list[0], C.int(meta.list_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		vals := make([]int64, meta.list_size)
		for i, val := range list {
			vals[i] = int64(val)
		}
		return vals, nil

	case C.TF_ATTR_FLOAT:
		if meta.list_size == 0 {
			return []float32(nil), nil
		}
		list := make([]C.float, meta.list_size)
		C.TF_OperationGetAttrFloatList(op.c, cname, &list[0], C.int(meta.list_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		vals := make([]float32, meta.list_size)
		for i, val := range list {
			vals[i] = float32(val)
		}
		return vals, nil

	case C.TF_ATTR_BOOL:
		if meta.list_size == 0 {
			return []bool(nil), nil
		}
		list := make([]C.uchar, meta.list_size)
		C.TF_OperationGetAttrBoolList(op.c, cname, &list[0], C.int(meta.list_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		vals := make([]bool, meta.list_size)
		for i, val := range list {
			vals[i] = val == 1
		}
		return vals, nil

	case C.TF_ATTR_TYPE:
		if meta.list_size == 0 {
			return []DataType(nil), nil
		}
		list := make([]C.TF_DataType, meta.list_size)
		C.TF_OperationGetAttrTypeList(op.c, cname, &list[0], C.int(meta.list_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		vals := make([]DataType, meta.list_size)
		for i, val := range list {
			vals[i] = DataType(val)
		}
		return vals, nil

	case C.TF_ATTR_TENSOR:
		if meta.list_size == 0 {
			return []*Tensor(nil), nil
		}
		list := make([]*C.TF_Tensor, meta.list_size)
		C.TF_OperationGetAttrTensorList(op.c, cname, &list[0], C.int(meta.list_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		vals := make([]*Tensor, meta.list_size)
		for i, t := range list {
			vals[i] = newTensorFromC(t)
		}
		return vals, nil

	case C.TF_ATTR_SHAPE:
		if meta.list_size == 0 {
			return []Shape(nil), nil
		}
		dims := make([]*C.int64_t, meta.list_size)
		numDims := make([]C.int, meta.list_size)
		// Add one element in case total_size is zero.
		storage := make([]C.int64_t, meta.total_size+1)
		C.TF_OperationGetAttrShapeList(op.c, cname, &dims[0], &numDims[0], C.int(meta.list_size), &storage[0], C.int(meta.total_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		list := make([]Shape, meta.list_size)
		for i, dim := range dims {
			numDim := numDims[i]
			// If the number of dimensions is unknown, default to empty shape.
			if numDim < 0 {
				continue
			}
			// A []C.int64_t slice backed by C memory.
			// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
			// Using [1<<27] instead of [1<<30] so it works on 32-bit architecture
			slice := (*[1 << 27]C.int64_t)(unsafe.Pointer(dim))[:numDim:numDim]
			list[i] = makeCShape(slice)
		}
		return list, nil

	default:
		return nil, fmt.Errorf("list type %v not supported", meta._type)
	}
}

func scalarAttribute(op *Operation, cname *C.char, meta C.TF_AttrMetadata) (interface{}, error) {
	status := newStatus()

	switch meta._type {
	case C.TF_ATTR_STRING:
		if meta.total_size == 0 {
			return "", nil
		}
		v := make([]C.char, meta.total_size)
		C.TF_OperationGetAttrString(op.c, cname, unsafe.Pointer(&v[0]), C.size_t(meta.total_size), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		return C.GoStringN(&v[0], C.int(meta.total_size)), nil

	case C.TF_ATTR_INT:
		var v C.int64_t
		C.TF_OperationGetAttrInt(op.c, cname, &v, status.c)
		return int64(v), status.Err()

	case C.TF_ATTR_FLOAT:
		var v C.float
		C.TF_OperationGetAttrFloat(op.c, cname, &v, status.c)
		return float32(v), status.Err()

	case C.TF_ATTR_BOOL:
		var v C.uchar
		C.TF_OperationGetAttrBool(op.c, cname, &v, status.c)
		return v == 1, status.Err()

	case C.TF_ATTR_TYPE:
		var v C.TF_DataType
		C.TF_OperationGetAttrType(op.c, cname, &v, status.c)
		return DataType(v), status.Err()

	case C.TF_ATTR_TENSOR:
		var v *C.TF_Tensor
		C.TF_OperationGetAttrTensor(op.c, cname, &v, status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		return newTensorFromC(v), nil

	case C.TF_ATTR_SHAPE:
		numDims := meta.total_size
		// If number of dims is unknown return empty shape to indicate that.
		if numDims < 0 {
			return Shape{}, nil
		}
		if numDims == 0 {
			return ScalarShape(), nil
		}
		dims := make([]C.int64_t, numDims)
		C.TF_OperationGetAttrShape(op.c, cname, (*C.int64_t)(unsafe.Pointer(&dims[0])), C.int(numDims), status.c)
		if err := status.Err(); err != nil {
			return nil, err
		}
		return makeCShape(dims), nil

	default:
		return nil, fmt.Errorf("type %v not supported", meta._type)
	}
}
