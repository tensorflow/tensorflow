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
// #include "tensorflow/c/eager/c_api.h"
import "C"
import (
	"runtime"
	"unsafe"
)

// TensorHandle is a handle to a tensor on a device.
//
// A Tensor referenced by a TensorHandle may be on any device, whereas a Tensor
// always resides in the host CPU's memory.
//
// A Tensor referenced by a TensorHandle may not have been computed yet. For
// example, a TensorHandle might reference the output of an operation that has
// not finished executing. Because of this, various methods, such as Shape() may
// block until the tensor has been instantiated.
//
// This allows multiple operations to be performed on tensors on a device
// (e.g. a GPU) without sending these values back to the host CPU in between
// every operation.
type TensorHandle struct {
	c *C.TFE_TensorHandle
}

// NewTensorHandle creates a new tensor handle from a tensor.
func NewTensorHandle(t *Tensor) (*TensorHandle, error) {
	status := newStatus()
	cHandle := C.TFE_NewTensorHandle(t.c, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	th := &TensorHandle{c: cHandle}
	runtime.SetFinalizer(th, (*TensorHandle).finalizer)
	return th, nil
}

func (th *TensorHandle) finalizer() {
	C.TFE_DeleteTensorHandle(th.c)
}

// newTensorHandleFromC takes ownership of c and returns the owning TensorHandle.
func newTensorHandleFromC(c *C.TFE_TensorHandle) *TensorHandle {
	th := &TensorHandle{c: c}
	runtime.SetFinalizer(th, (*TensorHandle).finalizer)
	return th
}

// DataType returns the TensorHandle's datatype.
func (th *TensorHandle) DataType() DataType {
	return DataType(C.TFE_TensorHandleDataType(th.c))
}

// Shape returns the shape of the Tensor referenced by th.
func (th *TensorHandle) Shape() ([]int64, error) {
	n, err := th.numDims()
	if err != nil {
		return nil, err
	}
	r := make([]int64, n)
	for i := 0; i < n; i++ {
		if r[i], err = th.dim(i); err != nil {
			return nil, err
		}
	}
	return r, nil
}

// numDims returns the number of dimensions of the TensorHandle. It blocks
// until the operation that produces the handle has completed.
func (th *TensorHandle) numDims() (int, error) {
	status := newStatus()
	n := int(C.TFE_TensorHandleNumDims(th.c, status.c))
	return n, status.Err()
}

// dim returns the size of the index'th dimension of the TensorHandle. It
// blocks until the operation that produces the handle has completed.
func (th *TensorHandle) dim(index int) (int64, error) {
	status := newStatus()
	n := int64(C.TFE_TensorHandleDim(th.c, C.int(index), status.c))
	if err := status.Err(); err != nil {
		return 0, err
	}
	return n, nil
}

// DeviceName returns the name of the device of the operation that produced the
// TensorHandle. If the handle was produced by a copy, it returns the
// destination device of the copy. Note that returned device name is not always
// the device holding the tensor handle's memory. If you want the latter, use
// BackingDeviceName. This function will block till the operation that produces
// th has completed.
func (th *TensorHandle) DeviceName() (string, error) {
	status := newStatus()
	name := C.TFE_TensorHandleDeviceName(th.c, status.c)
	if err := status.Err(); err != nil {
		return "", err
	}
	return C.GoString(name), nil
}

// BackingDeviceName returns the name of the device in whose memory the tensor
// handle resides. This function will block till the operation that produces
// `h` has completed.
//
// WARNING: The implementation currently returns the same as DeviceName().
// After TensoFlow 1.13's C library is released, this implementation will
// be updated to return what the documentation says!
func (th *TensorHandle) BackingDeviceName() (string, error) {
	// TODO(ashankar): Restore after TensorFlow 1.13 is released.
	// See https://github.com/tensorflow/tensorflow/issues/23257#issuecomment-433751410
	return th.DeviceName()
	/*
	status := newStatus()
	name := C.TFE_TensorHandleBackingDeviceName(th.c, status.c)
	if err := status.Err(); err != nil {
		return "", err
	}
	return C.GoString(name), nil
	*/
}

// ToTensor returns the Tensor referenced by th. It may block if this tensor is
// not yet computed.
func (th *TensorHandle) ToTensor() (*Tensor, error) {
	status := newStatus()
	cTensor := C.TFE_TensorHandleResolve(th.c, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	return newTensorFromC(cTensor), nil
}

// CopyToDevice creates a new TensorHandle with the same contents as this
// TensorHandle but placed in the memory of the device 'deviceName'. If source
// and destination are the same device, then this creates a new handle that
// shares the underlying buffer. Otherwise, it currently requires at least one
// of the source or destination devices to be CPU (i.e., for the source or
// destination tensor to be placed in host memory).
func (th *TensorHandle) CopyToDevice(c *Context, deviceName string) (*TensorHandle, error) {
	status := newStatus()
	n := C.CString(deviceName)
	newTh := C.TFE_TensorHandleCopyToDevice(th.c, c.c, n, status.c)
	C.free(unsafe.Pointer(n))
	if err := status.Err(); err != nil {
		return nil, err
	}
	return newTensorHandleFromC(newTh), nil
}
