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
	"fmt"
	"runtime"
)

// ContextOptions contains configuration information for a session
type ContextOptions struct {
	// Config is a binary-serialized representation of the
	// tensorflow.ConfigProto protocol message
	// (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
	Config []byte

	// Sets the default execution mode
	Async bool
}

// c converts the ContextOptions to the C API's TF_ContextOptions.
// Caller takes ownership of returned object.
func (o *ContextOptions) c() (*C.TFE_ContextOptions, error) {
	opt := C.TFE_NewContextOptions()
	if o == nil {
		return opt, nil
	}

	if sz := len(o.Config); sz > 0 {
		status := newStatus()
		cConfig := C.CBytes(o.Config)
		C.TFE_ContextOptionsSetConfig(opt, cConfig, C.size_t(sz), status.c)
		C.free(cConfig)
		if err := status.Err(); err != nil {
			C.TFE_DeleteContextOptions(opt)
			return nil, fmt.Errorf("invalid ContextOptions.Config: %v", err)
		}
	}

	var async uint8
	if o.Async {
		async = 1
	}
	C.TFE_ContextOptionsSetAsync(opt, C.uchar(async))

	return opt, nil
}

// Context for executing operations eagerly.
//
// A Context allows operations to be executed immediately. It encapsulates
// information such as the available devices, resource manager etc. It also
// allows the user to configure execution using a ConfigProto, as they can
// configure a Session when executing a Graph.
type Context struct {
	c *C.TFE_Context
}

// NewContext creates a new context for eager execution.
// options may be nil to use the default options.
func NewContext(options *ContextOptions) (*Context, error) {
	status := newStatus()
	cOpt, err := options.c()
	if err != nil {
		return nil, err
	}
	defer C.TFE_DeleteContextOptions(cOpt)
	cContext := C.TFE_NewContext(cOpt, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	c := &Context{c: cContext}
	runtime.SetFinalizer(c, (*Context).finalizer)
	return c, nil
}

func (c *Context) finalizer() {
	C.TFE_DeleteContext(c.c)
}

// ListDevices returns the list of devices associated with a Context.
func (c *Context) ListDevices() ([]Device, error) {
	status := newStatus()
	devicesList := C.TFE_ContextListDevices(c.c, status.c)
	if err := status.Err(); err != nil {
		return nil, fmt.Errorf("SessionListDevices() failed: %v", err)
	}
	defer C.TF_DeleteDeviceList(devicesList)
	return deviceSliceFromDeviceList(devicesList)
}
