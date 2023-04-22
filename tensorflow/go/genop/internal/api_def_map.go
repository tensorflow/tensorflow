/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package internal

/*
#include <stdlib.h>
#include <string.h>

#include "tensorflow/c/c_api.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/golang/protobuf/proto"
	adpb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/api_def_go_proto"
	odpb "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/op_def_go_proto"
)

// Encapsulates a collection of API definitions.
//
// apiDefMap represents a map from operation name to corresponding
// ApiDef proto (see
// https://www.tensorflow.org/code/tensorflow/core/framework/api_def.proto
// for ApiDef proto definition).
type apiDefMap struct {
	c *C.TF_ApiDefMap
}

// Creates and returns a new apiDefMap instance.
//
// oplist is and OpList proto instance (see
// https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto
// for OpList proto definition).

func newAPIDefMap(oplist *odpb.OpList) (*apiDefMap, error) {
	// Create a buffer containing the serialized OpList.
	opdefSerialized, err := proto.Marshal(oplist)
	if err != nil {
		return nil, fmt.Errorf("could not serialize OpDef for %s", oplist.String())
	}
	data := C.CBytes(opdefSerialized)
	defer C.free(data)

	opbuf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(opbuf)
	opbuf.data = data
	opbuf.length = C.size_t(len(opdefSerialized))

	// Create ApiDefMap.
	status := C.TF_NewStatus()
	defer C.TF_DeleteStatus(status)
	capimap := C.TF_NewApiDefMap(opbuf, status)
	if C.TF_GetCode(status) != C.TF_OK {
		return nil, errors.New(C.GoString(C.TF_Message(status)))
	}
	apimap := &apiDefMap{capimap}
	runtime.SetFinalizer(
		apimap,
		func(a *apiDefMap) {
			C.TF_DeleteApiDefMap(a.c)
		})
	return apimap, nil
}

// Updates apiDefMap with the overrides specified in `data`.
//
// data - ApiDef text proto.
func (m *apiDefMap) Put(data string) error {
	cdata := C.CString(data)
	defer C.free(unsafe.Pointer(cdata))
	status := C.TF_NewStatus()
	defer C.TF_DeleteStatus(status)
	C.TF_ApiDefMapPut(m.c, cdata, C.size_t(len(data)), status)
	if C.TF_GetCode(status) != C.TF_OK {
		return errors.New(C.GoString(C.TF_Message(status)))
	}
	return nil
}

// Returns ApiDef proto instance for the TensorFlow operation
// named `opname`.
func (m *apiDefMap) Get(opname string) (*adpb.ApiDef, error) {
	cname := C.CString(opname)
	defer C.free(unsafe.Pointer(cname))
	status := C.TF_NewStatus()
	defer C.TF_DeleteStatus(status)
	apidefBuf := C.TF_ApiDefMapGet(
		m.c, cname, C.size_t(len(opname)), status)
	defer C.TF_DeleteBuffer(apidefBuf)
	if C.TF_GetCode(status) != C.TF_OK {
		return nil, errors.New(C.GoString(C.TF_Message(status)))
	}
	if apidefBuf == nil {
		return nil, fmt.Errorf("could not find ApiDef for %s", opname)
	}

	var (
		apidef = new(adpb.ApiDef)
		size   = int(apidefBuf.length)
		// A []byte backed by C memory.
		// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
		data = (*[1 << 30]byte)(unsafe.Pointer(apidefBuf.data))[:size:size]
		err  = proto.Unmarshal(data, apidef)
	)
	if err != nil {
		return nil, err
	}
	return apidef, nil
}
