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

// Package internal generates Go source code with functions for TensorFlow operations.
//
// The generated APIs are unstable and can change without notice.
package internal

// #include "tensorflow/c/c_api.h"
import "C"

import (
	"fmt"
	"io"
	"unsafe"

	"github.com/golang/protobuf/proto"
	pb "github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/tensorflow/core/framework"
)

// GenerateFunctionsForRegisteredOps writes a Go source code file to w
// containing functions for each TensorFlow operation registered in the address
// space of the calling process.
func GenerateFunctionsForRegisteredOps(w io.Writer) error {
	ops, err := registeredOps()
	if err != nil {
		return err
	}
	fmt.Fprintf(w, `// DO NOT EDIT
// This file was machine generated.
//
// This code generation process is a work in progress and is not ready yet.
// Eventually, the code generator will generate approximately %d wrapper
// functions for adding TensorFlow operations to a Graph.

package op
`, len(ops.Op))
	return nil
}

func registeredOps() (*pb.OpList, error) {
	buf := C.TF_GetAllOpList()
	defer C.TF_DeleteBuffer(buf)
	var (
		list = new(pb.OpList)
		size = int(buf.length)
		// A []byte backed by C memory.
		// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
		data = (*[1 << 30]byte)(unsafe.Pointer(buf.data))[:size:size]
		err  = proto.Unmarshal(data, list)
	)
	return list, err
}
