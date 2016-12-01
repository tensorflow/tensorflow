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

// #include "tensorflow/c/c_api.h"
import "C"

import "runtime"

type code C.TF_Code

// status holds error information returned by TensorFlow. We convert all
// TF statuses to Go errors.
type status struct {
	c *C.TF_Status
}

func newStatus() *status {
	s := &status{C.TF_NewStatus()}
	runtime.SetFinalizer(s, (*status).finalizer)
	return s
}

func (s *status) finalizer() {
	C.TF_DeleteStatus(s.c)
}

func (s *status) Code() code {
	return code(C.TF_GetCode(s.c))
}

func (s *status) String() string {
	return C.GoString(C.TF_Message(s.c))
}

// Err converts the status to a Go error and returns nil if the status is OK.
func (s *status) Err() error {
	if s == nil || s.Code() == C.TF_OK {
		return nil
	}
	return (*statusError)(s)
}

// statusError is distinct from status because it fulfills the error interface.
// status itself may have a TF_OK code and is not always considered an error.
//
// TODO(jhseu): Make public, rename to Error, and provide a way for users to
// check status codes.
type statusError status

func (s *statusError) Error() string {
	return (*status)(s).String()
}
