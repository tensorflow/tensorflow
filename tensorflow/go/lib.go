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

// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../
//
// // TODO(ashankar): Remove this after TensorFlow 1.1 has been released.
// // Till then, the TensorFlow C API binary releases do not contain
// // the TF_DeletePRunHandle symbol. We work around that by
// // implementing the equivalent in session.cpp
// extern void tfDeletePRunHandle(const char*);
import "C"

func deletePRunHandle(h *C.char) {
	C.tfDeletePRunHandle(h)
}
