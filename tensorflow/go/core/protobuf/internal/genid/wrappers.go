// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genid

import protoreflect "google.golang.org/protobuf/reflect/protoreflect"

// Generic field name and number for messages in wrappers.proto.
const (
	WrapperValue_Value_field_name   protoreflect.Name        = "value"
	WrapperValue_Value_field_number protoreflect.FieldNumber = 1
)
