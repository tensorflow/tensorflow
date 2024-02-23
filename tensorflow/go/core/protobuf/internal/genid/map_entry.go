// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genid

import protoreflect "google.golang.org/protobuf/reflect/protoreflect"

// Generic field names and numbers for synthetic map entry messages.
const (
	MapEntry_Key_field_name   protoreflect.Name = "key"
	MapEntry_Value_field_name protoreflect.Name = "value"

	MapEntry_Key_field_number   protoreflect.FieldNumber = 1
	MapEntry_Value_field_number protoreflect.FieldNumber = 2
)
