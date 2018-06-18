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

import (
	"reflect"
	"testing"
)

func TestOperationAttrs(t *testing.T) {
	attrs := map[string]interface{}{
		"dtype": Float,
	}

	g := NewGraph()
	op, err := g.AddOperation(OpSpec{
		Type:  "Placeholder",
		Name:  "placeholder",
		Attrs: attrs,
	})
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range attrs {
		out, err := op.Attr(key)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("%q: Got %+v, wanted %+v", key, out, want)
		}
	}
}
