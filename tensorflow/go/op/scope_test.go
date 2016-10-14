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

package op

import (
	"fmt"
	"testing"
)

func TestScopeSubScope(t *testing.T) {
	constant := func(s *Scope) string {
		c, err := Const(s, int64(1))
		if err != nil {
			t.Fatal(err)
		}
		return c.Op.Name()
	}
	var (
		root  = NewScope()
		sub1  = root.SubScope("x")
		sub2  = root.SubScope("x")
		sub1a = sub1.SubScope("y")
		sub2a = sub2.SubScope("y")
	)
	testdata := []struct {
		got, want string
	}{
		{constant(root), "Const"},
		{constant(sub1), "x/Const"},
		{constant(sub1a), "x/y/Const"},
		{constant(sub2), "x_1/Const"},
		{constant(sub2a), "x_1/y/Const"},
	}
	for idx, test := range testdata {
		if test.got != test.want {
			t.Errorf("#%d: Got %q, want %q", idx, test.got, test.want)
		}
	}

}

func ExampleScope_SubScope() {
	var (
		s     = NewScope()
		c1, _ = Const(s.SubScope("x"), int64(1))
		c2, _ = Const(s.SubScope("x"), int64(1))
	)
	fmt.Println(c1.Op.Name(), c2.Op.Name())
	// Output: x/Const x_1/Const
}
