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

package tensorflow

import "testing"

func TestSavedModel(t *testing.T) {
	bundle, err := LoadSavedModel("../cc/saved_model/testdata/half_plus_two/00000123", []string{"serve"}, nil)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}
	if op := bundle.Graph.Operation("y"); op == nil {
		t.Fatalf("\"y\" not found in graph")
	}
	t.Logf("SavedModel: %+v", bundle)
	// TODO(jhseu): half_plus_two has a tf.Example proto dependency to run. Add a
	// more thorough test when the generated protobufs are available.
}
