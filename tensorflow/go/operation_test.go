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

import (
	"runtime"
	"runtime/debug"
	"testing"
)

// createGraphAndOp creates an Operation but loses the reference to the Graph.
func createGraphAndOp() (*Operation, error) {
	t, err := NewTensor(int64(1))
	if err != nil {
		return nil, err
	}
	g := NewGraph()
	output, err := Placeholder(g, "my_placeholder", t.DataType())
	if err != nil {
		return nil, err
	}
	return output.Op, nil
}

func TestOperationLifetime(t *testing.T) {
	// Ensure that the Graph is not garbage collected while the program
	// still has access to the Operation.
	op, err := createGraphAndOp()
	if err != nil {
		t.Fatal(err)
	}
	forceGC()
	if got, want := op.Name(), "my_placeholder"; got != want {
		t.Errorf("Got '%s', want '%s'", got, want)
	}
	if got, want := op.Type(), "Placeholder"; got != want {
		t.Errorf("Got '%s', want '%s'", got, want)
	}
}

func forceGC() {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	// It was empirically observed that without this extra allocation
	// TestOperationLifetime would fail only 50% of the time if
	// Operation did not hold on to a reference to Graph. With this
	// additional allocation, and with the bug where Operation does
	// not hold onto a Graph, the test failed 90+% of the time.
	//
	// The author is aware that this technique is potentially fragile
	// and fishy. Suggestions for alternatives are welcome.
	bytesTillGC := mem.NextGC - mem.HeapAlloc + 1
	_ = make([]byte, bytesTillGC)
	runtime.GC()
	debug.FreeOSMemory()
}
