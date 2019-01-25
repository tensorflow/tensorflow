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
	"strings"
	"testing"
)

func TestNewTensorHandle(t *testing.T) {
	vals := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	tensor, err := NewTensor(vals)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = NewTensorHandle(tensor); err != nil {
		t.Fatal(err)
	}
}

func TestTensorHandleDataType(t *testing.T) {
	vals := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	tensor, err := NewTensor(vals)
	if err != nil {
		t.Fatal(err)
	}
	th, err := NewTensorHandle(tensor)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := th.DataType(), Float; got != want {
		t.Errorf("Got %v, want %v", got, want)
	}
}

func TestTensorHandleShape(t *testing.T) {
	vals := [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
	tensor, err := NewTensor(vals)
	if err != nil {
		t.Fatal(err)
	}
	th, err := NewTensorHandle(tensor)
	if err != nil {
		t.Fatal(err)
	}

	got, err := th.Shape()
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{2, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("Got %#v, want %#v", got, want)
	}
}

func TestTensorHandleDeviceName(t *testing.T) {
	vals := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	tensor, err := NewTensor(vals)
	if err != nil {
		t.Fatal(err)
	}
	th, err := NewTensorHandle(tensor)
	if err != nil {
		t.Fatal(err)
	}

	d, err := th.DeviceName()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(d, "CPU") {
		t.Errorf("DeviceName() did not return a CPU device; got: %s", d)
	}
}

func TestTensorHandleBackingDeviceName(t *testing.T) {
	vals := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	tensor, err := NewTensor(vals)
	if err != nil {
		t.Fatal(err)
	}
	th, err := NewTensorHandle(tensor)
	if err != nil {
		t.Fatal(err)
	}

	d, err := th.BackingDeviceName()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(d, "CPU") {
		t.Errorf("BackingDeviceName() did not return a CPU device; got: %s", d)
	}
}

func TestTensorHandleToTensor(t *testing.T) {
	initialVals := [][]float32{{1.0, 2.0}, {3.0, 4.0}}
	initialTensor, err := NewTensor(initialVals)
	if err != nil {
		t.Fatal(err)
	}
	th, err := NewTensorHandle(initialTensor)
	if err != nil {
		t.Fatal(err)
	}

	tensor, err := th.ToTensor()
	if v := tensor.Value().([][]float32); !reflect.DeepEqual(v, initialVals) {
		t.Errorf("Got %#v, want %#v", v, initialVals)
	}
}
