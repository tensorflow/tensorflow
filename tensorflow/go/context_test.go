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
	"fmt"
	"testing"

	corepb "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto"
	"google.golang.org/protobuf/proto"
)

func TestContextConfigSetAsync(t *testing.T) {
	tests := []bool{false, true}
	for _, test := range tests {
		t.Run(fmt.Sprint(test), func(t *testing.T) {
			opt := &ContextOptions{Async: test}
			if _, err := NewContext(opt); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestContextConfigListDevices(t *testing.T) {
	c, err := NewContext(nil)
	if err != nil {
		t.Fatal(err)
	}
	devs, err := c.ListDevices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devs) < 1 {
		t.Fatalf("No devices found using ListDevices()")
	}
	foundCPUDevice := false
	for _, d := range devs {
		if d.Type == "CPU" {
			foundCPUDevice = true
		}
	}
	if !foundCPUDevice {
		t.Error("Failed to find CPU device using ListDevices()")
	}
}

// This example demonstrates use of the tensorflow.ConfigProto protocol buffer message
// to set context options.
func ExampleContextOptions_configProto() {
	// ConfigProto is a tensorflow.ConfigProto protocol buffer message.
	m := corepb.ConfigProto{
		// See ConfigProto documentation for a complete list of options.  The
		// options set in this example are only intended to demonstrate use.
		GpuOptions: &corepb.GPUOptions{
			AllowGrowth: true,
		},
		LogDevicePlacement: true,
	}

	// Marshal ConfigProto m to its binary serialized representation and into byte
	// slice b.
	b, err := proto.Marshal(&m)
	if err != nil {
		panic(err)
	}

	// ContextOptions accepts the binary serialized ConfigProto message.
	_ = ContextOptions{Config: b}
}
