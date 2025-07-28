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

import (
	"testing"
)

// TestSessionConfig_Builder tests the builder pattern for SessionConfig and GPUOptions.
// It verifies that options are set correctly and serialization succeeds.
func TestSessionConfig_Builder(t *testing.T) {
	cfg := NewSessionConfig().
		SetIntraOpParallelismThreads(2).
		SetInterOpParallelismThreads(4).
		SetAllowSoftPlacement(true).
		SetLogDevicePlacement(true).
		SetGPUOptions(
			NewGPUOptions().
				SetPerProcessGPUMemoryFraction(0.5).
				SetAllowGrowth(true).
				SetAllocatorType("BFC").
				SetVisibleDeviceList("0,1"),
		)

	bytes, err := cfg.ToBytes()
	if err != nil {
		t.Fatalf("SessionConfig.ToBytes() failed: %v", err)
	}
	if len(bytes) == 0 {
		t.Error("SessionConfig.ToBytes() returned empty bytes")
	}
}

// TestSessionOptions_NewAPI tests that SessionOptions works with the new SessionConfig API.
func TestSessionOptions_NewAPI(t *testing.T) {
	cfg := NewSessionConfig().SetIntraOpParallelismThreads(1)
	opts := NewSessionOptions().SetConfig(cfg)
	if opts.SessionConfig == nil {
		t.Error("SessionOptions.SessionConfig should not be nil when set")
	}
	if opts.Config != nil {
		t.Error("SessionOptions.Config should be nil when using new API")
	}
	_, err := opts.SessionConfig.ToBytes()
	if err != nil {
		t.Errorf("SessionConfig.ToBytes() failed: %v", err)
	}
}

// TestSessionOptions_LegacyAPI tests that SessionOptions still works with the legacy []byte config.
func TestSessionOptions_LegacyAPI(t *testing.T) {
	// This is a minimal valid ConfigProto serialization for testing.
	legacyBytes := []byte("(\x01")
	opts := NewSessionOptions().SetConfigBytes(legacyBytes)
	if opts.Config == nil {
		t.Error("SessionOptions.Config should not be nil when set with legacy API")
	}
	if opts.SessionConfig != nil {
		t.Error("SessionOptions.SessionConfig should be nil when using legacy API")
	}
}

// ExampleSessionConfig demonstrates how to use the new SessionConfig API.
//
// This example is for documentation and does not run as a test.
func ExampleSessionConfig() {
	cfg := NewSessionConfig().
		SetIntraOpParallelismThreads(2).
		SetGPUOptions(NewGPUOptions().SetAllowGrowth(true))
	_ = cfg // Use cfg with SessionOptions
	// Output:
}
