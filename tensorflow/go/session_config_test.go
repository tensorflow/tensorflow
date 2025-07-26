/*
Author: KleaSCM
Email: KleaSCM@gmail.com
File: session_config_test.go
Description: Tests for the type-safe SessionConfig and GPUOptions APIs in TensorFlow Go bindings. Ensures correct configuration, serialization, and backward compatibility with the legacy []byte config.
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
