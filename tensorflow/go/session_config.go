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
	corepb "github.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto"
	"google.golang.org/protobuf/proto"
)

// SessionConfig provides a type-safe, modular API for configuring TensorFlow sessions in Go.
// This struct wraps the generated ConfigProto protobuf and exposes common configuration options
// through idiomatic Go methods. It is intended to replace the legacy []byte config API.
type SessionConfig struct {
	config *corepb.ConfigProto
}

// NewSessionConfig creates a new SessionConfig with default settings.
// Returns a pointer to a SessionConfig struct ready for further configuration.
func NewSessionConfig() *SessionConfig {
	return &SessionConfig{
		config: &corepb.ConfigProto{},
	}
}

// SetIntraOpParallelismThreads sets the number of threads used for parallelizing individual operations.
// This controls intra-op parallelism. A value of 0 lets TensorFlow pick an appropriate default.
func (sc *SessionConfig) SetIntraOpParallelismThreads(threads int32) *SessionConfig {
	sc.config.IntraOpParallelismThreads = threads
	return sc
}

// SetInterOpParallelismThreads sets the number of threads used for parallelizing independent operations.
// This controls inter-op parallelism. A value of 0 lets TensorFlow pick an appropriate default.
func (sc *SessionConfig) SetInterOpParallelismThreads(threads int32) *SessionConfig {
	sc.config.InterOpParallelismThreads = threads
	return sc
}

// SetGPUOptions sets the GPUOptions for this session configuration.
// Use the GPUOptions builder to construct the options before passing them here.
func (sc *SessionConfig) SetGPUOptions(options *GPUOptions) *SessionConfig {
	sc.config.GpuOptions = options.proto
	return sc
}

// SetAllowSoftPlacement enables or disables soft device placement.
// When enabled, operations will be placed on CPU if no GPU implementation is available.
func (sc *SessionConfig) SetAllowSoftPlacement(allow bool) *SessionConfig {
	sc.config.AllowSoftPlacement = allow
	return sc
}

// SetLogDevicePlacement enables or disables logging of device placement decisions.
func (sc *SessionConfig) SetLogDevicePlacement(log bool) *SessionConfig {
	sc.config.LogDevicePlacement = log
	return sc
}

// ToBytes serializes the session configuration to a byte slice suitable for use with the TensorFlow C API.
// Returns the serialized protobuf or an error if serialization fails.
func (sc *SessionConfig) ToBytes() ([]byte, error) {
	return proto.Marshal(sc.config)
}

// GPUOptions provides a builder-style API for configuring GPU-specific session options.
// This struct wraps the generated GPUOptions protobuf.
type GPUOptions struct {
	proto *corepb.GPUOptions
}

// NewGPUOptions creates a new GPUOptions struct with default settings.
// Returns a pointer to a GPUOptions struct ready for further configuration.
func NewGPUOptions() *GPUOptions {
	return &GPUOptions{
		proto: &corepb.GPUOptions{},
	}
}

// SetPerProcessGPUMemoryFraction sets the fraction of GPU memory to allocate per process.
// 1.0 means allocate all memory, 0.5 means allocate 50%, etc.
func (goOpts *GPUOptions) SetPerProcessGPUMemoryFraction(fraction float64) *GPUOptions {
	goOpts.proto.PerProcessGpuMemoryFraction = fraction
	return goOpts
}

// SetAllowGrowth enables or disables incremental GPU memory allocation.
// When enabled, TensorFlow will allocate memory as needed rather than pre-allocating all memory.
func (goOpts *GPUOptions) SetAllowGrowth(allow bool) *GPUOptions {
	goOpts.proto.AllowGrowth = allow
	return goOpts
}

// SetAllocatorType sets the GPU memory allocator type (e.g., "BFC" for best-fit with coalescing).
func (goOpts *GPUOptions) SetAllocatorType(allocatorType string) *GPUOptions {
	goOpts.proto.AllocatorType = allocatorType
	return goOpts
}

// SetVisibleDeviceList sets the list of visible GPU devices (e.g., "0,1").
// This is similar to setting CUDA_VISIBLE_DEVICES.
func (goOpts *GPUOptions) SetVisibleDeviceList(deviceList string) *GPUOptions {
	goOpts.proto.VisibleDeviceList = deviceList
	return goOpts
}
