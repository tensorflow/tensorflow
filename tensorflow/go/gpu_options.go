package tensorflow

// GPUOptions represents configuration options for controlling GPU behavior
// in TensorFlow sessions. This struct corresponds to fields in
// ConfigProto.GPUOptions (see tensorflow/core/protobuf/config.proto).
type GPUOptions struct {
	// AllowGrowth, if true, enables dynamic allocation of GPU memory.
	// Instead of pre-allocating all available memory on the device,
	// TensorFlow will allocate memory as needed.
	AllowGrowth bool

	// PerProcessGPUMemoryFraction specifies the fraction of the total
	// available GPU memory that this process is allowed to allocate.
	// A value of 0 means the entire GPU memory can be allocated.
	// Example: 0.5 = use up to 50% of GPU memory.
	PerProcessGPUMemoryFraction float64

	// VisibleDeviceList allows specifying which GPUs are visible to
	// the process, using a comma-separated list of GPU IDs, e.g. "0,1".
	// Leave empty to make all GPUs visible.
	VisibleDeviceList string
}
