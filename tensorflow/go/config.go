package tensorflow

// ConfigProto mirrors tensorflow.ConfigProto.
// This is a Go-friendly representation that can later
// be serialized into protobuf bytes.
type ConfigProto struct {
    GPUOptions *GPUOptions
}

// GPUOptions mirrors tensorflow.GPUOptions.
type GPUOptions struct {
    AllowGrowth bool
}