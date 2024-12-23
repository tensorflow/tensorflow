package tensorflow

// GPUOptions defines GPU-specific configurations.
type GPUOptions struct {
    AllowGrowth bool   // Whether GPU memory allocation should grow as needed.
    AllocatorType string // Type of GPU memory allocator to use.
}
