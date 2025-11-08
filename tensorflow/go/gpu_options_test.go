package tensorflow

import (
	"testing"
)

func TestMakeConfigProtoBytes(t *testing.T) {
	opts := &SessionOptions{
		Config: &Config{
			GPUOptions: &GPUOptions{
				AllowGrowth:                true,
				PerProcessGPUMemoryFraction: 0.25,
				VisibleDeviceList:          "0",
			},
		},
	}
	b, err := opts.makeConfigProtoBytes()
	if err != nil {
		t.Fatalf("makeConfigProtoBytes returned error: %v", err)
	}
	if len(b) == 0 {
		t.Fatalf("expected non-empty protobuf bytes")
	}
}
