/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/dataset.h"

#include "ignite_dataset.h"

namespace tensorflow {

class KafkaDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
        std::string cache_name = "";
	OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "cache_name", &cache_name));
  	std::string host = "";
  	OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "host", &host));
  	int32 port = -1;
  	OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "port", &port));
  	bool local = false;
  	OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "local", &local));
  	int32 part = -1;
  	OP_REQUIRES_OK(ctx, ParseScalarArgument<int32>(ctx, "part", &part));
	
	const Tensor* schema_tensor;
	OP_REQUIRES_OK(ctx, ctx->input("schema", &schema_tensor));
	OP_REQUIRES(ctx, schema_tensor->dims() == 1, errors::InvalidArgument("`schema` must be a vector."));

	std::vector<int32> schema;
	schema.reserve(schema_tensor->NumElements());
	for (int i = 0; i < schema_tensor->NumElements(); i++) {
		schema.push_back(schema_tensor->flat<int32>()(i));
	}
	
    	*output = new ignite::IgniteDataset(ctx, cache_name, host, port, local, part, schema);
  }
};

REGISTER_KERNEL_BUILDER(Name("KafkaDataset").Device(DEVICE_CPU), KafkaDatasetOp);

}  // namespace tensorflow
