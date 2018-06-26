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

#include "ignite_dataset_iterator.h"

namespace ignite {

IgniteDataset::IgniteDataset(tensorflow::OpKernelContext* ctx, std::string cache_name, std::string host, tensorflow::int32 port, bool local, tensorflow::int32 part, std::vector<tensorflow::int32> schema, std::vector<tensorflow::int32> permutation) 
  : GraphDatasetBase(ctx),
    cache_name_(cache_name),
    host_(host),
    port_(port),
    local_(local),
    part_(part),
    schema_(schema),
    permutation_(permutation) {
  // Print schema    
	std::cout << "Schema: ";
	for (auto e: schema_) {
		std::cout << e << " ";
	}	
	std::cout << "\n";
  // Print permutation
  std::cout << "Permutation: ";
  for (auto e: permutation_) {
    std::cout << e << " ";
  } 
  std::cout << "\n";
}

std::unique_ptr<tensorflow::IteratorBase> IgniteDataset::MakeIteratorInternal(const tensorflow::string& prefix) const {
  return std::unique_ptr<tensorflow::IteratorBase>(new IgniteDatasetIterator({this, tensorflow::strings::StrCat(prefix, "::Kafka")}, 
    this->host_, this->port_, this->cache_name_, this->local_, this->part_, this->schema_, this->permutation_));
}

const tensorflow::DataTypeVector& IgniteDataset::output_dtypes() const {
  static tensorflow::DataTypeVector* dtypes = new tensorflow::DataTypeVector();

  std::cout << "DTypes: ";
  for (auto e: schema_) {
    if (e == 1 || e == 12) {
      dtypes->push_back(tensorflow::DT_INT8);
    }
    else if (e == 2 || e == 13) {
      dtypes->push_back(tensorflow::DT_INT16);
    }
    else if (e == 3 || e == 14) {
      std::cout << "int32 ";
      dtypes->push_back(tensorflow::DT_INT32);
    }
    else if (e == 4 || e == 15) {
      dtypes->push_back(tensorflow::DT_INT64);
    }
    else if (e == 5 || e == 16) {
      dtypes->push_back(tensorflow::DT_FLOAT);
    }
    else if (e == 6 || e == 17) {
      std::cout << "double ";
      dtypes->push_back(tensorflow::DT_DOUBLE);
    }
    else if (e == 7 || e == 18) {
      dtypes->push_back(tensorflow::DT_UINT8);
    }
    else if (e == 8 || e == 19) {
      dtypes->push_back(tensorflow::DT_BOOL);
    }
    else if (e == 9 || e == 20) {
      dtypes->push_back(tensorflow::DT_STRING);
    }
    else {
      // skip.
    }
    std::cout << "\n";
  }

  return *dtypes;
}

const std::vector<tensorflow::PartialTensorShape>& IgniteDataset::output_shapes() const {
  static std::vector<tensorflow::PartialTensorShape>* shapes =new std::vector<tensorflow::PartialTensorShape>();

  for (auto e: schema_) {
    if (e >= 1 && e < 10) {
      shapes->push_back(tensorflow::PartialTensorShape({}));
    }
    else if (e >= 12 && e < 21) {
      shapes->push_back(tensorflow::PartialTensorShape({-1}));
    }
    else {
      // skip.
    }
  }

  return *shapes;
}

tensorflow::string IgniteDataset::DebugString() const { 
  return "KafkaDatasetOp::Dataset"; 
}

tensorflow::Status IgniteDataset::AsGraphDefInternal(DatasetGraphDefBuilder* b, tensorflow::Node** output) const {
  return tensorflow::Status::OK();
}

} // namespace ignite
