/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/ignite/kernels/ignite_holder.h"

using namespace ignite;
using namespace cache;
using namespace query;
using namespace binary;

namespace tensorflow {

    class IgniteDatasetOp : public DatasetOpKernel {
    public:
        using DatasetOpKernel::DatasetOpKernel;

        void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
            std::string cache = "";
            OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "cache", &cache));
            *output = new Dataset(ctx, cache);
        }

    private:
        class Dataset : public GraphDatasetBase {
        private:
            Ignite grid;
            const std::string cacheName;
        public:
            Dataset(OpKernelContext *ctx, const string &cache)
                    : GraphDatasetBase(ctx), cacheName(cache) {
                grid = IgniteHolder::Instance().getIgnite();
            }

            std::unique_ptr<IteratorBase> MakeIteratorInternal(
                    const string &prefix) const override {
                return std::unique_ptr<IteratorBase>(
                        new Iterator({this, strings::StrCat(prefix, "::Ignite")}));
            }

            const DataTypeVector &output_dtypes() const override {
                static DataTypeVector *dtypes = new DataTypeVector({DT_STRING});
                return *dtypes;
            }

            const std::vector<PartialTensorShape> &output_shapes() const override {
                static std::vector<PartialTensorShape> *shapes =
                        new std::vector<PartialTensorShape>({{}});
                return *shapes;
            }

            string DebugString() const override { return "IgniteDatasetOp::Dataset"; }

        protected:
            Status AsGraphDefInternal(DatasetGraphDefBuilder *b,
                                      Node **output) const override {
                return errors::Unimplemented();
            }

        private:
            class Iterator : public DatasetIterator<Dataset> {
                typedef std::vector<CacheEntry<int, std::string> > ResVector;
            private:
                ResVector::const_iterator iter;
                ResVector::const_iterator iter_end;
                ResVector res;
            public:
                explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params) {
                    ScanQuery scan;

                    const char *cacheName = params.dataset->cacheName.c_str();

                    Ignite grid = params.dataset->grid;
                    Cache<int, std::string> cac = grid.GetCache<int, std::string>(cacheName);

                    cac.Query(scan).GetAll(res);

                    iter = res.begin();
                    iter_end = res.end();
                }

                Status GetNextInternal(IteratorContext *ctx,
                                       std::vector<Tensor> *out_tensors,
                                       bool *end_of_sequence) override {
                    Tensor line_tensor(cpu_allocator(), DT_STRING, {});

                    if (iter != iter_end) {
                        std::string val = iter->GetValue();
                        line_tensor.scalar<string>()() = std::move(val);
                        out_tensors->emplace_back(std::move(line_tensor));
                        iter++;
                        *end_of_sequence = false;
                    } else {
                        *end_of_sequence = true;
                    }

                    return Status::OK();
                }

            protected:
                Status SaveInternal(IteratorStateWriter *writer) override {
                    return error::Unimplemented();
                }

                Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
                    return error::Unimplemented();
                }
            };
        };
    };

    REGISTER_KERNEL_BUILDER(Name("IgniteDataset").Device(DEVICE_CPU),
                            IgniteDatasetOp);

}  // namespace tensorflow
