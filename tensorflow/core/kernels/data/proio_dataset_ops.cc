#include <math.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/data/dataset.h"

#include "external/proio_archive/cpp-proio/src/reader.h"

namespace tensorflow {
namespace {

const double sqrt3 = sqrt(3.);

class ProIODatasetOp : public DatasetOpKernel {
   public:
    using DatasetOpKernel::DatasetOpKernel;

    void MakeDataset(OpKernelContext *ctx, DatasetBase **output) override {
        std::string filename;
        OP_REQUIRES_OK(ctx, ParseScalarArgument<std::string>(ctx, "filename", &filename));

        *output = new Dataset(ctx, filename);
    }

   private:
    class Dataset : public GraphDatasetBase {
       public:
        Dataset(OpKernelContext *ctx, std::string filename)
            : GraphDatasetBase(ctx), filename_(filename) {}

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::ProIO")}, filename_));
        }

        const DataTypeVector &output_dtypes() const override {
            static DataTypeVector *dtypes = new DataTypeVector({DT_STRING});
            return *dtypes;
        }
        const std::vector<PartialTensorShape> &output_shapes() const override {
            static std::vector<PartialTensorShape> *shapes = new std::vector<PartialTensorShape>({{}});
            return *shapes;
        }

        string DebugString() const override { return "ProIODatasetOp::Dataset"; }

       protected:
        Status AsGraphDefInternal(DatasetGraphDefBuilder *b, Node **output) const override {
            Node *filename = nullptr;
            TF_RETURN_IF_ERROR(b->AddScalar(filename_, &filename));
            TF_RETURN_IF_ERROR(b->AddDataset(this, {filename}, output));
            return Status::OK();
        }

       private:
        class Iterator : public DatasetIterator<Dataset> {
           public:
            explicit Iterator(const Params &params, std::string filename)
                : DatasetIterator<Dataset>(params), reader_(new proio::Reader(filename)) {}

            Status GetNextInternal(IteratorContext *ctx, std::vector<Tensor> *out_tensors,
                                   bool *end_of_sequence) override {
                Tensor data_tensor(ctx->allocator({}), DT_STRING, {});

                std::unique_lock<std::mutex> lock(*reader_);
                if (!reader_->Next((std::string *)&data_tensor.scalar<string>()())) {
                    *end_of_sequence = true;
                    return Status::OK();
                }
                lock.unlock();

                out_tensors->emplace_back(std::move(data_tensor));
                *end_of_sequence = false;
                return Status::OK();
            }

           private:
            std::shared_ptr<proio::Reader> reader_;
        };

       private:
        std::string filename_;
    };
};

REGISTER_KERNEL_BUILDER(Name("ProIODataset").Device(DEVICE_CPU), ProIODatasetOp);

}  // namespace
}  // namespace tensorflow
