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

#include "parquet/api/reader.h"

namespace tensorflow {

class ParquetDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;
  explicit ParquetDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx, dt == DT_INT32 || dt == DT_INT64 || dt == DT_FLOAT ||
                           dt == DT_DOUBLE || dt == DT_BOOL,
                  errors::InvalidArgument(
                      "Each element of `output_types_` must be one of: "
                      "DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL "));
    }
    for (const PartialTensorShape& pts : output_shapes_) {
      OP_REQUIRES(ctx, pts.dims() == 0,
                  errors::InvalidArgument(
                      "Each element of `output_shapes_` must be a scalar."));
    }
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    const Tensor* columns_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_tensor));
    OP_REQUIRES(
        ctx, columns_tensor->dims() <= 1,
        errors::InvalidArgument("`columns` must be a scalar or a vector."));

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    std::vector<int64> columns;
    columns.reserve(columns_tensor->NumElements());
    for (int i = 0; i < columns_tensor->NumElements(); ++i) {
      columns.push_back(columns_tensor->flat<int32>()(i));
    }

    *output =
        new Dataset(ctx, filenames, columns, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames,
            const std::vector<int64>& columns,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(filenames),
          columns_(columns),
          output_types_(output_types),
          output_shapes_(output_shapes) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Parquet")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "ParquetDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      Node* columns = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(columns_, &columns));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames, columns}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // Loop until we find a row to read or there are no more files left to
        // read
        while (true) {
          if (parquet_reader_) {
            // We are currently processing a file, so try to read the next row
            // group.
            while (current_row_group_ < file_metadata_->num_row_groups()) {
              if (current_row_ < row_group_reader_->metadata()->num_rows()) {
                // Read columns to outputs.
                for (int64 i = 0; i < dataset()->columns_.size(); i++) {
                  DataType dt = dataset()->output_types_[i];
                  int64 column = dataset()->columns_[i];
                  std::shared_ptr<parquet::ColumnReader> column_reader =
                      column_readers_[i];
                  Tensor tensor(ctx->allocator({}), dt, {});
                  TF_RETURN_IF_ERROR(GetTensorValue(
                      current_row_, dt, column_reader.get(), &tensor));
                  out_tensors->emplace_back(std::move(tensor));
                }
                ++current_row_;
                *end_of_sequence = false;
                return Status::OK();
              }
              // We have reached the end of the current row group, so maybe
              // move on to next row group.
              current_row_ = 0;
              row_group_reader_.reset();
              ++current_row_group_;
              if (current_row_group_ < file_metadata_->num_row_groups()) {
                row_group_reader_ =
                    parquet_reader_->RowGroup(current_row_group_);
                column_readers_.clear();
                for (int64 i = 0; i < dataset()->columns_.size(); i++) {
                  int64 column = dataset()->columns_[i];
                  std::shared_ptr<parquet::ColumnReader> column_reader =
                      row_group_reader_->Column(column);
                  column_readers_.emplace_back(column_reader);
                }
              }
            }
            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }

          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
      }

      template <typename DType>
      Status FillTensorValue(parquet::ColumnReader* column_reader,
                             typename DType::c_type* value) {
        parquet::TypedColumnReader<DType>* reader =
            static_cast<parquet::TypedColumnReader<DType>*>(column_reader);
        // Read one value at a time. The number of rows read is returned.
        // values_read contains the number of non-null rows
        int64_t values_read = 0;
        int64_t rows_read =
            reader->ReadBatch(1, nullptr, nullptr, value, &values_read);
        // Ensure only one value is read and there are no NULL values in the
        // rows read
        if (rows_read != 1) {
          return errors::Internal("rows_read (", rows_read,
                                  ") != 1 or values_read (", values_read,
                                  ") != 1");
        }
        return Status::OK();
      }

      Status GetTensorValue(int64 row, const DataType& data_type,
                            parquet::ColumnReader* column_reader,
                            Tensor* tensor) {
        switch (data_type) {
          case DT_INT32: {
            parquet::TypedColumnReader<parquet::Int32Type>* reader =
                static_cast<parquet::TypedColumnReader<parquet::Int32Type>*>(
                    column_reader);
            int32_t value;
            TF_RETURN_IF_ERROR(
                FillTensorValue<parquet::Int32Type>(reader, &value));
            tensor->scalar<int32>()() = value;
          } break;
          case DT_INT64: {
            parquet::TypedColumnReader<parquet::Int64Type>* reader =
                static_cast<parquet::TypedColumnReader<parquet::Int64Type>*>(
                    column_reader);
            int64_t value;
            TF_RETURN_IF_ERROR(
                FillTensorValue<parquet::Int64Type>(reader, &value));
            tensor->scalar<int64>()() = value;
          } break;
          case DT_FLOAT: {
            parquet::TypedColumnReader<parquet::FloatType>* reader =
                static_cast<parquet::TypedColumnReader<parquet::FloatType>*>(
                    column_reader);
            float value;
            TF_RETURN_IF_ERROR(
                FillTensorValue<parquet::FloatType>(reader, &value));
            tensor->scalar<float>()() = value;
          } break;
          case DT_DOUBLE: {
            parquet::TypedColumnReader<parquet::DoubleType>* reader =
                static_cast<parquet::TypedColumnReader<parquet::DoubleType>*>(
                    column_reader);
            double value;
            TF_RETURN_IF_ERROR(
                FillTensorValue<parquet::DoubleType>(reader, &value));
            tensor->scalar<double>()() = value;
          } break;
          case DT_BOOL: {
            parquet::TypedColumnReader<parquet::BooleanType>* reader =
                static_cast<parquet::TypedColumnReader<parquet::BooleanType>*>(
                    column_reader);
            bool value;
            TF_RETURN_IF_ERROR(
                FillTensorValue<parquet::BooleanType>(reader, &value));
            tensor->scalar<bool>()() = value;
          } break;
          default:
            return errors::Unimplemented(
                DataTypeString(data_type),
                " is currently not supported in ParquetDataset");
        }
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented("SaveInternal is currently not supported");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      // Sets up Parquet streams to read from the topic at
      // `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        DCHECK_LT(current_file_index_, dataset()->filenames_.size());

        // Actually move on to next file.
        const string& next_filename =
            dataset()->filenames_[current_file_index_];
        // TODO (yongtang): Switch to the Open() interface and use
        // RandomAccessFile.
        parquet_reader_ =
            parquet::ParquetFileReader::OpenFile(next_filename, false);
        file_metadata_ = parquet_reader_->metadata();
        current_row_group_ = 0;
        if (current_row_group_ < file_metadata_->num_row_groups()) {
          row_group_reader_ = parquet_reader_->RowGroup(current_row_group_);
          column_readers_.clear();
          for (int64 i = 0; i < dataset()->columns_.size(); i++) {
            int64 column = dataset()->columns_[i];
            std::shared_ptr<parquet::ColumnReader> column_reader =
                row_group_reader_->Column(column);
            column_readers_.emplace_back(column_reader);
          }
        }
        current_row_ = 0;
        return Status::OK();
      }

      // Resets all Parquet streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        current_row_ = 0;
        column_readers_.clear();
        row_group_reader_.reset();
        current_row_group_ = 0;
        file_metadata_.reset();
        parquet_reader_.reset();
      }

      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<parquet::ParquetFileReader> parquet_reader_
          GUARDED_BY(mu_);
      std::shared_ptr<parquet::FileMetaData> file_metadata_ GUARDED_BY(mu_);
      int64 current_row_group_ GUARDED_BY(mu_) = 0;
      std::shared_ptr<parquet::RowGroupReader> row_group_reader_
          GUARDED_BY(mu_);
      std::vector<std::shared_ptr<parquet::ColumnReader>> column_readers_
          GUARDED_BY(mu_);
      int64 current_row_ GUARDED_BY(mu_) = 0;
    };

    const std::vector<string> filenames_;
    const std::vector<int64> columns_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ParquetDataset").Device(DEVICE_CPU),
                        ParquetDatasetOp);

}  // namespace tensorflow
