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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class CSVDatasetOp : public DatasetOpKernel {
 public:
  explicit CSVDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx),
        op_version_(ctx->def().op() == "CSVDatasetV2" ? 2 : 1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    tstring compression_type;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "compression_type",
                                                     &compression_type));

    OpInputList record_defaults_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("record_defaults", &record_defaults_list));
    for (int i = 0; i < record_defaults_list.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults_list[i].dims() <= 1,
                  errors::InvalidArgument(
                      "Each record default should be at most rank 1"));
      OP_REQUIRES(ctx, record_defaults_list[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults_list[i].NumElements()));
    }

    const Tensor* select_cols_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("select_cols", &select_cols_tensor));
    OP_REQUIRES(ctx, select_cols_tensor->dims() == 1,
                errors::InvalidArgument("`select_cols` must be a vector."));

    std::vector<int64_t> exclude_cols;
    if (op_version_ > 1) {
      const Tensor* exclude_cols_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("exclude_cols", &exclude_cols_tensor));
      OP_REQUIRES(ctx, exclude_cols_tensor->dims() == 1,
                  errors::InvalidArgument("`exclude_cols` must be a vector"));
      exclude_cols.reserve(exclude_cols_tensor->NumElements());
      for (int i = 0; i < exclude_cols_tensor->NumElements(); ++i) {
        exclude_cols.push_back(exclude_cols_tensor->flat<int64_t>()(i));
      }
    }

    int64_t buffer_size = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64_t>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size > 0,
                errors::InvalidArgument("buffer_size should be positive"));

    tstring delim;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<tstring>(ctx, "field_delim", &delim));
    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));

    bool header;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "header", &header));

    bool use_quote_delim;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "use_quote_delim",
                                                  &use_quote_delim));
    tstring na_value;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<tstring>(ctx, "na_value", &na_value));

    std::vector<Tensor> record_defaults;
    record_defaults.reserve(record_defaults_list.size());
    for (const Tensor& t : record_defaults_list) {
      record_defaults.push_back(t);
    }

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<tstring>()(i));
    }

    io::ZlibCompressionOptions zlib_compression_options =
        io::ZlibCompressionOptions::DEFAULT();
    if (compression_type == "ZLIB") {
      zlib_compression_options = io::ZlibCompressionOptions::DEFAULT();
    } else if (compression_type == "GZIP") {
      zlib_compression_options = io::ZlibCompressionOptions::GZIP();
    } else {
      OP_REQUIRES(ctx, compression_type.empty(),
                  errors::InvalidArgument(
                      "Unsupported compression_type: ", compression_type, "."));
    }
    zlib_compression_options.input_buffer_size = buffer_size;

    std::vector<int64_t> select_cols;
    select_cols.reserve(select_cols_tensor->NumElements());
    for (int i = 0; i < select_cols_tensor->NumElements(); ++i) {
      select_cols.push_back(select_cols_tensor->flat<int64_t>()(i));
    }
    OP_REQUIRES(
        ctx, output_types_.size() == select_cols.size() || select_cols.empty(),
        errors::InvalidArgument("select_cols should match output size"));
    for (int i = 1; i < select_cols.size(); i++) {
      OP_REQUIRES(ctx, select_cols[i - 1] < select_cols[i],
                  errors::InvalidArgument(
                      "select_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, select_cols.empty() || select_cols.front() >= 0,
        errors::InvalidArgument("select_cols should be non-negative indices"));

    OP_REQUIRES(ctx, select_cols.empty() || exclude_cols.empty(),
                errors::InvalidArgument(
                    "Either select_cols or exclude_cols should be empty"));
    for (int i = 1; i < exclude_cols.size(); i++) {
      OP_REQUIRES(ctx, exclude_cols[i - 1] < exclude_cols[i],
                  errors::InvalidArgument(
                      "exclude_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, exclude_cols.empty() || exclude_cols.front() >= 0,
        errors::InvalidArgument("exclude_cols should be non-negative indices"));

    *output = new Dataset(ctx, std::move(filenames), header,
                          std::move(compression_type), zlib_compression_options,
                          output_types_, output_shapes_,
                          std::move(record_defaults), std::move(select_cols),
                          std::move(exclude_cols), use_quote_delim, delim[0],
                          std::move(na_value), op_version_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> filenames, bool header,
            string compression_type, io::ZlibCompressionOptions options,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::vector<Tensor> record_defaults,
            std::vector<int64_t> select_cols, std::vector<int64_t> exclude_cols,
            bool use_quote_delim, char delim, string na_value, int op_version)
        : DatasetBase(DatasetContext(ctx)),
          filenames_(std::move(filenames)),
          header_(header),
          out_type_(output_types),
          output_shapes_(output_shapes),
          record_defaults_(std::move(record_defaults)),
          select_cols_(std::move(select_cols)),
          exclude_cols_(std::move(exclude_cols)),
          use_quote_delim_(use_quote_delim),
          delim_(delim),
          na_value_(std::move(na_value)),
          op_version_(op_version),
          use_compression_(!compression_type.empty()),
          compression_type_(std::move(compression_type)),
          options_(options) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::CSV")});
    }

    const DataTypeVector& output_dtypes() const override { return out_type_; }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "CSVDatasetOp::Dataset"; }

    absl::Status CheckExternalState() const override {
      return absl::OkStatus();
    }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->clear();
      return absl::OkStatus();
    }

   protected:
    absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    Node** output) const override {
      Node* filenames = nullptr;
      Node* compression_type = nullptr;
      Node* buffer_size = nullptr;
      Node* header = nullptr;
      Node* delim = nullptr;
      Node* use_quote_delim = nullptr;
      Node* na_value = nullptr;
      Node* select_cols = nullptr;
      Node* exclude_cols = nullptr;

      std::vector<Node*> record_defaults;
      record_defaults.reserve(record_defaults_.size());
      for (const Tensor& t : record_defaults_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        record_defaults.emplace_back(node);
      }

      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
      TF_RETURN_IF_ERROR(
          b->AddScalar(options_.input_buffer_size, &buffer_size));
      TF_RETURN_IF_ERROR(b->AddScalar(header_, &header));

      tstring delim_string(1, delim_);
      TF_RETURN_IF_ERROR(b->AddScalar(delim_string, &delim));
      TF_RETURN_IF_ERROR(b->AddScalar(use_quote_delim_, &use_quote_delim));
      TF_RETURN_IF_ERROR(b->AddScalar(na_value_, &na_value));
      TF_RETURN_IF_ERROR(b->AddVector(select_cols_, &select_cols));
      TF_RETURN_IF_ERROR(b->AddVector(exclude_cols_, &exclude_cols));

      if (op_version_ > 1) {
        TF_RETURN_IF_ERROR(b->AddDataset(
            this,
            {std::make_pair(0, filenames), std::make_pair(1, compression_type),
             std::make_pair(2, buffer_size), std::make_pair(3, header),
             std::make_pair(4, delim), std::make_pair(5, use_quote_delim),
             std::make_pair(6, na_value), std::make_pair(7, select_cols),
             std::make_pair(9, exclude_cols)},     // Single tensor inputs
            {std::make_pair(8, record_defaults)},  // Tensor list inputs
            {}, output));
      } else {
        TF_RETURN_IF_ERROR(b->AddDataset(
            this,
            {
                std::make_pair(0, filenames),
                std::make_pair(1, compression_type),
                std::make_pair(2, buffer_size),
                std::make_pair(3, header),
                std::make_pair(4, delim),
                std::make_pair(5, use_quote_delim),
                std::make_pair(6, na_value),
                std::make_pair(7, select_cols),
            },                                     // Single tensor inputs
            {std::make_pair(8, record_defaults)},  // Tensor list inputs
            {}, output));
      }
      return absl::OkStatus();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      absl::Status GetNextInternal(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
        mutex_lock l(mu_);
        bool select_all =
            dataset()->select_cols_.empty() && dataset()->exclude_cols_.empty();
        do {
          // We are currently processing a file, so try to read the next record
          if (input_stream_) {
            absl::Status s =
                ReadRecord(ctx, out_tensors, select_all,
                           dataset()->select_cols_, dataset()->exclude_cols_);
            if (s.ok()) {
              // Validate output
              if (out_tensors->size() != dataset()->out_type_.size()) {
                return errors::InvalidArgument(
                    "Expect ", dataset()->out_type_.size(), " fields but have ",
                    out_tensors->size(), " in record");
              }

              *end_of_sequence = false;
              return s;
            }
            if (!absl::IsOutOfRange(s)) {
              // Not at the end of file, return OK or non-EOF errors to caller.
              *end_of_sequence = false;
              return s;
            }
            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }
          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return absl::OkStatus();
          }
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      absl::Status SaveInternal(SerializationContext* ctx,
                                IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_file_index"),
                                               current_file_index_));
        // `input_stream_` is empty if
        // 1. GetNext has not been called even once.
        // 2. All files have been read and the iterator has been exhausted.
        if (input_stream_ && num_buffer_reads_ > 0) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("pos"), pos_));
          // If num_buffer_reads_ == 0, the buffer hasn't been filled even once.
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_buffer_reads"),
                                                 num_buffer_reads_));
        }
        return absl::OkStatus();
      }

      absl::Status RestoreInternal(IteratorContext* ctx,
                                   IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        int64_t current_file_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_file_index"),
                                              &current_file_index));
        current_file_index_ = size_t(current_file_index);
        // The keys "pos" and "num_buffer_reads" are written only if
        // the iterator was saved with an open, partially read file.
        if (reader->Contains(full_name("pos"))) {
          int64_t pos, num_buffer_reads;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("pos"), &pos));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_buffer_reads"),
                                                &num_buffer_reads));

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));

          num_buffer_reads_ = size_t(num_buffer_reads - 1);

          // Restores the most recently held buffer
          absl::Status s = input_stream_->SkipNBytes(
              num_buffer_reads_ * dataset()->options_.input_buffer_size);
          if (!s.ok() && !absl::IsOutOfRange(s)) {
            // We might get out of range error here if the size of the file
            // is not an exact multiple of the buffer size, and the last buffer
            // read is < buffer_size. This is valid and we do not surface the
            // error.
            return s;
          }

          absl::Status s2 = FillBuffer(&buffer_);
          if (!s2.ok() && !absl::IsOutOfRange(s2)) {
            return s2;
          }
          pos_ = size_t(pos);
        }
        return absl::OkStatus();
      }

     private:
      // Reads an entire CSV row from the input stream, either from the
      // existing buffer or by filling the buffer as needed. Converts extracted
      // fields to output tensors as we go.
      //
      // When this function is called, pos_ should be the index of the first
      // character of the record in buffer_, or past the end of the buffer.
      // Note: ctx and out_tensors are only used in this function
      // when fields are included in the record.
      absl::Status ReadRecord(IteratorContext* ctx,
                              std::vector<Tensor>* out_tensors, bool select_all,
                              const std::vector<int64_t>& selected,
                              const std::vector<int64_t>& excluded)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          // At the end of the file, this will return errors::OutOfRange
          TF_RETURN_IF_ERROR(FillBuffer(&buffer_));
          pos_ = 0;
        }

        // The first character may be \n if this is the continuation of a
        // \r\n linebreak between this and the previous record. If so, skip it.

        bool end_of_record = false;  // Keep track of when we find \n, \r or EOF
        size_t num_parsed = 0;
        size_t num_selected_parsed = 0;
        size_t num_excluded_parsed = 0;

        absl::Status result;

        while (!end_of_record) {  // Read till we reach \n, \r or EOF
          bool explicit_exclude = num_excluded_parsed < excluded.size() &&
                                  excluded[num_excluded_parsed] == num_parsed;
          bool include = select_all ||
                         (num_selected_parsed < selected.size() &&
                          selected[num_selected_parsed] == num_parsed) ||
                         (!excluded.empty() && !explicit_exclude);

          // Don't fail fast, so that the next call to GetNext may still return
          // a valid record
          result.Update(
              ParseOneField(ctx, out_tensors, &end_of_record, include));

          num_parsed++;
          if (include) num_selected_parsed++;
          if (explicit_exclude) num_excluded_parsed++;
        }

        return result;
      }

      // Parses one field from position pos_ in the buffer. Fields are
      // delimited by delim, CRLF, or EOF. Advances pos_ to the first char of
      // the next field.
      absl::Status ParseOneField(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_record, bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          // If we get here, this means the previous field's end coincided
          // with the end of the buffer. We can fill the buffer without abandon.
          absl::Status s = FillBuffer(&buffer_);

          if (absl::IsOutOfRange(s)) {
            // Reached EOF, and last field is empty
            *end_of_record = true;
            if (include) {
              return FieldToOutput(ctx, absl::string_view(), out_tensors);
            } else {
              return absl::OkStatus();
            }
          } else if (!s.ok()) {
            return s;  // Surface other errors back to caller
          }

          pos_ = 0;
        }

        if (dataset()->use_quote_delim_ && buffer_[pos_] == '"') {
          return ParseQuotedField(ctx, out_tensors, end_of_record, include);
        }

        return ParseUnquotedField(ctx, out_tensors, end_of_record, include);
      }

      // For keeping track of relevant parts of a field from a previous buffer
      struct Piece {
        size_t start;
        size_t len;
        string buffer;

        Piece(string buffer, size_t start, size_t len)
            : start(start), len(len), buffer(std::move(buffer)) {}
      };

      // Given that pos_ exceeds the buffer, saves the relevant part of the
      // current buffer (if necessary), fills the buffer, and resets indices to
      // 0.
      absl::Status SaveAndFillBuffer(std::vector<Piece>* earlier_pieces,
                                     size_t* start, bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        tstring temp_buffer;

        buffer_.swap(temp_buffer);
        if (include && pos_ > *start) {
          earlier_pieces->push_back(
              Piece(std::move(temp_buffer), *start, pos_ - *start));
        }
        pos_ = 0;
        *start = 0;
        return FillBuffer(&buffer_);
      }

      // Parses unquoted field from position pos_ in the buffer. Continually
      // reads from buffer until end of field is reached (delim, CRLF, or EOF).
      // Advances pos_ to keep track of our position in the buffer as we go,
      // stopping at the first character of the next field.
      absl::Status ParseQuotedField(IteratorContext* ctx,
                                    std::vector<Tensor>* out_tensors,
                                    bool* end_of_record, bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<Piece> earlier_pieces;
        size_t start = pos_;
        pos_++;  // Starting quotation mark

        absl::Status parse_result;
        while (true) {  // Each iter reads 1 char, filling buffer if necessary
          if (pos_ >= buffer_.size()) {
            absl::Status s =
                SaveAndFillBuffer(&earlier_pieces, &start, include);
            if (absl::IsOutOfRange(s)) {
              return errors::InvalidArgument(
                  "Reached end of file without closing quoted field in "
                  "record");
            } else if (!s.ok()) {
              return s;  // Surface all other errors to caller
            }
          }

          char ch = buffer_[pos_];
          if (ch == '"') {
            // When we encounter a quote, we look ahead to the next character to
            // decide what to do
            pos_++;
            if (pos_ >= buffer_.size()) {
              absl::Status s =
                  SaveAndFillBuffer(&earlier_pieces, &start, include);
              if (absl::IsOutOfRange(s)) {
                // This was the last field. We are done
                *end_of_record = true;
                parse_result.Update(
                    QuotedFieldToOutput(ctx, absl::string_view(), out_tensors,
                                        earlier_pieces, include));
                return parse_result;
              } else if (!s.ok()) {
                return s;
              }
            }

            char next = buffer_[pos_];
            pos_++;
            if (next == dataset()->delim_) {
              parse_result.Update(QuotedFieldToOutput(
                  ctx, absl::string_view(&buffer_[start], pos_ - 1 - start),
                  out_tensors, earlier_pieces, include));
              return parse_result;

            } else if (next == '\n' || next == '\r') {
              *end_of_record = true;
              parse_result.Update(QuotedFieldToOutput(
                  ctx, absl::string_view(&buffer_[start], pos_ - 1 - start),
                  out_tensors, earlier_pieces, include));
              if (next == '\r') SkipNewLineIfNecessary();
              return parse_result;
            } else if (next != '"') {
              // Take note of the error, but keep going to end of field.
              include = false;  // So we don't get funky errors when trying to
                                // unescape the quotes.
              parse_result.Update(errors::InvalidArgument(
                  "Quote inside a string has to be escaped by another quote"));
            }

          } else {
            pos_++;
          }
        }
      }

      // Converts quoted field to an output tensor, removing the starting
      // and ending quotes from it and unescaping double quotations if
      // necessary.
      absl::Status QuotedFieldToOutput(IteratorContext* ctx,
                                       absl::string_view field,
                                       std::vector<Tensor>* out_tensors,
                                       const std::vector<Piece>& earlier_pieces,
                                       bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!include) return absl::OkStatus();

        if (earlier_pieces.empty()) {
          if (field.find('\"', 1) == field.size() - 1) {
            // `field` contains no escaped quotation marks.
            // Exclude framing quotation marks
            field.remove_prefix(1);
            field.remove_suffix(1);
            return FieldToOutput(ctx, field, out_tensors);
          }
        }
        string field_complete;
        size_t str_len = field.size();
        for (const Piece& p : earlier_pieces) {
          str_len += p.len;
        }
        field_complete.reserve(str_len);

        // This bool flips every time we see a quote, so that we skip the second
        // quote of every pair of adjacent quotes in the field. We need to track
        // this across iterations of the for loop because adjacent double quotes
        // may be in different buffers. Initialize to true because we also skip
        // the opening quotation mark of the quoted field.
        bool skip_next_quote = true;
        for (const Piece& p : earlier_pieces) {
          AppendUnescapedPiece(absl::string_view(&p.buffer[p.start], p.len),
                               &field_complete, &skip_next_quote);
        }
        AppendUnescapedPiece(field, &field_complete, &skip_next_quote);
        absl::string_view result = absl::string_view(field_complete);
        result.remove_suffix(1);  // Skip final quote

        return FieldToOutput(ctx, result, out_tensors);
      }

      void AppendUnescapedPiece(absl::string_view piece, string* field_complete,
                                bool* skip_next_quote) {
        size_t from = 0;
        size_t found = piece.find('\"', from);
        while (found != string::npos) {
          if (!*skip_next_quote) {
            // This is the first quote in a pair of adjacent double quotes
            field_complete->append(piece.data() + from, found + 1 - from);
          }
          *skip_next_quote = !*skip_next_quote;
          from = found + 1;
          found = piece.find('\"', from);
        }
        // Include the chunk after the last quotation mark in the string
        if (from < piece.size()) {
          field_complete->append(piece.data() + from, piece.size() - from);
        }
      }

      // Parses unquoted field from position pos_ in the buffer. Continually
      // reads from buffer until end of field is reached (delim, CRLF, or EOF).
      // Advances pos_ to keep track of our position in the buffer as we go,
      // stopping at the first character of the next field.
      absl::Status ParseUnquotedField(IteratorContext* ctx,
                                      std::vector<Tensor>* out_tensors,
                                      bool* end_of_record, bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<Piece> earlier_pieces;
        size_t start = pos_;
        absl::Status parse_result;

        while (true) {  // Each iter reads 1 char, filling buffer if necessary
          if (pos_ >= buffer_.size()) {
            absl::Status s =
                SaveAndFillBuffer(&earlier_pieces, &start, include);
            // Handle errors
            if (absl::IsOutOfRange(s)) {
              // Whatever we have is the last field of the last record
              *end_of_record = true;
              parse_result.Update(UnquotedFieldToOutput(
                  ctx, absl::string_view(&buffer_[start], pos_ - start),
                  out_tensors, earlier_pieces, include));
              return parse_result;
            } else if (!s.ok()) {
              return s;  // Surface all other errors to caller
            }
          }

          char ch = buffer_[pos_];

          if (ch == dataset()->delim_) {
            parse_result.Update(UnquotedFieldToOutput(
                ctx, absl::string_view(&buffer_[start], pos_ - start),
                out_tensors, earlier_pieces, include));
            pos_++;
            return parse_result;
          }
          if (ch == '\n' || ch == '\r') {
            // need special case to skip over first \n of record if the line
            // breaks are \r\n
            parse_result.Update(UnquotedFieldToOutput(
                ctx, absl::string_view(&buffer_[start], pos_ - start),
                out_tensors, earlier_pieces, include));
            *end_of_record = true;
            pos_++;
            if (ch == '\r') SkipNewLineIfNecessary();
            return parse_result;
          }
          if (dataset()->use_quote_delim_ && ch == '"') {
            // Take note of the error, but keep going to end of field.
            parse_result.Update(errors::InvalidArgument(
                "Unquoted fields cannot have quotes inside"));
          }
          // Otherwise, go to next character
          pos_++;
        }
      }

      absl::Status FillBuffer(tstring* result)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        result->clear();
        ++num_buffer_reads_;
        absl::Status s = input_stream_->ReadNBytes(
            dataset()->options_.input_buffer_size, result);

        if (absl::IsOutOfRange(s) && !result->empty()) {
          // Ignore OutOfRange error when ReadNBytes read < N bytes.
          return absl::OkStatus();
        }
        return s;
      }

      // Given a field, converts it to the right output tensor type
      absl::Status FieldToOutput(IteratorContext* ctx, absl::string_view field,
                                 std::vector<Tensor>* out_tensors) {
        size_t output_idx = out_tensors->size();
        if (output_idx >= dataset()->out_type_.size()) {
          // We can get here if we're selecting all columns, but the number of
          // fields exceeds the number of defaults provided
          return errors::InvalidArgument("Expect ", dataset()->out_type_.size(),
                                         " fields but have more in record");
        }
        const DataType& dtype = dataset()->out_type_[output_idx];
        out_tensors->emplace_back(ctx->allocator({}), dtype, TensorShape({}));
        Tensor& component = out_tensors->back();
        if ((field.empty() || field == dataset()->na_value_) &&
            dataset()->record_defaults_[output_idx].NumElements() != 1) {
          // If the field is empty or NA value, and default is not given,
          // report error.
          return errors::InvalidArgument("Field ", output_idx,
                                         " is required but missing in record!");
        }

        switch (dtype) {
          // For each case, if the field is empty, we use the default.
          // Otherwise, we convert it to the right type.
          case DT_INT32: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int32>()() =
                  dataset()->record_defaults_[output_idx].flat<int32>()(0);
            } else {
              int32_t value;
              if (!absl::SimpleAtoi(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int32: ", field);
              }
              component.scalar<int32>()() = value;
            }
            break;
          }
          case DT_INT64: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int64_t>()() =
                  dataset()->record_defaults_[output_idx].flat<int64_t>()(0);
            } else {
              int64_t value;
              if (!absl::SimpleAtoi(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int64: ", field);
              }
              component.scalar<int64_t>()() = value;
            }
            break;
          }
          case DT_FLOAT: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<float>()() =
                  dataset()->record_defaults_[output_idx].flat<float>()(0);
            } else {
              float value;
              if (!absl::SimpleAtof(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid float: ", field);
              }
              component.scalar<float>()() = value;
            }
            break;
          }
          case DT_DOUBLE: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<double>()() =
                  dataset()->record_defaults_[output_idx].flat<double>()(0);
            } else {
              double value;
              if (!absl::SimpleAtod(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid double: ", field);
              }
              component.scalar<double>()() = value;
            }
            break;
          }
          case DT_STRING: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<tstring>()() =
                  dataset()->record_defaults_[output_idx].flat<tstring>()(0);
            } else {
              component.scalar<tstring>()() = string(field);
            }
            break;
          }
          default:
            return errors::InvalidArgument("csv: data type ", dtype,
                                           " not supported in field ",
                                           output_idx);
        }
        return absl::OkStatus();
      }

      // Records can be delimited by "\r\n" line breaks. When we encounter a
      // '\r', we have to check the next character to see if it is part of the
      // linebreak, and ignore it if so.
      void SkipNewLineIfNecessary() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          absl::Status s = FillBuffer(&buffer_);
          pos_ = 0;
          // If we failed to fill buffer, it doesn't matter because we're done
          // with the record
          if (!s.ok()) return;
        }
        if (buffer_[pos_] == '\n') {
          pos_++;
        }
      }

      // Given a string field, and its index in the output,
      // converts it to a Tensor of the right type and adds it to the
      // out_tensors vector.
      absl::Status UnquotedFieldToOutput(
          IteratorContext* ctx, absl::string_view field,
          std::vector<Tensor>* out_tensors,
          const std::vector<Piece>& earlier_pieces, bool include)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!include) return absl::OkStatus();

        if (earlier_pieces.empty()) {
          return FieldToOutput(ctx, field, out_tensors);
        }

        size_t str_len = field.size();
        for (const Piece& p : earlier_pieces) {
          str_len += p.len;
        }
        string field_complete;
        field_complete.reserve(str_len);

        for (const Piece& p : earlier_pieces) {
          field_complete.append(p.buffer, p.start, p.len);
        }

        field_complete.append(field.data(), field.size());
        return FieldToOutput(ctx, field_complete, out_tensors);
      }

      // Sets up reader streams to read from the file at `current_file_index_`.
      absl::Status SetupStreamsLocked(Env* env)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(
            dataset()->filenames_[current_file_index_], &file_));
        random_access_input_stream_ =
            std::make_shared<io::RandomAccessInputStream>(file_.get(), false);

        if (dataset()->use_compression_) {
          input_stream_ = std::make_shared<io::ZlibInputStream>(
              random_access_input_stream_.get(),
              dataset()->options_.input_buffer_size,
              dataset()->options_.input_buffer_size, dataset()->options_);
        } else {
          input_stream_ = random_access_input_stream_;
        }
        buffer_.clear();
        pos_ = 0;
        num_buffer_reads_ = 0;
        if (dataset()->header_) {
          // Read one line, but don't include it. Pass nullptrs as dummy
          // pointers to objects that shouldn't be invoked anyway
          // We need to process this as a record here instead of just finding
          // the first newline because it might contain quoted fields with
          // newlines in the header as well
          std::vector<int64_t> empty;
          absl::Status s = ReadRecord(nullptr, nullptr, false, empty, empty);
          if (!s.ok()) {
            return errors::InvalidArgument("Can't read header of file");
          }
        }
        return absl::OkStatus();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        input_stream_.reset();
        file_.reset();
      }

      mutex mu_;
      tstring buffer_ TF_GUARDED_BY(mu_);  // Maintain our own buffer
      size_t pos_ TF_GUARDED_BY(
          mu_);  // Index into the buffer must be maintained between iters
      size_t num_buffer_reads_ TF_GUARDED_BY(mu_);
      std::shared_ptr<io::RandomAccessInputStream> random_access_input_stream_
          TF_GUARDED_BY(mu_);
      std::shared_ptr<io::InputStreamInterface> input_stream_
          TF_GUARDED_BY(mu_);
      size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          TF_GUARDED_BY(mu_);  // must outlive input_stream_
    };                         // class Iterator

    const std::vector<string> filenames_;
    const bool header_;
    const DataTypeVector out_type_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::vector<Tensor> record_defaults_;
    const std::vector<int64_t> select_cols_;
    const std::vector<int64_t> exclude_cols_;
    const bool use_quote_delim_;
    const char delim_;
    const tstring na_value_;
    const int op_version_;
    const bool use_compression_;
    const tstring compression_type_;
    const io::ZlibCompressionOptions options_;
  };  // class Dataset

  const int op_version_;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};  // class CSVDatasetOp

REGISTER_KERNEL_BUILDER(Name("CSVDataset").Device(DEVICE_CPU), CSVDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalCSVDataset").Device(DEVICE_CPU),
                        CSVDatasetOp);
REGISTER_KERNEL_BUILDER(Name("CSVDatasetV2").Device(DEVICE_CPU), CSVDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
