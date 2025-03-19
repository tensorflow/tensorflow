/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/nanort/ifrt_client.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/client_impl_util.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/mem.h"

namespace xla::cpu {
namespace {

static const char kMemoryKind[] = "";

// Returns a Future that is immediately ready with the given status. This is
// mostly useful because everything NanoRT does is immediately ready.
ifrt::Future<> Ready(absl::Status status = absl::OkStatus()) {
  return ifrt::Future<>(std::move(status));
}

// Base class for all value types. This class doesn't participate in the llvm
// RTTI hierarchy (you can't dynamically cast to it), rather it just
// implements some virtual methods that have the same implementation for all
// NanoRT value types.
template <typename Self, typename Base>
class NanoValue : public llvm::RTTIExtends<Self, Base> {
 public:
  explicit NanoValue(NanoIfrtClient* client) : client_(client) {}

  ifrt::Client* client() const override { return client_; }

  // Called by subclasses to get access to client() without having to cast.
  NanoIfrtClient* nano_client() const { return client_; }

  // All nano values are immediately ready.
  ifrt::Future<> GetReadyFuture() const override { return Ready(); }

  // Subclasses must still implement Delete().
  ifrt::Future<> Delete() override = 0;
  bool IsDeleted() const override = 0;

  // Helper that returns an error if this value is accessed after it has been
  // deleted. Meant to be called with TF_RETURN_IF_ERROR at the top of
  // relevant methods.
  absl::Status ValidateNotDeleted() const {
    if (IsDeleted()) {
      return absl::FailedPreconditionError("Tried to access a deleted value.");
    }
    return absl::OkStatus();
  }

 private:
  NanoIfrtClient* client_;
};

// Array implementation.
//
// This class always holds a continuous buffer of memory, if a sharding is
// provided, it will be disassembled as needed to satisfy caller expectations.
//
// See ShardedNanoArray for the case where the array is constructed from
// multiple existing shards.
class NanoArray final : public NanoValue<NanoArray, ifrt::Array> {
 public:
  // A pointer to the underlying buffer. We use a shared_ptr because for some
  // operations (like disassembly) we can just alias the memory, but we still
  // need to support deletion of the NanoArray that created the buffer.
  using DataPtr = std::shared_ptr<void>;

  NanoArray(NanoIfrtClient* client, ifrt::DType dtype, ifrt::Shape shape,
            DataPtr data, std::shared_ptr<const ifrt::Sharding> sharding)
      : NanoValue<NanoArray, ifrt::Array>(client),
        dtype_(std::move(dtype)),
        shape_(std::move(shape)),
        data_(std::move(data)),
        sharding_(std::move(sharding)) {}

  // Allocates a new array of the given type and shape.
  static absl::StatusOr<tsl::RCReference<NanoArray>> Allocate(
      NanoIfrtClient* client, ifrt::DType dtype, ifrt::Shape shape,
      std::shared_ptr<const ifrt::Sharding> sharding) {
    TF_RET_CHECK(dtype.byte_size().has_value());
    TF_ASSIGN_OR_RETURN(
        DataPtr data_ptr,
        AllocateData(dtype.byte_size().value() * shape.num_elements()));
    return tsl::TakeRef(new NanoArray(client, dtype, std::move(shape),
                                      std::move(data_ptr),
                                      std::move(sharding)));
  }

  // Creates an array from a host buffer. The buffer will be used directly
  // without a copy if the copy semantics allow it and the layout is row major
  // and dense.
  static absl::StatusOr<tsl::RCReference<NanoArray>> FromBuffer(
      NanoIfrtClient* client, void* data, ifrt::DType dtype, ifrt::Shape shape,
      std::shared_ptr<const ifrt::Sharding> sharding,
      std::optional<absl::Span<const int64_t>> byte_strides, bool make_copy,
      std::function<void()> on_done_with_host_buffer) {
    auto size = dtype.byte_size().value_or(0) * shape.num_elements();
    TF_RET_CHECK(size > 0);
    DataPtr data_ptr;

    bool layout_compatible = LayoutCompatible(dtype, shape, byte_strides);
    bool aligned = reinterpret_cast<uintptr_t>(data) % Align() == 0;

    if (!layout_compatible || !aligned) {
      // Input is not aligned, or has a weird layout, so we need to copy it.
      make_copy = true;
    }

    if (ABSL_PREDICT_FALSE(make_copy)) {
      TF_ASSIGN_OR_RETURN(data_ptr, AllocateData(size));
      if (layout_compatible) {
        // Input has a compatible layout, so we can just do a memcpy.
        memcpy(data_ptr.get(), data, size);
      } else {
        // Input has an incompatible layout, so we need to copy it with an
        // appropriate stride.
        TF_ASSIGN_OR_RETURN(auto dense_strides, DenseByteStrides(dtype, shape));
        TF_RETURN_IF_ERROR(CopyWithByteStrides(
            reinterpret_cast<std::byte*>(data_ptr.get()), dense_strides,
            reinterpret_cast<std::byte*>(data),
            byte_strides.value_or(dense_strides), shape.dims(),
            dtype.byte_size().value()));
      }
      // We're done with the input buffer, so we can allow the caller to clean
      // it up.
      if (on_done_with_host_buffer) on_done_with_host_buffer();
    } else {
      // We're allowed to keep the input buffer, and it's dense and row major,
      // so we can just use it directly.
      data_ptr = DataPtr(
          data, [done = std::move(on_done_with_host_buffer)](void* ptr) {
            if (done) done();
          });
    }
    TF_RET_CHECK(data_ptr != nullptr);
    return tsl::TakeRef(new NanoArray(client, dtype, std::move(shape),
                                      std::move(data_ptr),
                                      std::move(sharding)));
  }

  const DataPtr& data() const { return data_; }

  // Copies a sub-array of the given size from src to dst. The dst array must
  // already be allocated and of the correct type and shape. Values outside of
  // the specified sub-array of dst will be left untouched.
  //
  // This is mostly intended to support sharding and assembling.
  static absl::Status CopySubArray(NanoArray& dst,
                                   absl::Span<const int64_t> dst_loc,
                                   NanoArray& src,
                                   absl::Span<const int64_t> src_loc,
                                   absl::Span<const int64_t> size) {
    // Make sure the arrays are the same type and the type is supported.
    TF_RET_CHECK(dst.dtype() == src.dtype());
    TF_RET_CHECK(dst.dtype().byte_size().has_value());

    // Make sure all the dims are compatible.
    TF_RET_CHECK(dst.shape().dims().size() == size.size());
    TF_RET_CHECK(src.shape().dims().size() == size.size());
    TF_RET_CHECK(dst.shape().dims().size() == size.size());
    TF_RET_CHECK(dst_loc.size() == size.size());
    TF_RET_CHECK(src_loc.size() == size.size());

    // Make sure what we're copying is within the bounds of the arrays.
    for (size_t i = 0; i < size.size(); ++i) {
      TF_RET_CHECK(dst_loc[i] + size[i] <= dst.shape().dims()[i]);
      TF_RET_CHECK(src_loc[i] + size[i] <= src.shape().dims()[i]);
    }

    int64_t element_size = dst.dtype().byte_size().value();

    // Returns the size of a row in bytes for the given shape.
    auto row_size = [=](absl::Span<const int64_t> shape) {
      if (shape.empty()) return element_size;  // Scalar.
      return shape.back() * element_size;
    };

    // Since this is always row major, we can do one memcpy per row, and rows
    // will always be evenly spaces within the arrays.
    int64_t src_row_stride = row_size(src.shape().dims());
    int64_t dst_row_stride = row_size(dst.shape().dims());
    int64_t copy_row_size = row_size(size);

    // How many rows do we have to copy?
    int64_t copy_num_rows = 1;
    for (int64_t i = 0; i + 1 < size.size(); ++i) {
      copy_num_rows *= size[i];
    }

    // Returns a pointer to the given position in the array.
    auto get_row_ptr = [&](NanoArray& array,
                           absl::Span<const int64_t> position) -> std::byte* {
      size_t offset = 0;
      size_t stride = 1;
      for (int i = position.size() - 1; i >= 0; --i) {
        offset += stride * position[i];
        stride *= array.shape().dims()[i];
      }
      offset *= element_size;
      return static_cast<std::byte*>(array.data().get()) + offset;
    };

    // Get the pointers to the start of the rows we're copying.
    std::byte* dst_row_start = get_row_ptr(dst, dst_loc);
    std::byte* src_row_start = get_row_ptr(src, src_loc);

    // Copy the rows.
    for (int64_t i = 0; i < copy_num_rows; ++i) {
      memcpy(dst_row_start, src_row_start, copy_row_size);
      dst_row_start += dst_row_stride;
      src_row_start += src_row_stride;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<tsl::RCReference<NanoArray>>> Disassemble() {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    if (sharding().IsFullyReplicated()) {
      if (sharding().devices()->size() == 1) {
        // Only one device and one shard, so we can just return a reference to
        // this array.
        return std::vector<tsl::RCReference<NanoArray>>{tsl::FormRef(this)};
      }

      // If the array is fully replicated and there are multiple "devices", we
      // need to make one "copy" per device.
      std::vector<tsl::RCReference<NanoArray>> shards;
      shards.reserve(sharding().devices()->size());
      for (auto* device : sharding().devices()->devices()) {
        auto one_device_sharding = ifrt::SingleDeviceSharding::Create(
            device, sharding().memory_kind());
        shards.push_back(
            tsl::TakeRef(new NanoArray(nano_client(), dtype_, shape_, data_,
                                       std::move(one_device_sharding))));
      }
      return shards;
    }

    // The array is sharded, copy the appropriate sub-arrays.
    TF_ASSIGN_OR_RETURN(auto index_domains, sharding().IndexDomains(shape()));
    TF_RET_CHECK(index_domains.size() == sharding().devices()->size());
    std::vector<tsl::RCReference<NanoArray>> shards;
    shards.reserve(index_domains.size());
    for (int i = 0; i < index_domains.size(); ++i) {
      const auto& index_domain = index_domains[i];
      auto* device = sharding().devices()->devices()[i];
      auto one_device_sharding =
          ifrt::SingleDeviceSharding::Create(device, sharding().memory_kind());
      TF_ASSIGN_OR_RETURN(
          auto shard,
          NanoArray::Allocate(nano_client(), dtype(), index_domain.shape(),
                              std::move(one_device_sharding)));
      TF_RETURN_IF_ERROR(NanoArray::CopySubArray(
          // To the origin of this shard.
          *shard, ifrt::Index::Zeros(shape().dims().size()).elements(),
          // From the assembled array.
          *this, index_domain.origin().elements(),
          // The in the shape of this shard.
          index_domain.shape().dims()));
      shards.push_back(std::move(shard));
    }
    return shards;
  }

  NanoRtExecutable::Argument AsArgument() {
    return NanoRtExecutable::Argument(
        reinterpret_cast<std::byte*>(data_.get()),
        dtype_.byte_size().value() * shape_.num_elements());
  }

  NanoRtExecutable::Result AsResult() {
    return NanoRtExecutable::Result(
        reinterpret_cast<std::byte*>(data_.get()),
        dtype_.byte_size().value() * shape_.num_elements());
  }

  std::string DebugString() const override {
    return absl::StrCat("NanoArray(", dtype_.DebugString(), ", ",
                        shape_.DebugString(), ", @",
                        reinterpret_cast<uintptr_t>(data_.get()), ")");
  }

  ifrt::Future<> Delete() override {
    data_ = nullptr;
    return Ready();
  }

  bool IsDeleted() const override { return data_ == nullptr; }

  ifrt::DType dtype() const override { return dtype_; }

  const ifrt::Shape& shape() const override { return shape_; }

  const ifrt::Sharding& sharding() const override { return *sharding_; }

  absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> shared_ptr_sharding()
      const override {
    return sharding_;
  }

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> layout() const override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    return std::make_shared<PjRtLayout>(xla::Layout(shape().dims()));
  }

  absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(
      ifrt::ArrayCopySemantics semantics) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    TF_ASSIGN_OR_RETURN(auto shards, Disassemble());
    return std::vector<tsl::RCReference<Array>>(shards.begin(), shards.end());
  }

  absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(
      ifrt::ArrayCopySemantics array_copy_semantics,
      ifrt::SingleDeviceShardSemantics single_device_shard_semantics) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    return DisassembleIntoSingleDeviceArrays(array_copy_semantics);
  }

  absl::StatusOr<tsl::RCReference<Array>> FullyReplicatedShard(
      ifrt::ArrayCopySemantics semantics) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    return tsl::FormRef(this);
  }

  ifrt::Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ifrt::ArrayCopySemantics semantics) override {
    // Run everything in a lambda so we can use error macros and convert to a
    // future once.
    return Ready([&] {
      TF_RETURN_IF_ERROR(ValidateNotDeleted());
      TF_ASSIGN_OR_RETURN(xla::PrimitiveType xla_dtype,
                          ifrt::ToPrimitiveType(dtype()));
      if (!byte_strides.has_value() ||
          xla::HasMajorToMinorLayout(xla_dtype, shape().dims(),
                                     *byte_strides)) {
        memcpy(data, data_.get(),
               dtype().byte_size().value() * shape().num_elements());
      } else {
        TF_ASSIGN_OR_RETURN(auto in_strides,
                            DenseByteStrides(dtype(), shape()));
        TF_RETURN_IF_ERROR(CopyWithByteStrides(
            reinterpret_cast<std::byte*>(data), *byte_strides,
            reinterpret_cast<std::byte*>(data_.get()), in_strides,
            shape().dims(), dtype().byte_size().value()));
      }
      return absl::OkStatus();
    }());
  }

  static char ID;  // NOLINT

 private:
  // Returns true if the given data type, shape, and strides are compatible
  // with NanoArray (we can either use this memory directly or memcpy it into
  // our own memory).
  static bool LayoutCompatible(
      ifrt::DType dtype, const ifrt::Shape& shape,
      std::optional<absl::Span<const int64_t>> byte_strides) {
    if (!dtype.byte_size().has_value()) {
      return false;
    }
    auto xla_dtype = ifrt::ToPrimitiveType(dtype);
    if (!xla_dtype.ok()) {
      return false;
    }
    if (!byte_strides.has_value()) {
      return true;
    }
    return xla::HasMajorToMinorLayout(*xla_dtype, shape.dims(), *byte_strides);
  }

  // Returns the byte strides for a dense array with the given type and shape.
  static absl::StatusOr<absl::InlinedVector<int64_t, 4>> DenseByteStrides(
      ifrt::DType dtype, ifrt::Shape shape) {
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType xla_dtype,
                        ifrt::ToPrimitiveType(dtype));
    auto xla_shape = xla::ShapeUtil::MakeShape(xla_dtype, shape.dims());
    auto strides = xla::ShapeUtil::ByteStrides(xla_shape);
    if (!strides.has_value()) {
      return InvalidArgument("Couldn't compute byte strides for shape: %s",
                             xla_shape.ToString());
    }
    return std::move(*strides);
  }

  // Allocates an aligned buffer of the given size.
  static absl::StatusOr<DataPtr> AllocateData(size_t size) {
    DataPtr data_ptr(
        tsl::port::AlignedMalloc(std::max<size_t>(size, Align()), Align()),
        [](void* ptr) { tsl::port::AlignedFree(ptr); });
    if (ABSL_PREDICT_FALSE(data_ptr == nullptr)) {
      return Internal("Failed to allocate memory for NanoArray. Errno: %s",
                      strerror(errno));
    }
    // Suppress msan warnings for memory that will be initialized by the
    // jit-compiled code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(data_ptr.get(), size);
    return data_ptr;
  }

  // Copies data between two buffers that represent the same shape but have
  // different byte strides. This is a recursive method that peels back dims
  // until we get to a scalar, which isn't very efficient but the common case
  // is expected to be a row major array without padding.
  static absl::Status CopyWithByteStrides(
      std::byte* dst, absl::Span<const int64_t> dst_byte_strides,
      const std::byte* src, absl::Span<const int64_t> src_byte_strides,
      absl::Span<const int64_t> dims, int64_t elem_size) {
    TF_RET_CHECK(dims.size() == dst_byte_strides.size());
    TF_RET_CHECK(dims.size() == src_byte_strides.size());
    // Scalar. Just copy it.
    if (dims.empty()) {
      memcpy(dst, src, elem_size);
      return absl::OkStatus();
    }
    // Peel back dims recursively until we get to a scalar.
    for (int64_t i = 0; i < dims[0]; ++i) {
      TF_RETURN_IF_ERROR(CopyWithByteStrides(dst, dst_byte_strides.subspan(1),
                                             src, src_byte_strides.subspan(1),
                                             dims.subspan(1), elem_size));
      dst += dst_byte_strides[0];
      src += src_byte_strides[0];
    }
    return absl::OkStatus();
  }

  ifrt::DType dtype_;
  ifrt::Shape shape_;
  DataPtr data_;
  std::shared_ptr<const ifrt::Sharding> sharding_;
};

ABSL_ATTRIBUTE_UNUSED char NanoArray::ID = 'A';  // NOLINT

// Sharded array implementation. Represents an array that should be assembled
// from multiple arrays, but we aren't sure how to assemble it yet.
class ShardedNanoArray final : public NanoValue<ShardedNanoArray, ifrt::Array> {
 public:
  // Creates an array from the given shards. Note that if we can assemble the
  // array using the given sharding, this method will return a NanoArray.
  static absl::StatusOr<tsl::RCReference<ifrt::Array>> FromShards(
      NanoIfrtClient* client, ifrt::Shape shape,
      std::shared_ptr<const ifrt::Sharding> sharding,
      std::vector<tsl::RCReference<NanoArray>> shards) {
    if (shards.empty()) {
      return InvalidArgument("Can't create a sharded array with no shards.");
    }
    xla::ifrt::DType dtype = shards[0]->dtype();

    auto array = tsl::TakeRef(new ShardedNanoArray(
        client, dtype, shape, sharding, std::move(shards)));

    // Try to eagerly assemble the array. Sometimes this cannot be done
    // because arrays are loaded with a simple per device sharding and we
    // won't know how to assemble it until the program is run.
    if (auto dense_array = array->Assemble(sharding); dense_array.ok()) {
      return dense_array;
    }

    // If we can't assemble the array, we'll just return the sharded array. It
    // will be assembled at execution time when we know the actual sharding.
    return array;
  }

  const std::vector<tsl::RCReference<NanoArray>>& shards() { return shards_; }

  // Assembles the array using the given sharding to prepare it as an input to
  // execution. If this array has already been assembled using the given
  // sharding, this method will return the cached result. This optimizes a
  // common case where a checkpoint is loaded with an unknown sharding, but
  // then we find the real sharding when the program is run.
  absl::StatusOr<tsl::RCReference<NanoArray>> AssembleForExecution(
      std::shared_ptr<const ifrt::Sharding> sharding) {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    absl::call_once(assemble_once_, [this, sharding]() {
      assemble_result_ = Assemble(sharding);
    });
    TF_RETURN_IF_ERROR(assemble_result_.status());
    if (assemble_result_.value()->shared_ptr_sharding() != sharding) {
      // Bleh... We cached the wrong sharding somehow. This means one sharded
      // array was an input to two different programs with different
      // shardings, this should be unlikely.
      return Assemble(sharding);
    }
    return assemble_result_;
  }

  ifrt::Future<> Delete() override {
    // Sharded arrays are never borrowed like dense arrays are, so we can just
    // clear the shards and let them be destroyed.
    shards_.clear();
    assemble_result_ = absl::Status(absl::StatusCode::kUnavailable, "");
    return Ready();
  }

  bool IsDeleted() const override { return shards_.empty(); }

  std::string DebugString() const override {
    auto result =
        absl::StrCat("ShardedNanoArray(", dtype_.DebugString(), ", ",
                     shape_.DebugString(), ", ", sharding_->DebugString());
    for (const auto& shard : shards_) {
      absl::StrAppend(&result, ", ", shard->DebugString());
    }
    absl::StrAppend(&result, ")");
    return result;
  }

  ifrt::DType dtype() const override { return dtype_; }

  const ifrt::Shape& shape() const override { return shape_; }

  const ifrt::Sharding& sharding() const override { return *sharding_; }

  absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> shared_ptr_sharding()
      const override {
    return sharding_;
  }

  absl::StatusOr<std::shared_ptr<const PjRtLayout>> layout() const override {
    return std::make_shared<PjRtLayout>(xla::Layout(shape().dims()));
  }

  absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(
      ifrt::ArrayCopySemantics semantics) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    return std::vector<tsl::RCReference<Array>>(shards_.begin(), shards_.end());
  }

  absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(
      ifrt::ArrayCopySemantics array_copy_semantics,
      ifrt::SingleDeviceShardSemantics single_device_shard_semantics) override {
    return DisassembleIntoSingleDeviceArrays(array_copy_semantics);
  }

  absl::StatusOr<tsl::RCReference<Array>> FullyReplicatedShard(
      ifrt::ArrayCopySemantics semantics) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    return tsl::FormRef(this);
  }

  ifrt::Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ifrt::ArrayCopySemantics semantics) override {
    return Ready(Internal("Cannot copy sharded array to host buffer."));
  }

  ABSL_ATTRIBUTE_UNUSED static char ID;  // NOLINT

 private:
  ShardedNanoArray(NanoIfrtClient* client, ifrt::DType dtype, ifrt::Shape shape,
                   std::shared_ptr<const ifrt::Sharding> sharding,
                   std::vector<tsl::RCReference<NanoArray>> shards)
      : NanoValue<ShardedNanoArray, ifrt::Array>(client),
        dtype_(std::move(dtype)),
        shape_(std::move(shape)),
        sharding_(std::move(sharding)),
        shards_(std::move(shards)) {}

  absl::StatusOr<tsl::RCReference<NanoArray>> Assemble(
      std::shared_ptr<const ifrt::Sharding> sharding) {
    TF_ASSIGN_OR_RETURN(auto index_domains, sharding->IndexDomains(shape()));
    if (index_domains.size() != shards_.size()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Number of index domains ", index_domains.size(),
                       " not equal to number of arrays ", shards_.size()));
    }

    for (int i = 0; i < index_domains.size(); ++i) {
      if (index_domains[i].shape() != shards_[i]->shape()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Index domain ", index_domains[i].shape().DebugString(),
            " not equal to array shape ", shards_[i]->shape().DebugString()));
      }
    }

    // If the sharding is replicated in any way, this comparator will dedupe
    // arrays that have the same logical destination.
    struct IndexDomainCmp {
      bool operator()(const ifrt::IndexDomain& a,
                      const ifrt::IndexDomain& b) const {
        return std::lexicographical_compare(
            a.origin().elements().begin(), a.origin().elements().end(),
            b.origin().elements().begin(), b.origin().elements().end());
      }
    };

    // Index the arrays by where we are copying them to. Note that this will
    // implicitly filter out replicated shards since they will have the same
    // destination in the assembled array.
    absl::btree_map<ifrt::IndexDomain, NanoArray*, IndexDomainCmp>
        index_domain_device_arrays;
    for (int i = 0; i < index_domains.size(); ++i) {
      index_domain_device_arrays[index_domains[i]] = shards_[i].get();
    }

    TF_ASSIGN_OR_RETURN(auto result, NanoArray::Allocate(nano_client(), dtype(),
                                                         shape(), sharding));

    // Copy the shards into the final array.
    auto shard_origin = ifrt::Index::Zeros(shards_[0]->shape().dims().size());
    for (const auto& [index_domain, shard] : index_domain_device_arrays) {
      TF_RETURN_IF_ERROR(NanoArray::CopySubArray(
          *result, index_domain.origin().elements(), *shard,
          shard_origin.elements(), shard->shape().dims()));
    }

    return result;
  }

  ifrt::DType dtype_;
  ifrt::Shape shape_;
  std::shared_ptr<const ifrt::Sharding> sharding_;
  std::vector<tsl::RCReference<NanoArray>> shards_;

  absl::once_flag assemble_once_;
  absl::StatusOr<tsl::RCReference<NanoArray>> assemble_result_;
};

char ShardedNanoArray::ID = 'A';  // NOLINT

// Tuple implementation.
class NanoTuple final : public NanoValue<NanoTuple, ifrt::Tuple> {
 public:
  explicit NanoTuple(NanoIfrtClient* client,
                     absl::Span<tsl::RCReference<ifrt::Value>> values)
      : NanoValue<NanoTuple, ifrt::Tuple>(client),
        values_(values.begin(), values.end()) {}

  ifrt::Future<> Delete() override {
    for (auto& value : values_) {
      value->Delete();
    }
    values_.clear();
    deleted_ = true;
    return Ready();
  }

  bool IsDeleted() const override {
    for (auto& value : values_) {
      if (value->IsDeleted()) {
        return true;
      }
    }
    return deleted_;
  }

  // Returns the arity of the tuple.
  int Arity() override { return values_.size(); }

  // Unpacks the tuple into its constituent pieces.
  absl::Status Unpack(
      absl::Span<tsl::RCReference<ifrt::Value>> values) override {
    TF_RETURN_IF_ERROR(ValidateNotDeleted());
    if (values.size() != values_.size()) {
      return InvalidArgument("Tuple arity mismatch: expected %d, got %d",
                             values_.size(), values.size());
    }
    for (int i = 0; i < values_.size(); ++i) {
      values[i] = values_[i];
    }
    return absl::OkStatus();
  }

  std::string DebugString() const override {
    std::string result = "NanoTuple(";
    for (const auto& value : values_) {
      absl::StrAppend(&result, value->DebugString(), ", ");
    }
    absl::StrAppend(&result, ")");
    return result;
  }

  static char ID;  // NOLINT

 private:
  bool deleted_ = false;
  std::vector<tsl::RCReference<ifrt::Value>> values_;
};

ABSL_ATTRIBUTE_UNUSED char NanoTuple::ID = 'T';  // NOLINT

// Executable implementation.
class NanoExecutable final
    : public llvm::RTTIExtends<NanoExecutable, ifrt::LoadedExecutable> {
 public:
  // Creates a NanoExecutable from an ifrt::Program.
  static absl::StatusOr<std::unique_ptr<NanoExecutable>> Create(
      NanoIfrtClient* client, std::unique_ptr<ifrt::Program> program) {
    auto* xla_program = llvm::dyn_cast<ifrt::HloProgram>(program.get());
    if (xla_program == nullptr) {
      return InvalidArgument("NanoRT requires an HloProgram");
    }
    XlaComputation computation;
    TF_RETURN_IF_ERROR(MlirToXlaComputation(xla_program->mlir_module,
                                            computation, false, true, false));
    TF_ASSIGN_OR_RETURN(auto nano_executable,
                        client->nano_client()->Compile(computation));

    if (computation.proto().computations().size() != 1) {
      return InvalidArgument(
          "NanoRT only supports single-computation programs, got %d",
          computation.proto().computations().size());
    }

    TF_ASSIGN_OR_RETURN(auto program_shape, computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(auto proto_input_shardings,
                        GetInputShardings(program_shape, computation));
    TF_ASSIGN_OR_RETURN(auto proto_output_shardings,
                        GetOutputShardings(program_shape, computation));
    auto input_shardings =
        IfrtShardingsFromProto(client, proto_input_shardings);
    auto output_shardings =
        IfrtShardingsFromProto(client, proto_output_shardings);

    return absl::WrapUnique(new NanoExecutable(
        client, std::move(computation), std::move(program_shape),
        std::move(nano_executable), std::move(input_shardings),
        std::move(output_shardings)));
  }

  ifrt::Client* client() const override { return client_; }

  absl::string_view name() const override { return program_.name(); }

  absl::StatusOr<ExecuteResult> Execute(
      absl::Span<tsl::RCReference<ifrt::Array>> args,
      const ExecuteOptions& options,
      std::optional<ifrt::DeviceListRef> devices) override {
    if (ABSL_PREDICT_FALSE(args.size() != input_shardings_.size())) {
      return InvalidArgument(
          "Number of arguments %d is not what executable expects %d",
          args.size(), input_shardings_.size());
    }

    // Convert the ifrt arrays to nano arrays. 'tmp' holds any arrays that had
    // to be assembled.
    std::vector<tsl::RCReference<NanoArray>> tmp;
    TF_ASSIGN_OR_RETURN(auto nano_args,
                        NanoArgumentsFromIfrtArguments(args, tmp));

    TF_ASSIGN_OR_RETURN(auto result_arrays, AllocateResults());
    std::vector<xla::cpu::NanoRtExecutable::Result> nano_results;
    nano_results.reserve(result_arrays.size());
    for (auto& result_array : result_arrays) {
      nano_results.push_back(result_array->AsResult());
    }

    NanoRtExecutable::ManagedTemp<128> temp_buffer(
        executable_->temp_buffer_size());
    auto event = executable_->Execute(nano_args, nano_results, temp_buffer);

    // TODO(jsoyke): Consider making this non-blocking if we ever use this
    // interface for models that require threading, or if we want to delay
    // execution until we know where the outputs will be stored.
    tsl::BlockUntilReady(event);

    if (ABSL_PREDICT_FALSE(event.IsError())) {
      return event.GetError();
    }
    if (ABSL_PREDICT_FALSE(!event.IsConcrete())) {
      return Internal("NanoRT result is not concrete.");
    }

    ExecuteResult result;
    if (options.fill_status) {
      result.status = Ready();
    }
    result.outputs.insert(result.outputs.end(),
                          std::make_move_iterator(result_arrays.begin()),
                          std::make_move_iterator(result_arrays.end()));
    return result;
  }

  // Returns a fingerprint of this executable.
  absl::StatusOr<std::optional<std::string>> Fingerprint() const override {
    return absl::UnimplementedError("Fingerprint is not implemented.");
  }

  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError("Serialize is not implemented.");
  }

  ifrt::Future<> GetReadyFuture() const override { return Ready(); }

  int num_devices() const override { return 1; }

  int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return absl::UnimplementedError(
        "GetCompiledMemoryStats is not implemented.");
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    auto shardings = GetInputShardings(program_shape_, program_);
    if (!shardings.ok()) return std::nullopt;
    return *shardings;
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    auto shardings = GetOutputShardings(program_shape_, program_);
    if (!shardings.ok()) return std::nullopt;
    return *shardings;
  }

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override {
    std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
    layouts.reserve(program_shape_.parameters().size());
    for (const auto& shape : program_shape_.parameters()) {
      layouts.push_back(
          std::make_shared<PjRtLayout>(xla::Layout(shape.dimensions())));
    }
    return layouts;
  }

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    const auto& result_shape = program_shape_.result();
    const auto result_shapes =
        result_shape.IsTuple()
            ? absl::MakeConstSpan(result_shape.tuple_shapes())
            : absl::MakeConstSpan(&result_shape, 1);
    std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
    layouts.reserve(result_shapes.size());
    for (const auto& shape : result_shapes) {
      layouts.push_back(
          std::make_shared<PjRtLayout>(xla::Layout(shape.dimensions())));
    }
    return layouts;
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    std::vector<std::shared_ptr<HloModule>> hlo_modules(1);
    TF_ASSIGN_OR_RETURN(
        hlo_modules[0],
        HloModule::CreateFromProto(program_.proto(), HloModuleConfig()));
    return hlo_modules;
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    std::vector<std::vector<absl::string_view>> memory_kinds;
    memory_kinds.reserve(output_shardings_.size());
    for (const auto& _ : output_shardings_) {
      memory_kinds.push_back({kMemoryKind});
    }
    return memory_kinds;
  }

  absl::StatusOr<ifrt::AttributeMap> GetCostAnalysis() const override {
    return absl::UnimplementedError("GetCostAnalysis is not implemented.");
  }

  ifrt::Future<> Delete() override {
    client_ = nullptr;
    program_ = {};
    program_shape_ = {};
    executable_.reset();
    input_shardings_.clear();
    output_shardings_.clear();
    return Ready();
  }

  bool IsDeleted() const override { return executable_ == nullptr; }

  absl::Span<ifrt::Device* const> addressable_devices() const override {
    return client_->addressable_devices();
  }

  static char ID;  // NOLINT

 private:
  NanoExecutable(NanoIfrtClient* client, XlaComputation program,
                 ProgramShape program_shape,
                 std::unique_ptr<NanoRtExecutable> executable,
                 std::vector<std::shared_ptr<ifrt::Sharding>> input_shardings,
                 std::vector<std::shared_ptr<ifrt::Sharding>> output_shardings)
      : client_(client),
        program_(std::move(program)),
        program_shape_(std::move(program_shape)),
        executable_(std::move(executable)),
        input_shardings_(std::move(input_shardings)),
        output_shardings_(std::move(output_shardings)) {}

  // Converts an OpSharding proto (from an HLO Instruction) to an ifrt
  // sharding.
  static std::vector<std::shared_ptr<ifrt::Sharding>> IfrtShardingsFromProto(
      NanoIfrtClient* client, absl::Span<const OpSharding> shardings) {
    std::vector<std::shared_ptr<ifrt::Sharding>> result;
    result.reserve(shardings.size());
    for (const auto& sharding : shardings) {
      if (sharding.type() == OpSharding::REPLICATED ||
          sharding.type() == OpSharding::MAXIMAL) {
        result.push_back(client->default_sharding());
        continue;
      }
      int num_tiles = 1;
      for (const auto dim : sharding.tile_assignment_dimensions()) {
        num_tiles *= dim;
      }
      // Repeat the device for each tile. We only have one device anyway so
      // just used the first.
      auto device_list = ifrt::BasicDeviceList::Create(
          ifrt::BasicDeviceList::Devices(num_tiles, client->devices()[0]));
      auto xla_sharding = *HloSharding::FromProto(sharding);
      result.push_back(ifrt::HloSharding::Create(
          std::move(device_list), client->devices()[0]->Memories()[0]->Kind(),
          std::move(xla_sharding)));
    }
    return result;
  }

  static absl::StatusOr<std::vector<OpSharding>> GetInputShardings(
      const ProgramShape& program_shape, const XlaComputation& computation) {
    std::vector<OpSharding> shardings(program_shape.parameters().size());
    for (const auto& instruction :
         computation.proto().computations(0).instructions()) {
      if (instruction.opcode() == "parameter" && instruction.has_sharding()) {
        if (instruction.parameter_number() >= shardings.size()) {
          return InvalidArgument(
              "Parameter number %d is out of range for program with %d "
              "parameters.",
              instruction.parameter_number(),
              program_shape.parameters().size());
        }
        shardings[instruction.parameter_number()] = instruction.sharding();
      }
    }
    return shardings;
  }

  static absl::StatusOr<std::vector<OpSharding>> GetOutputShardings(
      const ProgramShape& program_shape, const XlaComputation& computation) {
    const auto& result_shape = program_shape.result();

    int output_id = computation.proto().computations(0).root_id();

    std::vector<OpSharding> shardings(
        (result_shape.IsTuple() ? result_shape.tuple_shapes().size() : 1));

    for (const auto& instruction :
         computation.proto().computations(0).instructions()) {
      // We found a sharded output instruction.
      if (instruction.id() == output_id && instruction.has_sharding()) {
        if (result_shape.IsTuple()) {
          TF_RET_CHECK(instruction.sharding().tuple_shardings().size() ==
                       result_shape.tuple_shapes().size());
          for (int i = 0; i < instruction.sharding().tuple_shardings().size();
               ++i) {
            shardings[i] = instruction.sharding().tuple_shardings()[i];
          }
        } else {
          shardings[0] = instruction.sharding();
        }
      }
    }
    return shardings;
  }

  // Allocates the results for the program.
  absl::StatusOr<std::vector<tsl::RCReference<NanoArray>>> AllocateResults() {
    const auto& result_shape = program_shape_.result();
    const auto result_shapes =
        result_shape.IsTuple()
            ? absl::MakeConstSpan(result_shape.tuple_shapes())
            : absl::MakeConstSpan(&result_shape, 1);
    TF_RET_CHECK(result_shapes.size() == output_shardings_.size());

    std::vector<tsl::RCReference<NanoArray>> result_arrays;
    result_arrays.reserve(result_shapes.size());

    for (int i = 0; i < result_shapes.size(); ++i) {
      TF_ASSIGN_OR_RETURN(auto ifrt_type,
                          ifrt::ToDType(result_shapes[i].element_type()));
      ifrt::Shape ifrt_shape(result_shapes[i].dimensions());
      TF_ASSIGN_OR_RETURN(
          result_arrays.emplace_back(),
          NanoArray::Allocate(client_, ifrt_type, std::move(ifrt_shape),
                              output_shardings_[i]));
    }

    return result_arrays;
  }

  // Converts the ifrt arrays to nano arguments. 'tmp' holds any arrays that
  // had to be assembled.
  absl::StatusOr<std::vector<xla::cpu::NanoRtExecutable::Argument>>
  NanoArgumentsFromIfrtArguments(
      absl::Span<tsl::RCReference<ifrt::Array>> args,
      std::vector<tsl::RCReference<NanoArray>>& tmp) {
    std::vector<xla::cpu::NanoRtExecutable::Argument> nano_args;
    nano_args.reserve(args.size());

    for (int i = 0; i < args.size(); ++i) {
      auto* nano_array = llvm::dyn_cast_or_null<NanoArray>(args[i].get());
      if (ABSL_PREDICT_FALSE(nano_array == nullptr)) {
        // The input isn't a nano array, so it must be a sharded array.
        auto* sharded_array =
            llvm::dyn_cast_or_null<ShardedNanoArray>(args[i].get());
        if (sharded_array == nullptr) {
          return InvalidArgument(
              "Argument is not a NanoArray or ShardedNanoArray: %s",
              args[i]->DebugString());
        }
        TF_ASSIGN_OR_RETURN(
            auto dense_array,
            sharded_array->AssembleForExecution(input_shardings_[i]));
        nano_array = dense_array.get();
        tmp.push_back(std::move(dense_array));
      }
      nano_args.push_back(nano_array->AsArgument());
    }

    return nano_args;
  }

  NanoIfrtClient* client_;
  XlaComputation program_;
  ProgramShape program_shape_;
  std::unique_ptr<NanoRtExecutable> executable_;
  std::vector<std::shared_ptr<ifrt::Sharding>> input_shardings_;
  std::vector<std::shared_ptr<ifrt::Sharding>> output_shardings_;
};

ABSL_ATTRIBUTE_UNUSED char NanoExecutable::ID = 'E';  // NOLINT

// Compiler implementation.
class NanoCompiler final
    : public llvm::RTTIExtends<NanoCompiler, ifrt::Compiler> {
 public:
  explicit NanoCompiler(NanoIfrtClient* client) : client_(client) {}

  absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
      std::unique_ptr<ifrt::Program> program,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return NanoExecutable::Create(client_, std::move(program));
  }

  absl::StatusOr<std::unique_ptr<ifrt::Executable>> Compile(
      std::unique_ptr<ifrt::Program> program, const ifrt::Topology& topology,
      std::unique_ptr<ifrt::CompileOptions> options) override {
    return absl::UnimplementedError("Partial compilation is not implemented.");
  }

  absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>>
  DeserializeLoadedExecutable(
      absl::string_view serialized,
      std::unique_ptr<ifrt::DeserializeExecutableOptions> options) override {
    return absl::UnimplementedError(
        "DeserializeLoadedExecutable is not implemented.");
  }
  static char ID;  // NOLINT

 private:
  NanoIfrtClient* client_;
};

ABSL_ATTRIBUTE_UNUSED char NanoCompiler::ID = 'C';  // NOLINT

// Memory implementation. There is only one address space so this doesn't do
// much.
class NanoMemory final : public llvm::RTTIExtends<NanoMemory, ifrt::Memory> {
 public:
  explicit NanoMemory(NanoIfrtClient* client) : client_(client) {}

  ifrt::MemoryId Id() const override { return ifrt::MemoryId(0); }

  const ifrt::MemoryKind& Kind() const override {
    static ifrt::MemoryKind mem_kind(kMemoryKind);
    return mem_kind;
  }

  absl::string_view ToString() const override { return "NanoRT CPU Memory"; }
  absl::string_view DebugString() const override { return ToString(); }
  absl::Span<ifrt::Device* const> Devices() const override {
    return client_->devices();
  }

  static char ID;  // NOLINT

 private:
  NanoMemory() = default;

  NanoIfrtClient* client_;
};

ABSL_ATTRIBUTE_UNUSED char NanoMemory::ID = 'M';  // NOLINT

// Device implementation. There is only one device so this doesn't do much.
class NanoDevice final : public llvm::RTTIExtends<NanoDevice, ifrt::Device> {
 public:
  NanoDevice(NanoIfrtClient* client, ifrt::Memory* memory)
      : client_(client), memory_(memory) {}

  ifrt::Client* client() const override { return client_; }

  ifrt::DeviceId Id() const override { return ifrt::DeviceId(0); }

  const ifrt::AttributeMap& Attributes() const override {
    static auto attributes = new ifrt::AttributeMap({});
    return *attributes;
  }

  absl::string_view Kind() const override { return "cpu"; }

  absl::string_view ToString() const override { return "NanoRT CPU"; }

  absl::string_view DebugString() const override { return ToString(); }

  absl::StatusOr<ifrt::Memory*> DefaultMemory() const override {
    return memory_;
  }

  absl::Span<ifrt::Memory* const> Memories() const override {
    return absl::MakeConstSpan(&memory_, 1);
  }

  bool IsAddressable() const override { return true; }

  int ProcessIndex() const override { return 0; }

  static char ID;  // NOLINT

 private:
  NanoIfrtClient* client_;
  ifrt::Memory* memory_;
};

ABSL_ATTRIBUTE_UNUSED char NanoDevice::ID = 'D';  // NOLINT

}  // namespace

NanoIfrtClient::~NanoIfrtClient() = default;

std::shared_ptr<NanoIfrtClient> NanoIfrtClient::Create() {
  return CreateWithDevices(1);
}

std::shared_ptr<NanoIfrtClient> NanoIfrtClient::CreateWithDevices(
    int num_devices) {
  return std::shared_ptr<NanoIfrtClient>(new NanoIfrtClient(num_devices));
}

std::shared_ptr<ifrt::Sharding> NanoIfrtClient::default_sharding() const {
  return ifrt::SingleDeviceSharding::Create(device_.get(), ifrt::MemoryKind{});
}

absl::StatusOr<tsl::RCReference<ifrt::Array>>
NanoIfrtClient::MakeArrayFromHostBuffer(
    const void* data, ifrt::DType dtype, ifrt::Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> sharding,
    HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  bool make_copy = false;
  switch (semantics) {
    case HostBufferSemantics::kImmutableUntilTransferCompletes:
    case HostBufferSemantics::kImmutableOnlyDuringCall:
      make_copy = true;
      break;
    case HostBufferSemantics::kImmutableZeroCopy:
    case HostBufferSemantics::kMutableZeroCopy:
      make_copy = false;
      break;
  }
  return NanoArray::FromBuffer(this, const_cast<void*>(data), dtype,
                               std::move(shape), std::move(sharding),
                               byte_strides, make_copy,
                               std::move(on_done_with_host_buffer));
}

absl::StatusOr<std::vector<tsl::RCReference<ifrt::Array>>>
NanoIfrtClient::MakeArraysFromHostBufferShards(
    absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
    HostBufferSemantics semantics) {
  return ifrt::ClientMakeArraysFromHostBufferShards(this, specs, semantics);
}

absl::StatusOr<tsl::RCReference<ifrt::Array>>
NanoIfrtClient::AssembleArrayFromSingleDeviceArrays(
    ifrt::Shape shape,
    absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> sharding,
    absl::Span<tsl::RCReference<ifrt::Array>> arrays,
    ifrt::ArrayCopySemantics semantics) {
  std::vector<tsl::RCReference<NanoArray>> nano_arrays;
  nano_arrays.reserve(arrays.size());
  for (const auto& array : arrays) {
    auto* nano_array = llvm::dyn_cast_or_null<NanoArray>(array.get());
    if (nano_array == nullptr) {
      return InvalidArgument("Array is not a NanoArray: %s",
                             array->DebugString());
    }
    nano_arrays.push_back(tsl::FormRef(nano_array));
  }
  return ShardedNanoArray::FromShards(this, shape, sharding,
                                      std::move(nano_arrays));
}

absl::StatusOr<tsl::RCReference<ifrt::Array>>
NanoIfrtClient::AssembleArrayFromSingleDeviceArrays(
    ifrt::Shape shape,
    absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> sharding,
    absl::Span<tsl::RCReference<ifrt::Array>> arrays,
    ifrt::ArrayCopySemantics array_copy_semantics,
    ifrt::SingleDeviceShardSemantics single_device_shard_semantics) {
  return AssembleArrayFromSingleDeviceArrays(shape, sharding, arrays,
                                             array_copy_semantics);
}

absl::StatusOr<tsl::RCReference<ifrt::Array>>
NanoIfrtClient::AssembleArrayFromSingleDeviceArrays(
    ifrt::DType dtype, ifrt::Shape shape,
    absl::Nonnull<std::shared_ptr<const ifrt::Sharding>> sharding,
    absl::Span<tsl::RCReference<ifrt::Array>> arrays,
    ifrt::ArrayCopySemantics array_copy_semantics,
    ifrt::SingleDeviceShardSemantics single_device_shard_semantics) {
  // NanoRT devices always have at least one buffer, so we can use the buffer
  // dtype.
  TF_RET_CHECK(!arrays.empty());
  TF_RET_CHECK(dtype == arrays.front()->dtype());
  return AssembleArrayFromSingleDeviceArrays(shape, sharding, arrays,
                                             array_copy_semantics);
}

absl::StatusOr<std::vector<tsl::RCReference<ifrt::Array>>>
NanoIfrtClient::CopyArrays(absl::Span<tsl::RCReference<ifrt::Array>> arrays,
                           std::optional<ifrt::DeviceListRef> devices,
                           std::optional<ifrt::MemoryKind> memory_kind,
                           ifrt::ArrayCopySemantics semantics) {
  std::vector<tsl::RCReference<ifrt::Array>> result;
  result.reserve(arrays.size());
  for (const auto& array : arrays) {
    tsl::RCReference<ifrt::Array> copy;
    TF_ASSIGN_OR_RETURN(auto sharding, array->sharding().WithDeviceAssignment(
                                           devices, memory_kind));
    if (auto nano_array = llvm::dyn_cast_or_null<NanoArray>(array.get())) {
      copy = tsl::TakeRef(new NanoArray(this, nano_array->dtype(),
                                        nano_array->shape(), nano_array->data(),
                                        std::move(sharding)));
    } else if (auto sharded_nano_array =
                   llvm::dyn_cast_or_null<ShardedNanoArray>(array.get())) {
      std::vector<tsl::RCReference<NanoArray>> shards_copy;
      shards_copy.reserve(sharded_nano_array->shards().size());
      for (const auto& shard : sharded_nano_array->shards()) {
        shards_copy.push_back(tsl::TakeRef(
            new NanoArray(this, shard->dtype(), shard->shape(), shard->data(),
                          shard->shared_ptr_sharding())));
      }
      TF_ASSIGN_OR_RETURN(
          copy, ShardedNanoArray::FromShards(this, sharded_nano_array->shape(),
                                             std::move(sharding),
                                             std::move(shards_copy)));
    } else {
      return InvalidArgument("Array is not a NanoArray or ShardedNanoArray: %s",
                             array->DebugString());
    }
    TF_RET_CHECK(copy != nullptr);
    result.push_back(copy);
  }
  return result;
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
NanoIfrtClient::RemapArrays(
    const ifrt::RemapPlan& plan,
    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
    ifrt::ArrayCopySemantics semantics) {
  return absl::UnimplementedError("RemapArrays is not implemented.");
}

ifrt::Future<> NanoIfrtClient::GetReadyFuture(
    absl::Span<const tsl::RCReference<ifrt::Value>> values) {
  return Ready();
}

absl::StatusOr<tsl::RCReference<ifrt::Tuple>> NanoIfrtClient::MakeTuple(
    absl::Span<tsl::RCReference<ifrt::Value>> values) {
  return tsl::MakeRef<NanoTuple>(this, std::move(values));
}

absl::string_view NanoIfrtClient::runtime_type() const { return "nano"; }

absl::string_view NanoIfrtClient::platform_name() const {
  return xla::CpuName();
}

absl::string_view NanoIfrtClient::platform_version() const {
  return xla::CpuName();
}

ifrt::PlatformId NanoIfrtClient::platform_id() const {
  return tsl::Fingerprint64(platform_name());
}

const ifrt::AttributeMap& NanoIfrtClient::Attributes() const {
  static auto attributes = new ifrt::AttributeMap({});
  return *attributes;
}

int NanoIfrtClient::device_count() const { return devices_.size(); }

int NanoIfrtClient::addressable_device_count() const { return device_count(); }

absl::Span<ifrt::Device* const> NanoIfrtClient::devices() const {
  return devices_;
}

absl::Span<ifrt::Device* const> NanoIfrtClient::addressable_devices() const {
  return devices();
}

int NanoIfrtClient::process_index() const { return 0; }

absl::Span<xla::ifrt::Device* const> NanoIfrtClient::GetAllDevices() const {
  return devices();
}

absl::StatusOr<ifrt::DeviceAssignment>
NanoIfrtClient::GetDefaultDeviceAssignment(int num_replicas,
                                           int num_partitions) const {
  return ifrt::DeviceAssignment(num_replicas, num_partitions);
}

absl::StatusOr<ifrt::Device*> NanoIfrtClient::LookupDevice(
    ifrt::DeviceId device_id) const {
  return LookupAddressableDevice(device_id.value());
}

absl::StatusOr<ifrt::Device*> NanoIfrtClient::LookupAddressableDevice(
    int local_hardware_id) const {
  return device_.get();
}

ifrt::DeviceListRef NanoIfrtClient::MakeDeviceList(
    absl::Span<ifrt::Device* const> devices) const {
  return xla::ifrt::BasicDeviceList::Create(devices);
}

ifrt::Compiler* NanoIfrtClient::GetDefaultCompiler() { return compiler_.get(); }

absl::StatusOr<std::shared_ptr<ifrt::Topology>>
NanoIfrtClient::GetTopologyForDevices(
    const ifrt::DeviceListRef& devices) const {
  return absl::UnimplementedError("GetTopologyForDevices is not implemented.");
}

absl::StatusOr<std::shared_ptr<const PjRtLayout>>
NanoIfrtClient::GetDefaultLayout(ifrt::DType dtype,
                                 absl::Span<const int64_t> dims,
                                 ifrt::Device* device,
                                 xla::ifrt::MemoryKind memory_kind) const {
  return std::make_shared<PjRtLayout>(xla::Layout(dims));
}

NanoIfrtClient::NanoIfrtClient(int32_t num_devices)
    : compiler_(std::make_unique<NanoCompiler>(this)),
      memory_(std::make_unique<NanoMemory>(this)),
      device_(std::make_unique<NanoDevice>(this, memory_.get())),
      default_sharding_(
          ifrt::SingleDeviceSharding::Create(device_.get(), memory_->Kind())),
      devices_(num_devices, device_.get()) {}

char NanoIfrtClient::ID = 'N';  // NOLINT

}  // namespace xla::cpu
