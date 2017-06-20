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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include <set>
#include <string>
#include <utility>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/hlo_test_base_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct HloTestBase::EigenThreadPoolWrapper {
  std::unique_ptr<EigenThreadPoolWrapper> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

HloTestBase::HloTestBase()
    : backend_(Backend::CreateDefaultBackend().ConsumeValueOrDie()) {
  // TODO(b/62411181): get rid of this flag entirely when the usual debug flags
  // are piped to all HLO tests.
  test_hlo_dumper_ = [](const HloModule& module, const string& label) {
    legacy_flags::HloTestBaseFlags* flags = legacy_flags::GetHloTestBaseFlags();
    if (flags->xla_hlo_test_generate_hlo_graph) {
      const bool show_addresses = true;
      const bool show_layouts = true;
      hlo_graph_dumper::DumpGraph(*module.entry_computation(), label,
                                  show_addresses, show_layouts);
    }
  };
  VLOG(1) << "executing on platform " << backend_->platform()->Name();
}

HloTestBase::~HloTestBase() {
  // Deallocate all the memory allocated during the tests.
  for (auto& allocation : allocations_) {
    backend_->default_stream_executor()->Deallocate(&allocation);
  }
}

std::unique_ptr<HloModule> HloTestBase::CreateNewModule() {
  HloModuleConfig config;
  config.set_debug_options(legacy_flags::GetDebugOptionsFromFlags());
  return MakeUnique<HloModule>(TestName(), VersionedComputationHandle(),
                               config);
}

StatusOr<perftools::gputools::DeviceMemoryBase> HloTestBase::Execute(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        arguments,
    Shape* result_shape) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      backend_->compiler()->Compile(std::move(module), test_hlo_dumper_,
                                    backend_->default_stream_executor()));

  se::Stream stream(backend_->default_stream_executor());
  stream.Init();

  ExecutableRunOptions run_options;
  run_options.set_stream(&stream);
  run_options.set_allocator(backend_->memory_allocator());
  run_options.set_inter_op_thread_pool(backend_->inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(
      backend_->eigen_intra_op_thread_pool_device());

  HloExecutionProfile hlo_execution_profile;
  ServiceExecutableRunOptions service_run_options(
      run_options, backend_->StreamBorrower(),
      backend_->inter_op_thread_pool());
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase result,
      executable->ExecuteOnStream(&service_run_options, arguments,
                                  &hlo_execution_profile));
  TF_RET_CHECK(stream.BlockHostUntilDone());

  allocations_.push_back(result);

  *result_shape = executable->result_shape();

  if (ShapeUtil::IsTuple(*result_shape)) {
    // We must record element buffers of tuples as well to avoid leaks.
    DCHECK(!ShapeUtil::IsNestedTuple(*result_shape));
    TF_ASSIGN_OR_RETURN(
        std::vector<se::DeviceMemoryBase> element_buffers,
        backend_->transfer_manager()->ShallowCopyTupleFromDevice(
            backend_->default_stream_executor(), result, *result_shape));

    // A tuple may contain the same buffer in more than one element. Keep track
    // of the buffers already added to avoid duplicates in allocations_.
    std::set<void*> added_opaques;
    for (auto element_buffer : element_buffers) {
      if (added_opaques.count(element_buffer.opaque()) == 0) {
        CHECK(element_buffer.opaque() != nullptr);
        added_opaques.insert(element_buffer.opaque());
        allocations_.push_back(element_buffer);
      }
    }
  }

  return result;
}

se::DeviceMemoryBase HloTestBase::TransferToDevice(const Literal& literal) {
  // Allocate memory on the device using the stream executor.
  int64 allocation_size =
      backend_->transfer_manager()->GetByteSizeRequirement(literal.shape());
  se::DeviceMemoryBase allocation =
      backend_->default_stream_executor()->AllocateArray<uint8>(
          allocation_size);
  allocations_.push_back(allocation);

  TF_CHECK_OK(backend_->transfer_manager()->TransferLiteralToDevice(
      backend_->default_stream_executor(), literal, &allocation));

  return allocation;
}

std::unique_ptr<Literal> HloTestBase::TransferFromDevice(
    const Shape& shape, se::DeviceMemoryBase device_base) {
  auto literal = MakeUnique<Literal>();
  TF_CHECK_OK(backend_->transfer_manager()->TransferLiteralFromDevice(
      backend_->default_stream_executor(), device_base, shape, shape,
      literal.get()));
  return literal;
}

std::unique_ptr<Literal> HloTestBase::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  Shape result_shape;
  se::DeviceMemoryBase device_base =
      Execute(std::move(module), arguments, &result_shape).ValueOrDie();
  return TransferFromDevice(result_shape, device_base);
}

/* static */
string HloTestBase::TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

int ParseDebugOptionsFlagsAndRunTests(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}

}  // namespace xla
