/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/original_tensor_summary_sequencer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/maker.h"
#include "riegeli/base/types.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_position.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"

namespace xla::numerics::comparison {

absl::StatusOr<std::unique_ptr<OriginalTensorSummarySequencer>>
OriginalTensorSummarySequencer::Create(const HloModule* original_module) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks;
  // We use a global counter for topological ranks across all computations. This
  // ensures that instructions in different computations are comparable, which
  // is useful to break ties for call-like instructions such as `while` that
  // invoke multiple computations. For example, instructions in while-condition
  // should be ranked before while-body.
  int64_t rank = 0;
  CHECK(original_module->has_entry_computation());
  absl::flat_hash_set<const HloComputation*> visited;
  std::function<void(HloComputation*)> visit;
  // Custom post-order traversal that visits while-condition before while-body.
  visit = [&](HloComputation* computation) {
    if (!visited.insert(computation).second) {
      return;
    }
    computation->ForEachInstructionPostOrder([&](HloInstruction* instruction) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        // Visit while-condition before while-body. We need this special
        // handling because `kBodyComputationIndex` is 0 and
        // `kConditionComputationIndex` is 1, which is inverse of the
        // topological order.
        visit(instruction->while_condition());
        visit(instruction->while_body());
      } else {
        for (HloComputation* called_computation :
             instruction->called_computations()) {
          visit(called_computation);
        }
      }

      topo_ranks[instruction->name()] = rank++;
    });
  };

  visit(original_module->entry_computation());

  return std::make_unique<OriginalTensorSummarySequencer>(
      std::move(topo_ranks));
}

struct KeyAndPos {
  AbsoluteScopedTensorKey key;
  riegeli::RecordPosition pos;
  int64_t output_rank;
};

absl::StatusOr<std::unique_ptr<IsOriginalTensorAlreadyRecoveredCallback>>
OriginalTensorSummarySequencer::Sequence(absl::string_view input_path,
                                         absl::string_view output_path) const {
  // The sequencing process involves reading summaries, sorting them according
  // to HLO data dependencies, and writing them to a new file in the sorted
  // order. Because the input file can be very large and may not fit into
  // memory, we cannot simply read all summaries, sort them in memory, and then
  // write them out.
  //
  // A naive approach would be to read only the keys and file offsets, sort the
  // keys in memory according to HLO data dependencies, and then seek to each
  // summary's offset in the input file to read and write it to the output file
  // in order. However, seeking to arbitrary locations in a file is inefficient
  // and can be very slow if the file is large.
  //
  // This implementation improves I/O performance by optimizing the read/write
  // process. We've observed that while the summaries in the input file are not
  // perfectly sorted, they are roughly in order. We can take advantage of this
  // by minimizing disk seeks. The approach is as follows:
  //
  // 1. Read all tensor keys and their file offsets from the input file.
  // 2. Sort keys based on HLO data dependencies to determine `output_rank`.
  // 3. Re-sort keys by file offset to allow for sequential reading of the input
  //    file. This avoids random disk seeks and is much more efficient.
  // 4. Read summaries sequentially and buffer them in memory. As summaries are
  //    read, check if the summary with the next `output_rank` is in the buffer.
  //    If it is, write it to the output file and remove it from the buffer.
  //    Repeat until all summaries are written.
  //
  // This method ensures that we read the input file efficiently with minimal
  // seeking while maintaining the correct output order, using a buffer to
  // handle cases where summaries are read out of HLO dependency order.

  // 1. Read all tensor keys and their file offsets from the input file.
  std::vector<KeyAndPos> keys_and_pos;
  riegeli::RecordReader reader(riegeli::Maker<riegeli::FdReader>(input_path));
  RecoveredTensorSummaryProto summary_proto;
  {
    std::optional<riegeli::Position> file_size = reader.Size();
    ProgressReporter read_progress_reporter("Reading summaries",
                                            file_size.value_or(0));
    while (true) {
      if (file_size.has_value()) {
        read_progress_reporter.Report(reader.pos().numeric());
      }
      const riegeli::RecordPosition pos = reader.pos();
      if (!reader.ReadRecord(summary_proto)) {
        if (!reader.ok()) {
          return reader.status();
        }
        break;  // End of file.
      }
      keys_and_pos.push_back({/*key=*/AbsoluteScopedTensorKey::FromProto(
                                  summary_proto.tensor_key()),
                              /*pos=*/pos});
      if (!file_size.has_value()) {
        read_progress_reporter.Report();
      }
    }
    if (!reader.Close()) {
      return reader.status();
    }
  }

  // 2. Sort keys based on HLO data dependencies to determine output rank.
  std::sort(keys_and_pos.begin(), keys_and_pos.end(),
            [this](const KeyAndPos& a, const KeyAndPos& b) {
              return CompareKeys(a.key, b.key);
            });
  for (int i = 0; i < keys_and_pos.size(); ++i) {
    keys_and_pos[i].output_rank = i;
  }

  // 3. Sort by pos to enable sequential read.
  std::sort(
      keys_and_pos.begin(), keys_and_pos.end(),
      [](const KeyAndPos& a, const KeyAndPos& b) { return a.pos < b.pos; });

  // 4. Read full summaries sequentially from input using offsets and write to
  // output in key-sorted order.
  riegeli::RecordReader reader2(riegeli::Maker<riegeli::FdReader>(input_path));
  riegeli::RecordWriter writer(
      riegeli::Maker<riegeli::FdWriter>(output_path),
      riegeli::RecordWriterBase::Options().set_transpose(true));

  absl::flat_hash_map<int64_t, RecoveredTensorSummaryProto> pending_summaries;
  int64_t next_rank_to_write = 0;
  {
    ProgressReporter write_progress_reporter("Writing summaries",
                                             keys_and_pos.size());
    for (const auto& key_and_pos : keys_and_pos) {
      if (!reader2.Seek(key_and_pos.pos)) {
        return reader2.status();
      }
      if (!reader2.ReadRecord(summary_proto)) {
        return absl::DataLossError(absl::StrCat(
            "Failed to read record at position ", key_and_pos.pos.ToString()));
      }
      pending_summaries.emplace(key_and_pos.output_rank, summary_proto);
      while (pending_summaries.contains(next_rank_to_write)) {
        if (!writer.WriteRecord(pending_summaries.at(next_rank_to_write))) {
          return writer.status();
        }
        pending_summaries.erase(next_rank_to_write);
        ++next_rank_to_write;
      }
      write_progress_reporter.Report();
    }
  }
  if (!reader2.Close()) {
    return reader2.status();
  }
  if (!writer.Close()) {
    return writer.status();
  }

  auto recovered_keys =
      std::make_shared<absl::flat_hash_set<AbsoluteScopedTensorKey>>();
  for (const auto& key_and_pos : keys_and_pos) {
    recovered_keys->insert(key_and_pos.key);
  }
  auto callback = [recovered_keys = std::move(recovered_keys)](
                      const AbsoluteScopedTensorKey& tensor_key) -> bool {
    return recovered_keys->contains(tensor_key);
  };
  return std::make_unique<IsOriginalTensorAlreadyRecoveredCallback>(callback);
}

int64_t OriginalTensorSummarySequencer::GetRank(
    absl::string_view instruction_name) const {
  auto it = topo_ranks_.find(instruction_name);
  if (it != topo_ranks_.end()) {
    return it->second;
  }
  return std::numeric_limits<int64_t>::max();
}

bool OriginalTensorSummarySequencer::CompareKeys(
    const AbsoluteScopedTensorKey& a, const AbsoluteScopedTensorKey& b) const {
  // Compare scopes
  const size_t len_a = a.scope_instructions.size();
  const size_t len_b = b.scope_instructions.size();
  const size_t min_len = std::min(len_a, len_b);
  for (size_t i = 0; i < min_len; ++i) {
    const auto& scope_a = a.scope_instructions[i];
    const auto& scope_b = b.scope_instructions[i];
    int64_t rank_a = GetRank(scope_a.instruction_name);
    int64_t rank_b = GetRank(scope_b.instruction_name);
    if (rank_a != rank_b) {
      return rank_a < rank_b;
    }
    if (scope_a.iteration_index != scope_b.iteration_index) {
      return scope_a.iteration_index < scope_b.iteration_index;
    }
  }

  if (len_a != len_b) {
    if (len_a < len_b) {
      // a scope instructions is prefix of b scope instructions.
      // E.g. a = scope/T_a, b = scope/S_b/T_b.
      // T_a and S_b are in the same computation.
      // If T_a == S_b, then T_a is a call instruction and T_b is in a called
      // computation. The summaries in b should appear before a.
      int64_t rank_a = GetRank(a.tensor_key.instruction_name);
      int64_t rank_b_scope =
          GetRank(b.scope_instructions[min_len].instruction_name);
      if (rank_a == rank_b_scope) {
        // a is a call instruction and b is in a called computation.
        // So the summaries in b should appear before a.
        return false;
      }
      return rank_a < rank_b_scope;
    }
    // b scope instructions is prefix of a scope instructions.
    // E.g. b = scope/T_b, a = scope/S_a/T_a
    // T_b and S_a are in the same computation.
    // If T_b == S_a, then T_b is a call instruction and T_a is in a called
    // computation. The summaries in a should appear before b.
    int64_t rank_a_scope =
        GetRank(a.scope_instructions[min_len].instruction_name);
    int64_t rank_b = GetRank(b.tensor_key.instruction_name);
    if (rank_a_scope == rank_b) {
      // b is a call instruction and a is in a called computation.
      // So the summaries in a should appear before b.
      return true;
    }
    return rank_a_scope < rank_b;
  }

  // Compare tensor keys
  int64_t arank = GetRank(a.tensor_key.instruction_name);
  int64_t brank = GetRank(b.tensor_key.instruction_name);
  if (arank != brank) {
    return arank < brank;
  }

  // Compare shape indices
  return a.tensor_key.shape_index < b.tensor_key.shape_index;
}

}  // namespace xla::numerics::comparison
