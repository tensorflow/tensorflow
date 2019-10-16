/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
MemoryDump ReadDumpFile(const string& fname) {
  Status status;
  uint64 file_size = 0;
  status = Env::Default()->GetFileSize(fname, &file_size);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get size of " << fname;
    exit(1);
  }
  std::unique_ptr<RandomAccessFile> file;
  status = Env::Default()->NewRandomAccessFile(fname, &file);
  if (!status.ok()) {
    LOG(ERROR) << "read from file " << fname << " failed " << status;
  }
  std::unique_ptr<char> buffer(static_cast<char*>(malloc(file_size + 1)));
  DCHECK(buffer.get());
  StringPiece contents(buffer.get(), file_size);
  status = file->Read(0, file_size, &contents, buffer.get());
  if (!status.ok()) {
    LOG(ERROR) << "read from file " << fname << " failed " << status;
  }
  MemoryDump md;
  md.ParseFromString(string(contents));
  return md;
}

void PrintChunk(const MemChunk& mc, const uint64 ac_offset, bool freed_at,
                const int64 total_bytes, int64* cumulative_bytes) {
  // A size class corresponding approximately to log base 100.
  int size_class = floor(0.5 * log10(static_cast<double>(mc.size())));
  *cumulative_bytes += mc.size();
  printf("  %c %d %p bin=%d bytes=%llu %3.1f%%", mc.in_use() ? 'U' : 'F',
         size_class, reinterpret_cast<const void*>(mc.address()), mc.bin(),
         mc.size(),
         100 * (*cumulative_bytes / static_cast<float>(total_bytes)));
  if (freed_at) {
    printf(" freed_at=%llu", mc.freed_at_count());
  }
  if (ac_offset > 0) {
    printf(" age=%llu", ac_offset - mc.action_count());
  } else {
    printf(" ac=%llu", mc.action_count());
  }
  // step_ids are random, so save space by showing only low 16 bits.
  printf(" step=%x op=%s\n", static_cast<uint>(0xFFFF & mc.step_id()),
         mc.op_name().c_str());
}

void PrintSummary(const MemoryDump& md) {
  printf("MemoryMap for allocator %s\n", md.allocator_name().c_str());
  for (auto& it : md.bin_summary()) {
    printf(
        "   Bin %2d total bytes=%10lld \tin use=%10lld \ttotal_chunks=%6lld "
        "\tin_use=%6lld\n",
        it.bin(), it.total_bytes_in_bin(), it.total_bytes_in_use(),
        it.total_chunks_in_bin(), it.total_chunks_in_use());
  }
}

void PrintSortedChunks(
    const MemoryDump& md,
    std::function<bool(const MemChunk*, const MemChunk*)> compare, bool by_age,
    bool freed_at, bool by_addr) {
  std::vector<const MemChunk*> chunks;
  chunks.reserve(md.chunk_size());
  int64 total_bytes = 0;
  int64 cumulative_bytes = 0;
  uint64 max_action_count = 0;
  for (auto& it : md.chunk()) {
    chunks.push_back(&it);
    total_bytes += it.size();
    if (by_age && it.action_count() > max_action_count) {
      max_action_count = it.action_count();
    }
  }
  sort(chunks.begin(), chunks.end(), compare);
  uint64 last_end = 0;
  for (int i = 0; i < chunks.size(); ++i) {
    const MemChunk* c = chunks[i];
    if (by_addr && i > 0 && last_end != c->address()) {
      printf("  empty range from %p to %p  (%lld)\n",
             reinterpret_cast<const void*>(last_end),
             reinterpret_cast<const void*>(c->address()),
             (c->address() - last_end));
    }
    PrintChunk(*c, max_action_count, freed_at, total_bytes, &cumulative_bytes);
    last_end = c->address() + c->size();
  }
}

void PrintChunksByAddress(const MemoryDump& md, bool by_age, bool freed_at) {
  printf("---------------Chunks by address:--------------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->address() < b->address();
      },
      by_age, freed_at, true /*by_addr*/);
}

void PrintChunksByActionCount(const MemoryDump& md, bool by_age,
                              bool freed_at) {
  printf("------------Chunks by last action count:----------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->action_count() < b->action_count();
      },
      by_age, freed_at, false /*by_addr*/);
}

void PrintChunksBySize(const MemoryDump& md, bool by_age, bool freed_at) {
  printf("------------Chunks by size:----------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->size() > b->size();
      },
      by_age, freed_at, false /*by_addr*/);
}
}  // namespace tensorflow

int main(int argc, char** argv) {
  std::string path = "";
  bool by_addr = false;
  bool by_ac = false;
  bool by_size = false;
  bool by_age = true;
  bool freed_at = false;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("path", &path,
                       "Path of GPU BFCAllocator memory dump file"),
      tensorflow::Flag("by_addr", &by_addr,
                       "Whether to print Chunks by memory address"),
      tensorflow::Flag("by_ac", &by_ac,
                       "Whether to print Chunks by action count"),
      tensorflow::Flag("by_size", &by_size,
                       "Whether to print Chunks by decreasing size"),
      tensorflow::Flag("by_age", &by_age,
                       "If true, replace absolute action count with age "
                       "(max_action_count - action_count) in display."),
      tensorflow::Flag("freed_at", &freed_at,
                       "Whether to display the freed_at value (only relevant "
                       "with timestamped allocator)."),
  };
  bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || path.empty()) {
    std::cerr << tensorflow::Flags::Usage(argv[0], flag_list);
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::MemoryDump md = tensorflow::ReadDumpFile(path);
  tensorflow::PrintSummary(md);
  if (by_addr) tensorflow::PrintChunksByAddress(md, by_age, freed_at);
  if (by_ac) tensorflow::PrintChunksByActionCount(md, by_age, freed_at);
  if (by_size) tensorflow::PrintChunksBySize(md, by_age, freed_at);
}
