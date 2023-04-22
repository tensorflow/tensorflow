#ifndef VTA_GEN_INS_H
#define VTA_GEN_INS_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../hw_spec.h"

#include <bitset>
#include <systemc.h>
#include "../hls_bus_if.h"
#include "mm_helper.h"

#include "systemc_integrator.h"


using namespace std;
typedef uint32_t uop_CT;
typedef int8_t wgt_CT;
typedef int8_t inp_CT;
typedef int8_t out_CT;
typedef int32_t acc_CT;
uint32_t globalSeed;

void printParameters() {
  // Some debugging code
  printf("Size of VTAInsn: %d\n", sizeof(VTAGenericInsn));
  printf("Size of VTAUop: %d\n", sizeof(VTAUop));
  printf("VTA_UOP_BUFF_DEPTH: %d\n", VTA_UOP_BUFF_DEPTH);
  printf("VTA_LOG_UOP_BUFF_DEPTH: %d\n", VTA_LOG_UOP_BUFF_DEPTH);
  printf("VTA_WGT_BUFF_DEPTH: %d\n", VTA_WGT_BUFF_DEPTH);
  printf("VTA_LOG_WGT_BUFF_DEPTH: %d\n", VTA_LOG_WGT_BUFF_DEPTH);
  printf("VTA_INP_BUFF_DEPTH: %d\n", VTA_INP_BUFF_DEPTH);
  printf("VTA_LOG_INP_BUFF_DEPTH: %d\n", VTA_LOG_INP_BUFF_DEPTH);
  printf("VTA_ACC_BUFF_DEPTH: %d\n", VTA_ACC_BUFF_DEPTH);
  printf("VTA_LOG_ACC_BUFF_DEPTH: %d\n", VTA_LOG_ACC_BUFF_DEPTH);
  printf("VTA_WGT_WORDS: %d\n", VTA_WGT_BUFF_DEPTH*VTA_BLOCK_IN*VTA_BLOCK_OUT);
  printf("VTA_INP_WORDS: %d\n", VTA_INP_BUFF_DEPTH*VTA_BLOCK_IN);
  printf("VTA_ACC_WORDS: %d\n", VTA_ACC_BUFF_DEPTH*VTA_BLOCK_OUT);
  printf("VTA_INS_ELEM_BYTES: %d\n", VTA_INS_ELEM_BYTES);
  printf("VTA_UOP_ELEM_BYTES: %d\n", VTA_UOP_ELEM_BYTES);
  printf("VTA_INP_ELEM_BYTES: %d\n", VTA_INP_ELEM_BYTES);
  printf("VTA_WGT_ELEM_BYTES: %d\n", VTA_WGT_ELEM_BYTES);
  printf("VTA_ACC_ELEM_BYTES: %d\n", VTA_ACC_ELEM_BYTES);
  printf("VTA_BLOCK_IN: %d\n", VTA_BLOCK_IN);
  printf("VTA_BLOCK_OUT: %d\n", VTA_BLOCK_OUT);
}

const char* getOpcodeString(int opcode, bool use_imm) {
  // Returns string name
  if (opcode == VTA_ALU_OPCODE_MIN) {
    if (use_imm) {
      return "min imm";
    } else {
      return "min";
    }
  } else if (opcode == VTA_ALU_OPCODE_MAX) {
    if (use_imm) {
      return "max imm";
    } else {
      return "max";
    }
  } else if (opcode == VTA_ALU_OPCODE_ADD) {
    if (use_imm) {
      return "add imm";
    } else {
      return "add";
    }
  } else if (opcode == VTA_ALU_OPCODE_SHR) {
    return "shr";
  }
  // else if (opcode == VTA_ALU_OPCODE_MUL) {
  //   return "mul";
  // }
  return "unknown op";
}

void printInstruction(int num_insn, VTAGenericInsn *insns) {
  // Keep tabs on dependence queues
  int l2g_queue = 0;
  int g2l_queue = 0;
  int s2g_queue = 0;
  int g2s_queue = 0;
  // Converter
  union VTAInsn c;
  // Iterate over all instructions
  printf("DEBUG - There are %u instructions\n", num_insn);
  for (int i = 0; i < num_insn; i++) {
    // Fetch instruction and decode opcode
    c.generic = insns[i];
    printf("DEBUG - INSTRUCTION %u: ", i);
    if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
      // Print instruction field information
      if (c.mem.opcode == VTA_OPCODE_LOAD) {
        printf("LOAD ");
        if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
        if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
        if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
        if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
      }
      if (c.mem.opcode == VTA_OPCODE_STORE) {
        printf("STORE ACC\n");
      }
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\tDRAM: 0x%08x, SRAM:0x%04x\n",
             static_cast<int>(c.mem.dram_base),
             static_cast<int>(c.mem.sram_base));
      printf("\ty: size=%d, pad=[%d, %d]\n",
             static_cast<int>(c.mem.y_size),
             static_cast<int>(c.mem.y_pad_0),
             static_cast<int>(c.mem.y_pad_1));
      printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n",
             static_cast<int>(c.mem.x_size),
             static_cast<int>(c.mem.x_stride),
             static_cast<int>(c.mem.x_pad_0),
             static_cast<int>(c.mem.x_pad_1));
      if (c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.pop_prev_dep) g2s_queue--;
        if (c.mem.push_prev_dep) s2g_queue++;
      } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
        (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
        if (c.mem.pop_next_dep) g2l_queue--;
        if (c.mem.push_next_dep) l2g_queue++;
      } else {
        if (c.mem.pop_prev_dep) l2g_queue--;
        if (c.mem.push_prev_dep) g2l_queue++;
        if (c.mem.pop_next_dep) s2g_queue--;
        if (c.mem.push_next_dep) g2s_queue++;
      }
    } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
      // Print instruction field information
      printf("GEMM\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\trange (%d, %d)\n",
             static_cast<int>(c.gemm.uop_bgn),
             static_cast<int>(c.gemm.uop_end));
      printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
      printf("\touter loop - iter: %d, acc: %d, inp: %d, wgt: %d\n",
             static_cast<int>(c.gemm.iter_out),
             static_cast<int>(c.gemm.dst_factor_out),
             static_cast<int>(c.gemm.src_factor_out),
             static_cast<int>(c.gemm.wgt_factor_out));
      printf("\tinner loop - iter: %d, acc: %d, inp: %d, wgt: %d\n",
             static_cast<int>(c.gemm.iter_in),
             static_cast<int>(c.gemm.dst_factor_in),
             static_cast<int>(c.gemm.src_factor_in),
             static_cast<int>(c.gemm.wgt_factor_in));
      if (c.gemm.pop_prev_dep) l2g_queue--;
      if (c.gemm.push_prev_dep) g2l_queue++;
      if (c.gemm.pop_next_dep) s2g_queue--;
      if (c.gemm.push_next_dep) g2s_queue++;
    } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
      printf("FINISH\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      if (c.gemm.pop_prev_dep) l2g_queue--;
      if (c.gemm.push_prev_dep) g2l_queue++;
      if (c.gemm.pop_next_dep) s2g_queue--;
      if (c.gemm.push_next_dep) g2s_queue++;
    } else if (c.mem.opcode == VTA_OPCODE_ALU) {
      // Print instruction field information
      printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
      printf("\trange (%d, %d)\n",
             static_cast<int>(c.alu.uop_bgn),
             static_cast<int>(c.alu.uop_end));
      printf("\touter loop - iter: %d, dst: %d, src: %d\n",
             static_cast<int>(c.alu.iter_out),
             static_cast<int>(c.alu.dst_factor_out),
             static_cast<int>(c.alu.src_factor_out));
      printf("\tinner loop - iter: %d, dst: %d, src: %d\n",
             static_cast<int>(c.alu.iter_in),
             static_cast<int>(c.alu.dst_factor_in),
             static_cast<int>(c.alu.src_factor_in));
      if (c.alu.pop_prev_dep) l2g_queue--;
      if (c.alu.push_prev_dep) g2l_queue++;
      if (c.alu.pop_next_dep) s2g_queue--;
      if (c.alu.push_next_dep) g2s_queue++;
    }
  }
  printf("DEBUG - l2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
  printf("DEBUG - s2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
}

// Helper function: Print micro-ops status
void printMicroOp(int num_uop, VTAUop *uops) {
  // Iterate over all micro ops
  printf("DEBUG - There are %u micro-ops\n", num_uop);
  for (int i = 0; i < num_uop; i++) {
    // Read micro-op
    printf("DEBUG - UOP %u: ", i);
    printf("acc=%u, inp= %u, wgt=%u\n", uops[i].dst_idx, uops[i].src_idx, uops[i].wgt_idx);
  }
}

void * allocBuffer(size_t num_bytes) {
  return malloc(num_bytes);
}

template <typename T>
T ** allocInit2dArray(int rows, int cols, int value) {
	srand(10);
  // Allocate
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  // Init
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      array[i][j] = static_cast<T>(value);
    }
  }
  return array;
}

template <typename T>
T ** allocInit2dArray(int rows, int cols) {
	srand(10);
  // Allocate
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  // Init
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      array[i][j] = static_cast<T>(1);
    }
  }
  return array;
}


template <typename T>
T ** alloc2dArray(int rows, int cols) {
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  return array;
}

template <typename T>
void free2dArray(T **array, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}

template <typename T>
void free3dArray(T *** array, int rows, int cols, int depth) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

void freeBuffer(void * buffer) {
  return free(buffer);
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void packBuffer(DST_T *dst, SRC_T **src, int y_size, int x_size, int y_block, int x_block) {
  assert((SRC_T_WIDTH * x_block * y_block) % DST_T_WIDTH  == 0);
  assert(DST_T_WIDTH <= 64);
  int buffer_idx = 0;
  int ratio = DST_T_WIDTH / SRC_T_WIDTH;
  long long int mask = (1ULL << SRC_T_WIDTH) - 1;
  DST_T tmp = 0;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block;
          tmp |= (src[i * y_block + k][j * x_block + l] & mask) << ((block_idx % ratio) * SRC_T_WIDTH);
          // When tmp is packed, write to destination array
          if (block_idx % ratio == ratio - 1) {
            dst[buffer_idx++] = tmp;
            tmp = 0;
          }
        }
      }
    }
  }
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBuffer(DST_T **dst, SRC_T *src, int y_size, int x_size, int y_block, int x_block) {
  assert((DST_T_WIDTH * x_block * y_block) % SRC_T_WIDTH == 0);
  int buffer_idx = 0;
  long long int mask = (1ULL << DST_T_WIDTH) - 1;
  int ratio = SRC_T_WIDTH / DST_T_WIDTH;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block;
          dst[i * y_block + k][j * x_block + l] = (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}



VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
    int y_size, int x_size, int x_stride, int y_pad, int x_pad, int pop_prev_dep, int pop_next_dep,
    int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = y_size;
  insn.x_size = x_size;
  insn.x_stride = x_stride;
  insn.y_pad_0 = y_pad;
  insn.y_pad_1 = y_pad;
  insn.x_pad_0 = x_pad;
  insn.x_pad_1 = x_pad;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
    int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = 1;
  insn.x_size = size;
  insn.x_stride = size;
  insn.y_pad_0 = 0;
  insn.y_pad_1 = 0;
  insn.x_pad_0 = 0;
  insn.x_pad_1 = 0;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
    bool uop_compression, int pop_prev_dep, int pop_next_dep, int push_prev_dep,
    int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_GEMM;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.reset_reg = false;
  if (!uop_compression) {
    insn.uop_bgn = uop_offset;
    insn.uop_end = uop_offset + batch * in_feat * out_feat;
    insn.iter_out = 1;
    insn.iter_in = 1;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 0;
    insn.wgt_factor_out = 0;
    insn.dst_factor_in = 0;
    insn.src_factor_in = 0;
    insn.wgt_factor_in = 0;
  } else {
    insn.uop_bgn = uop_offset;
    insn.uop_end = uop_offset + batch;
    insn.iter_out = in_feat;
    insn.iter_in = out_feat;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 1;
    insn.wgt_factor_out = 1;
    insn.dst_factor_in = 1;
    insn.src_factor_in = 0;
    insn.wgt_factor_in = in_feat;
  }
  converter.gemm = insn;
  return converter.generic;
}

VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_FINISH;
  insn.pop_prev_dep = pop_prev;
  insn.pop_next_dep = pop_next;
  insn.push_prev_dep = 0;
  insn.push_next_dep = 0;
  insn.reset_reg = false;
  insn.uop_bgn = 0;
  insn.uop_end = 0;
  insn.iter_out = 0;
  insn.iter_in = 0;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}


VTAUop * getGEMMUops(int batch, int in_feat, int out_feat, bool uop_compression,
    bool multi_threaded) {
  // Derive the total uop size
  int uop_size = (uop_compression) ? batch : batch * in_feat * out_feat;
  if (multi_threaded) uop_size *= 2;

  // Allocate buffer
  VTAUop *uop_buf = static_cast<VTAUop *>(malloc(sizeof(VTAUop) * uop_size));

  if (!uop_compression) {
    int uop_idx = 0;
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < in_feat; j++) {
        for (int k = 0; k < out_feat; k++) {
          uop_buf[uop_idx].dst_idx = i * out_feat + k;
          uop_buf[uop_idx].src_idx = i * in_feat + j;
          uop_buf[uop_idx].wgt_idx = k * in_feat + j;
          uop_idx++;
        }
      }
    }
  } else {
    for (int i = 0; i < batch; i++) {
      uop_buf[i].dst_idx = i * out_feat;
      uop_buf[i].src_idx = i * in_feat;
      uop_buf[i].wgt_idx = 0;
    }
  }

  if (multi_threaded) {
    if (!uop_compression) {
      int uop_idx = uop_size / 2;
      for (int i = 0; i < batch; i++) {
        for (int j = 0; j < in_feat; j++) {
          for (int k = 0; k < out_feat; k++) {
            uop_buf[uop_idx].dst_idx = i * out_feat + k;
            uop_buf[uop_idx].src_idx = batch * in_feat + i * in_feat + j;
            uop_buf[uop_idx].wgt_idx = out_feat * in_feat + k * in_feat + j;
            uop_idx++;
          }
        }
      }
    } else {
      for (int i = 0; i < batch; i++) {
        uop_buf[batch+i].dst_idx = i * out_feat;
        uop_buf[batch+i].src_idx = batch * in_feat + i * in_feat;
        uop_buf[batch+i].wgt_idx = out_feat * in_feat;
      }
    }
  }

  return uop_buf;
}







int blocked_gemm_test_tflite(
    systemC_sigs* scs,
		int batch, int in_channels, int out_channels, int block, bool uop_compression,int virtual_threads,
    uint32_t *inputs,
    uint32_t *weights,
    uint32_t *biases,
    uint32_t *outputs,
    uint32_t *crf,
    uint32_t *crx,
    bool save,
    int ra
    ) {

  // Some assertions
  assert(block % VTA_BLOCK_IN == 0);
  assert(block % VTA_BLOCK_OUT == 0);
  assert(block % VTA_BATCH == 0);
  assert(in_channels % block == 0);
  assert(out_channels % block == 0);
  assert(batch % block == 0);



  // Input/output channels
  int in_feat = in_channels;
  int out_feat = out_channels;
  // Derive number of elements that need to be loaded/stored
  int ins_size = batch / block * out_feat / block * (2 + in_feat / block * 3) + 2;
  int uop_size = uop_compression ?
      block / VTA_BATCH * virtual_threads :
      block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT * virtual_threads;
  int inp_size = batch / VTA_BATCH * in_feat / VTA_BLOCK_IN;
  int wgt_size = in_feat / VTA_BLOCK_IN * out_feat / VTA_BLOCK_OUT;
  int out_size = batch / VTA_BATCH * out_feat / VTA_BLOCK_OUT;
  // Blocked buffer sizes (in terms of elements)
  int inp_block_size = block / VTA_BATCH * block / VTA_BLOCK_IN;
  int wgt_block_size = block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT;
  int out_block_size = block / VTA_BATCH * block / VTA_BLOCK_OUT;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= VTA_UOP_BUFF_DEPTH);
  assert(inp_block_size <= VTA_INP_BUFF_DEPTH);
  assert(wgt_block_size <= VTA_WGT_BUFF_DEPTH);
  assert(out_block_size <= VTA_ACC_BUFF_DEPTH);

  // Initialize instruction buffer
  VTAGenericInsn *insn_buf =
      static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * ins_size));
  int insn_idx = 0;


  int crx_inc_len = 1 * out_channels/ sizeof(unsigned long long);

  // Load uops
  insn_buf[insn_idx++] = get1DLoadStoreInsn(VTA_OPCODE_LOAD,
                                            VTA_MEM_ID_UOP,
                                            0,
                                            0,
                                            uop_size,
                                            0,
                                            0,
                                            0,
                                            0);
  // Iterate over batch blocks

  // int dram_inc= crx_inc_len / (out_feat/block * batch/block);

  int sram_inc = (batch/block);
  // int dram_inc= out_feat/block * (batch/block);
  int dram_inc= block/8;
  for (int i = 0; i < batch; i += block) {
    int dram_jump=0;
    // Iterate over output channel blocks
    for (int j = 0; j < out_feat; j += block) {
      // Load bias block (pop next if not first, push prev)
      insn_buf[insn_idx++] = get2DLoadStoreInsn(
          VTA_OPCODE_LOAD,                                    // opcode
          VTA_MEM_ID_ACC,                                     // type
          dram_jump,                                          // sram offset
          (i / VTA_BATCH * out_feat + j) / VTA_BLOCK_OUT,     // dram offset
          block / VTA_BATCH,                                  // y size
          block / VTA_BLOCK_OUT,                              // x size
          out_feat / VTA_BLOCK_OUT,                           // x stride
          0,                                                  // y pad
          0,                                                  // x pad
          0,                                                  // pop prev dep
          (i > 0 || j > 0),                                   // pop next dep
          (virtual_threads == 1),                             // push prev dep
          0);                                                 // push next dep
      // Iterate over input channel blocks
      for (int k = 0; k < in_feat; k += block * virtual_threads) {
        for (int l = 0; l < block * virtual_threads; l += block) {
          // Derive dependence flags
          bool pop = (virtual_threads == 1) ?
              1 :
              (i > 0 || j > 0 || k > 0 || l > 0) && (k + l != block * virtual_threads - block);
          bool push_prev = (virtual_threads == 1) ?
              ((k + l) != in_feat - block) :
              ((k + l) != in_feat - virtual_threads * block) &&
              (
                  (k + l != in_feat - block) ||
                  (j != out_feat - block) ||
                  (i != batch - block));
          bool push_next = (k + l == in_feat - block);
          // Load weight block (pop next)
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_WGT,                                 // type
              l / VTA_BLOCK_IN * block / VTA_BLOCK_OUT,       // sram offset
              (j / VTA_BLOCK_OUT * in_feat + k + l) / VTA_BLOCK_IN,  // dram offset
              block / VTA_BLOCK_OUT,                          // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN,                         // x stride
              0,                                              // y pad
              0,                                              // x pad
              0,                                              // pop prev dep
              pop,                                            // pop next dep
              0,                                              // push prev dep
              0);                                             // push next dep
          // Load input block (push next)
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_INP,                                 // type
              l / VTA_BLOCK_IN * block / VTA_BATCH,           // sram offset
              (i / VTA_BATCH * in_feat + k + l) / VTA_BLOCK_IN,  // dram offset
              block / VTA_BATCH,                              // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN,                         // x stride
              0,                                              // y pad
              0,                                              // x pad
              0,                                              // pop prev dep
              0,                                              // pop next dep
              0,                                              // push prev dep
              1);                                             // push next dep
          // Perform GEMM (pop prev, push prev if not last, push next if last)
          insn_buf[insn_idx++] = getGEMMInsn(
              l / block * uop_size / virtual_threads,         // uop offset
              block / VTA_BATCH,                              // batch
              block / VTA_BLOCK_IN,                           // in_feat
              block / VTA_BLOCK_OUT,                          // out_feat
              uop_compression,                                // uop_compression
              1,                                              // pop_prev_dep
              0,                                              // pop_next_dep
              push_prev,                                      // push prev dep
              push_next);                                     // push_next_dep
        }
      }
      // Store output block (pop prev, push prev if not last)
      insn_buf[insn_idx++] = get2DLoadStoreInsn(
          VTA_OPCODE_STORE,                                   // opcode
          VTA_MEM_ID_OUT,                                     // type
          0,                                                  // sram offset
          (i / VTA_BATCH * out_feat + j) / VTA_BLOCK_OUT,     // dram offset
          block / VTA_BATCH,                                  // y size
          block / VTA_BLOCK_OUT,                              // x size
          out_feat / VTA_BLOCK_OUT,                           // x stride
          0,                                                  // y pad
          0,                                                  // x pad
          1,                                                  // pop prev dep
          0,                                                  // pop next dep
          1,                                                  // pop prev dep
          0);                                                 // push next dep
      dram_jump += dram_inc;
    }

  }
  // Finish
  insn_buf[insn_idx++] = getFinishInsn(0, 1);

  // Prepare the uop buffer
  VTAUop * uop_buf = getGEMMUops(
      block / VTA_BATCH,
      block / VTA_BLOCK_IN,
      block / VTA_BLOCK_OUT,
      uop_compression,
      virtual_threads > 1);


  //Save Data from Here
  int insns_len =  sizeof(VTAGenericInsn) * ins_size/ sizeof(unsigned long long);
  unsigned long long* insn_set = (unsigned long long*)insn_buf;
  scs->insns_mem.burst_write(0, insns_len, insn_set);
  scs->sig_insn_count.write(ins_size);

  int uops_len = sizeof(VTAUop) * uop_size / sizeof(unsigned int);
  unsigned int* uops_set = (unsigned int*)uop_buf;
  scs->uops_mem.burst_write(0, uops_len, uops_set);
  // for (int i=0;i<uops_len;i++) cout << uops_set[i] << endl;


  int input_len = VTA_INP_ELEM_BYTES * inp_size/ sizeof(unsigned long long);
  unsigned long long* input_set = (unsigned long long*)inputs;
  scs->data_mem.burst_write(0, input_len, input_set);


  int weight_len = VTA_WGT_ELEM_BYTES * wgt_size/ sizeof(unsigned long long);
  unsigned int weight_addr = input_len;
  unsigned long long* weight_set = (unsigned long long*)weights;
  scs->data_mem.burst_write(weight_addr, weight_len, weight_set);
  scs->sig_weight_addr.write(weight_addr);


  int bias_len = VTA_ACC_ELEM_BYTES * out_size/ sizeof(unsigned long long);
  unsigned int bias_addr = weight_addr+weight_len;
  unsigned long long* bias_set = (unsigned long long*)biases;
  scs->data_mem.burst_write(bias_addr, bias_len, bias_set);
  scs->sig_bias_addr.write(bias_addr);


  int crf_size = out_channels;
  int crf_len = 4 * crf_size/ sizeof(unsigned long long);
  unsigned int crf_addr = bias_addr+bias_len;
  unsigned long long* crf_set = (unsigned long long*)crf;
  scs->data_mem.burst_write(crf_addr, crf_len, crf_set);
  scs->sig_crf_addr.write(crf_addr);


  int crx_size = out_channels;
  int crx_len = 1 * crx_size/ sizeof(unsigned long long);
  unsigned int crx_addr = crf_addr+crf_len;
  unsigned long long* crx_set = (unsigned long long*)crx;
  scs->data_mem.burst_write(crx_addr, crx_len, crx_set);
  scs->sig_crx_addr.write(crx_addr);


  int out_len = VTA_OUT_ELEM_BYTES * out_size/ sizeof(unsigned long long);
  unsigned int out_addr = crx_addr+crx_len;
  scs->sig_output_addr.write(out_addr);
  
  // int out_len = VTA_OUT_ELEM_BYTES * out_size/ sizeof(unsigned long long);
  // unsigned int out_addr = bias_addr+bias_len;
  // scs->sig_output_addr.write(out_addr);


  // printInstruction(ins_size, insn_buf);
  // printMicroOp(uop_size, uop_buf);
  // printf("=====================================================================================\n");
  printf("INFO - Blocked GEMM test:isns=%d, batch=%d, in_channels=%d, out_channels=%d, block=%d, uop_comp=%d, vt=%d\n",
         insn_idx,batch, in_channels, out_channels, block, uop_compression, virtual_threads);

  sc_start();

  unsigned long long* out_set = (unsigned long long*)outputs;
  scs->data_mem.burst_read(out_addr, out_len, out_set);

  int total_indata_len = input_len+weight_len+bias_len+crf_len+crx_len;
  unsigned long long* input_to_file = new unsigned long long[total_indata_len];
  scs->data_mem.burst_read(0, total_indata_len, input_to_file);

  // if(save){
  //   ofstream myfile;
  //   myfile.open("a_Vta/vta_IO_metadata.txt");
  //   myfile << ra << endl;
  //   myfile << ins_size << endl;
  //   myfile << 0 << endl << insns_len << endl;
  //   myfile << 0 << endl << uops_len << endl;
  //   myfile << 0 << endl << input_len << endl;
  //   myfile << weight_addr << endl << weight_len << endl;
  //   myfile << bias_addr << endl << bias_len << endl;
  //   myfile << crf_addr << endl << crf_len << endl;
  //   myfile << crx_addr << endl << crx_len << endl;
  //   myfile << out_addr << endl << out_len << endl;
  //   myfile.close();

  //   ofstream uopsfile;
  //   uopsfile.open("a_Vta/vta_uops.txt");
  //   for(int i=0;i<uops_len;i++)uopsfile << uops_set[i] << endl;
  //   uopsfile.close();


  //   ofstream insnfile;
  //   insnfile.open("a_Vta/vta_insns.txt");
  //   for(int i=0;i<insns_len;i++)insnfile << insn_set[i] << endl;
  //   insnfile.close();


  //   ofstream datafile;
  //   datafile.open("a_Vta/vta_indata.txt");
  //   for(int i=0;i<total_indata_len;i++)datafile << input_to_file[i] << endl;
  //   datafile.close();


  //   ofstream outfile;
  //   outfile.open("a_Vta/vta_outdata.txt");
  //   for(int i=0;i<out_len;i++)outfile << out_set[i] << endl;
  //   outfile.close();
  // }

  // Save to Here

  // Free all allocated arrays
  freeBuffer(insn_buf);
  freeBuffer(uop_buf);
}



#endif // VTA_GEN_INS_H