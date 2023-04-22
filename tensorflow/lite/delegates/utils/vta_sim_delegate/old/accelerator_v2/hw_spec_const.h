/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef VTA_HW_SPEC_CONST_H_
#define VTA_HW_SPEC_CONST_H_

#define VTA_LOG_INP_WIDTH 3
#define VTA_LOG_WGT_WIDTH 3
#define VTA_LOG_ACC_WIDTH 5

#define VTA_LOG_BATCH 0
#define VTA_LOG_BLOCK_IN 4
#define VTA_LOG_BLOCK_OUT 4

#define VTA_LOG_UOP_BUFF_SIZE 15
#define VTA_LOG_INP_BUFF_SIZE 15
#define VTA_LOG_WGT_BUFF_SIZE 17
#define VTA_LOG_ACC_BUFF_SIZE 17


#define VTA_LOG_BUS_WIDTH 6
#define VTA_LOG_OUT_WIDTH 3

#define VTA_FETCH_INSN_COUNT_OFFSET 0x10
#define VTA_COMPUTE_DONE_WR_OFFSET 0x10






/*! Memory bus width */
#define VTA_BUS_WIDTH (1 << VTA_LOG_BUS_WIDTH)

/*! log2 of instruction data type width */
#define VTA_LOG_INS_WIDTH 7
/*! Instruction data type width */
#define VTA_INS_WIDTH (1 << VTA_LOG_INS_WIDTH)
/*! log2 of micro op data type width */
#define VTA_LOG_UOP_WIDTH 5
/*! Micro Op data type width */
#define VTA_UOP_WIDTH (1 << VTA_LOG_UOP_WIDTH)
/*! Weight data type width */
#define VTA_WGT_WIDTH (1 << VTA_LOG_WGT_WIDTH)
/*! Input data type width */
#define VTA_INP_WIDTH (1 << VTA_LOG_INP_WIDTH)
/*! Output data type width */
#define VTA_OUT_WIDTH (1 << VTA_LOG_OUT_WIDTH)
/*! Accumulator data type width */
#define VTA_ACC_WIDTH (1 << VTA_LOG_ACC_WIDTH)

/*! Batch size (corresponds to A in (A,B)x(B,C) mat mult)*/
#define VTA_BATCH (1 << VTA_LOG_BATCH)
/*! Blocking factor of inner most loop (corresponds to B in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_IN (1 << VTA_LOG_BLOCK_IN)
/*! Blocking factor of the outer loop (corresponds to C in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_OUT (1 << VTA_LOG_BLOCK_OUT)

/*! On-chip micro-op buffer size in B */
#define VTA_UOP_BUFF_SIZE (1 << VTA_LOG_UOP_BUFF_SIZE)
/*! On-chip weight buffer size in B */
#define VTA_WGT_BUFF_SIZE (1 << VTA_LOG_WGT_BUFF_SIZE)
/*! On-chip activation buffer size in B */
#define VTA_INP_BUFF_SIZE (1 << VTA_LOG_INP_BUFF_SIZE)
/*! On-chip accumulator buffer size in B */
#define VTA_ACC_BUFF_SIZE (1 << VTA_LOG_ACC_BUFF_SIZE)

/*! Input vector size in bits */
#define VTA_INP_MATRIX_WIDTH (VTA_INP_WIDTH * VTA_BATCH * VTA_BLOCK_IN)
/*! Weight vector size in bits */
#define VTA_WGT_MATRIX_WIDTH (VTA_WGT_WIDTH * VTA_BLOCK_OUT * VTA_BLOCK_IN)
/*! Accumulator vector size in bits */
#define VTA_ACC_MATRIX_WIDTH (VTA_ACC_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
/*! Output vector size in bits */
#define VTA_OUT_MATRIX_WIDTH (VTA_OUT_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)

/*! Ratio between input matrix size and axi width */
#define INP_MAT_AXI_RATIO (VTA_INP_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between weight matrix size and axi width */
#define WGT_MAT_AXI_RATIO (VTA_WGT_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between accumulator matrix size and axi width */
#define ACC_MAT_AXI_RATIO (VTA_ACC_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between output matrix size and axi width */
#define OUT_MAT_AXI_RATIO (VTA_OUT_MATRIX_WIDTH / VTA_BUS_WIDTH)

/*! Size of instruction buffer element in B */
#define VTA_INS_ELEM_BYTES (VTA_INS_WIDTH / 8)
/*! Size of uop buffer element in B*/
#define VTA_UOP_ELEM_BYTES (VTA_UOP_WIDTH / 8)
/*! Size of activation buffer element in B*/
#define VTA_INP_ELEM_BYTES (VTA_INP_MATRIX_WIDTH / 8)
/*! Size of weight buffer element in B*/
#define VTA_WGT_ELEM_BYTES (VTA_WGT_MATRIX_WIDTH / 8)
/*! Size of accumulator buffer element in B*/
#define VTA_ACC_ELEM_BYTES (VTA_ACC_MATRIX_WIDTH / 8)
/*! Size of output buffer element in B*/
#define VTA_OUT_ELEM_BYTES (VTA_OUT_MATRIX_WIDTH / 8)

/*! On-chip micro-op buffer depth */
#define VTA_UOP_BUFF_DEPTH (VTA_UOP_BUFF_SIZE / VTA_UOP_ELEM_BYTES)
/*! log2 of on-chip micro-op buffer depth */
#define VTA_LOG_UOP_BUFF_DEPTH (VTA_LOG_UOP_BUFF_SIZE - VTA_LOG_UOP_WIDTH + 3)
// ! \brief On-chip weight buffer depth
#define VTA_WGT_BUFF_DEPTH (VTA_WGT_BUFF_SIZE / VTA_WGT_ELEM_BYTES)
/*! log2 of weight micro-op buffer depth */
#define VTA_LOG_WGT_BUFF_DEPTH \
    (VTA_LOG_WGT_BUFF_SIZE - VTA_LOG_BLOCK_OUT - VTA_LOG_BLOCK_IN - VTA_LOG_WGT_WIDTH + 3 +1)
/*! On-chip activation buffer depth */
#define VTA_INP_BUFF_DEPTH (VTA_INP_BUFF_SIZE / VTA_INP_ELEM_BYTES)
/*! log2 of activation micro-op buffer depth */
#define VTA_LOG_INP_BUFF_DEPTH \
    (VTA_LOG_INP_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_IN - VTA_LOG_INP_WIDTH + 3)
/*! On-chip accumulator buffer depth */
#define VTA_ACC_BUFF_DEPTH (VTA_ACC_BUFF_SIZE / VTA_ACC_ELEM_BYTES)
/*! log2 of on-chip accumulator buffer depth */
#define VTA_LOG_ACC_BUFF_DEPTH \
    (VTA_LOG_ACC_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_OUT - VTA_LOG_ACC_WIDTH + 3)

/*! Instruction opcode field bitwidth */
#define VTA_OPCODE_BIT_WIDTH 3
/*! ALU opcode field bitwidth */
#define VTA_ALU_OPCODE_BIT_WIDTH 3

/*! Opcode: load encoding */
#define VTA_OPCODE_LOAD 0
/*! Opcode: store encoding */
#define VTA_OPCODE_STORE 1
/*! Opcode: GEMM encoding */
#define VTA_OPCODE_GEMM 2
/*! Opcode: finish encoding */
#define VTA_OPCODE_FINISH 3
/*! Opcode: ALU encoding */
#define VTA_OPCODE_ALU 4

/*! ALU opcode: unary min op */
#define VTA_ALU_OPCODE_MIN 0
/*! ALU opcode: unary max op */
#define VTA_ALU_OPCODE_MAX 1
/*! ALU opcode: binary add op */
#define VTA_ALU_OPCODE_ADD 2
/*! ALU opcode: shift right by immediate op */
#define VTA_ALU_OPCODE_SHR 3
/*! ALU opcode: mul */
#define VTA_ALU_OPCODE_MUL 4

/*! Memory type field bitwidth */
#define VTA_MEMOP_ID_BIT_WIDTH 3
/*! Load/Store Instruction: DRAM address width*/
#define VTA_MEMOP_SRAM_ADDR_BIT_WIDTH 16
/*! Load/Store Instruction: DRAM address width*/
#define VTA_MEMOP_DRAM_ADDR_BIT_WIDTH 32
/*! Load/Store Instruction: transfer size width*/
#define VTA_MEMOP_SIZE_BIT_WIDTH 16
/*! Load/Store Instruction: stride size width*/
#define VTA_MEMOP_STRIDE_BIT_WIDTH 16
/*! Load/Store Instruction: padding width*/
#define VTA_MEMOP_PAD_BIT_WIDTH 4
/*! Load/Store Instruction: padding value encoding width*/
#define VTA_MEMOP_PAD_VAL_BIT_WIDTH 2
/*! GEMM/ALU Instruction: loop max iter bits */
#define VTA_LOOP_ITER_WIDTH 14
/*! ALU Instruction: immediate bitwidth*/
#define VTA_ALUOP_IMM_BIT_WIDTH 16
/*! ALU Instruction: shift arg bitwidth*/
#define VTA_SHR_ARG_BIT_WIDTH (VTA_LOG_ACC_WIDTH)
/*! ALU Instruction: multiply arg bitwidth*/
#define VTA_MUL_ARG_BIT_WIDTH 8

/*! Mem ID constant: uop memory */
#define VTA_MEM_ID_UOP 0
/*! Mem ID constant: weight memory */
#define VTA_MEM_ID_WGT 1
/*! Mem ID constant: input memory */
#define VTA_MEM_ID_INP 2
/*! Mem ID constant: accumulator/bias memory */
#define VTA_MEM_ID_ACC 3
/*! Mem ID constant: output store buffer */
#define VTA_MEM_ID_OUT 4
/*! Mem ID constant: accumulator/bias memory (from int_8 buffer) */
#define VTA_MEM_ID_ACC_8BIT 5

#endif  // VTA_HW_SPEC_CONST_H_
