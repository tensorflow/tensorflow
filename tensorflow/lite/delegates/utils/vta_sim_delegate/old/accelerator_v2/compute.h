#include "vta.h"

#ifndef __SYNTHESIS__
template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
void ACCNAME::write_tensor(
  IDX_T idx,
  NARROW_T src[Y_DIM][X_DIM],
  WIDE_T *dst) {
#pragma HLS INLINE

	int stride = NARROW_W * Y_DIM * X_DIM / WIDE_W;
  for (int p = 0; p < NARROW_W * Y_DIM * X_DIM / WIDE_W; p++) {
    sc_uint<64> packet = 0;
    for (int w = 0; w < (WIDE_W / NARROW_W); w++) {
      int x = (p * (WIDE_W / NARROW_W) + w) / X_DIM;
      int y = (p * (WIDE_W / NARROW_W) + w) % X_DIM;
      int r_left = (w + 1) * NARROW_W - 1;
      int r_right = w * NARROW_W;
      packet.range(r_left,r_right) = src[x][y];
    }

    dst[p + idx*stride] = packet.to_uint64();
    VLOG(<< "data write: " << dst[p + idx*stride] << " address: " << &dst[p + idx*stride]  << endl);
  }

}



template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
void ACCNAME::read_tensor(
  IDX_T idx,
  WIDE_T *src,
  NARROW_T dst[Y_DIM][X_DIM]) {
#pragma HLS INLINE

	int stride = NARROW_W * Y_DIM * X_DIM / WIDE_W;
  // Read in weight tensor
  for (int p = 0; p < NARROW_W * Y_DIM * X_DIM / WIDE_W; p++) {

		sc_uint<64> packet = src[p + idx*stride];
		VLOG(<< "data read: " << src[p + idx*stride] << " address: " << &src[p + idx*stride]  << endl);
    for (int w = 0; w < (WIDE_W / NARROW_W); w++) {
      int x = (p * (WIDE_W / NARROW_W) + w) / X_DIM;
      int y = (p * (WIDE_W / NARROW_W) + w) % X_DIM;
      int r_left = (w + 1) * NARROW_W - 1;
      int r_right = w * NARROW_W;
      dst[x][y] = (NARROW_T) packet.range(r_left,r_right);
    }
  }
}

template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
void ACCNAME::print_mem(
  IDX_T idx,
  WIDE_T *src) {

	int stride = NARROW_W * Y_DIM * X_DIM / WIDE_W;
  for (int p = 0; p < NARROW_W * Y_DIM * X_DIM / WIDE_W; p++) {
		sc_uint<64> packet = src[p + idx*stride];
    for (int w = 0; w < (WIDE_W / NARROW_W); w++) {
      int r_left = (w + 1) * NARROW_W - 1;
      int r_right = w * NARROW_W;
      cout << (NARROW_T) packet.range(r_left,r_right) << endl;
    }
  }
}



void ACCNAME::alu(
  crf_type *crf_mem,
  crx_type *crx_mem,
  insn_T insn_raw,
  uop_T uop_mem[VTA_UOP_BUFF_DEPTH],
  bus_T *acc_mem,
  bus_T *inp_mem,
  bus_T *wgt_mem,
  bus_T *out_mem
  
  ) {
#pragma HLS INLINE

  unsigned long long temp_insn[2] ;
	temp_insn[0] = insn_raw.range(63,0).to_uint64(); // NEW
	temp_insn[1] = insn_raw.range(127,64).to_uint64(); // NEW
  VTAInsn insn;
	insn.generic = *((VTAGenericInsn *) &temp_insn);

  // Loop offset
  acc_idx_T dst_offset_out = 0;
  inp_idx_T src_offset_out = 0;

  // Outer Loop
  EXE_OUT_LOOP: for (int it_out = 0; it_out < insn.alu.iter_out; it_out++) {
    acc_idx_T dst_offset_in = dst_offset_out;
    inp_idx_T src_offset_in = src_offset_out;

    // Inner Loop
    EXE_IN_LOOP: for (int it_in = 0; it_in < insn.alu.iter_in; it_in++) {
      // Iterate over micro op
      READ_ALU_UOP: for (int upc = insn.alu.uop_bgn; upc < insn.alu.uop_end; upc++) {
#pragma HLS PIPELINE II = 2
        // Read micro-op fields
      	sc_uint<32> uop = uop_mem[upc];

        // Decode
        acc_idx_T dst_idx =
            uop.range(VTA_UOP_ALU_0_1, VTA_UOP_ALU_0_0) + dst_offset_in;
        acc_idx_T src_idx =
            uop.range(VTA_UOP_ALU_1_1, VTA_UOP_ALU_1_0) + src_offset_in;

        // Read in src tensor
        acc_T src_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(src_idx, acc_mem, src_tensor);
        // Read in dst tensor
        acc_T dst_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, acc_mem, dst_tensor);
        // Output tensor
        out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

        // Perform ALU op over matrix elements
        for (int i = 0; i < VTA_BATCH; i++) {
          for (int b = 0; b < VTA_BLOCK_OUT; b++) {
            // Read in operands
            acc_T src_0 = dst_tensor[i][b];
            acc_T src_1 = insn.alu.use_imm ? (acc_T) insn.alu.imm : src_tensor[i][b];
            // int ra = insn.alu.imm; // new
            aluop_shr_arg_T shft_by = src_1.range(VTA_SHR_ARG_BIT_WIDTH - 1, 0);
            aluop_mul_arg_T mul_by = src_1.range(VTA_MUL_ARG_BIT_WIDTH - 1, 0);
            if (insn.alu.alu_opcode == VTA_ALU_OPCODE_MIN || insn.alu.alu_opcode == VTA_ALU_OPCODE_MAX) {
              // Compute Min/Max
              acc_T mix_val = src_0 < src_1 ?
                  (insn.alu.alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
                  (insn.alu.alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
              dst_tensor[i][b] = mix_val;
              o_tensor[i][b] = (out_T) mix_val.range(VTA_OUT_WIDTH - 1, 0);
            } else if (insn.alu.alu_opcode == VTA_ALU_OPCODE_ADD) {
              // Compute Sum
              acc_T add_val =
                  src_0.range(VTA_ACC_WIDTH - 1, 0) + src_1.range(VTA_ACC_WIDTH - 1, 0);
              dst_tensor[i][b] = add_val;
              o_tensor[i][b] = (out_T) add_val.range(VTA_OUT_WIDTH - 1, 0);
            } else if (insn.alu.alu_opcode == VTA_ALU_OPCODE_SHR) {
              // Compute Shift Right
              acc_T shr_val = src_0 >> shft_by;
              dst_tensor[i][b] = shr_val;
              o_tensor[i][b] = (out_T) shr_val.range(VTA_OUT_WIDTH - 1, 0);

            }
          }
        }

        // Write the results back into accumulator
        write_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, dst_tensor, acc_mem);
        // Write the results back in the output buffer
        write_tensor<bus_T, out_T, acc_idx_T, VTA_BUS_WIDTH, VTA_OUT_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, o_tensor, out_mem);
      }
      // Update offsets
      dst_offset_in += insn.alu.dst_factor_in;
      src_offset_in += insn.alu.src_factor_in;
    }
    // Update offsets
    dst_offset_out += insn.alu.dst_factor_out;
    src_offset_out += insn.alu.src_factor_out;
  }
}



void ACCNAME::gemm(
  insn_T insn_raw,
  uop_T uop_mem[VTA_UOP_BUFF_DEPTH],
  bus_T *acc_mem,
  bus_T *inp_mem,
  bus_T *wgt_mem,
  bus_T *out_mem) {

	unsigned long long temp_insn[2] ;
	temp_insn[0] = insn_raw.range(63,0).to_uint64(); // NEW
	temp_insn[1] = insn_raw.range(127,64).to_uint64(); // NEW
  VTAInsn insn;
  insn.generic = *((VTAGenericInsn *) &temp_insn);

  // Loop offset
  acc_idx_T dst_offset_out = 0;
  inp_idx_T src_offset_out = 0;
  wgt_idx_T wgt_offset_out = 0;

  // Outer Loop
  EXE_OUT_LOOP: for (int it_out = 0; it_out < insn.gemm.iter_out; it_out++) { // K DIM
    acc_idx_T dst_offset_in = dst_offset_out;
    inp_idx_T src_offset_in = src_offset_out;
    wgt_idx_T wgt_offset_in = wgt_offset_out;

    // Inner Loop
    EXE_IN_LOOP: for (int it_in = 0; it_in < insn.gemm.iter_in; it_in++) { // M DIM

      // Iterate over micro op
      READ_GEMM_UOP: for (int upc = insn.gemm.uop_bgn; upc < insn.gemm.uop_end; upc++) { // N DIM
        // Read micro-op fields
      	sc_uint<32> uop = uop_mem[upc];

        // Decode indices
        acc_idx_T dst_idx =
            uop.range(VTA_UOP_GEM_0_1, VTA_UOP_GEM_0_0) + dst_offset_in;
        inp_idx_T src_idx =
            uop.range(VTA_UOP_GEM_1_1, VTA_UOP_GEM_1_0) + src_offset_in;
        wgt_idx_T wgt_idx =
            uop.range(VTA_UOP_GEM_2_1, VTA_UOP_GEM_2_0) + wgt_offset_in;

        int dst_idx_i = dst_idx;
        int src_idx_i = src_idx;
        int wgt_idx_i = wgt_idx;

        // Read in weight tensor
        wgt_T w_tensor[VTA_BLOCK_OUT][VTA_BLOCK_IN];
        VLOG( << "----------Weight Read----------" << endl);
        read_tensor<bus_T, wgt_T, wgt_idx_T, VTA_BUS_WIDTH, VTA_WGT_WIDTH, VTA_BLOCK_OUT, VTA_BLOCK_IN>(wgt_idx, wgt_mem, w_tensor);
        // Read in input tensor
        inp_T i_tensor[VTA_BATCH][VTA_BLOCK_IN];
        VLOG( << "----------Input Read----------" << endl);
        read_tensor<bus_T, inp_T, inp_idx_T, VTA_BUS_WIDTH, VTA_INP_WIDTH, VTA_BATCH, VTA_BLOCK_IN>(src_idx, inp_mem, i_tensor);
        // Read in accum tensor
        acc_T a_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        VLOG( << "----------Accum Read----------" << endl);
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, acc_mem, a_tensor);
        // Output tensor
        out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

        // cout << "---------------Load----------------" << endl;
        // for (int i=0; i <16;i++)cout << "o: " << o_tensor[0][i] << " a: " << a_tensor[0][i] << endl;
        // cout << "-----------------------------------" << endl;

        // cout << "-----------------------------------" << endl;
        // Inner GEMM loop  M DIM
        for (int b = 0; b < VTA_BATCH; b++) {  // 1
          for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) { // 16
            // Initialize the accumulator values
            acc_T accum = a_tensor[b][oc];
            // Dot product sum
            sum_T tmp = 0;
            // Inner matrix multiplication loop (input channel/feature)
            for (int ic = 0; ic < VTA_BLOCK_IN; ic++) { // 16
              wgt_T w_elem = w_tensor[oc][ic];
              inp_T i_elem = i_tensor[b][ic];
              mul_T prod_dsp = i_elem * w_elem;
              int x = i_elem;
              int y = w_elem;
              tmp += (sum_T) prod_dsp;
            }
            // Update summation
            accum += (acc_T) tmp;
            // Write back result acc_mem
            a_tensor[b][oc] = insn.gemm.reset_reg ? (acc_T) 0 : accum;
            // And output vector
            o_tensor[b][oc] = (out_T) accum.range(VTA_OUT_WIDTH - 1, 0);
          }
        }
        // cout << "-----------------------------------" << endl;

        // cout << "--------------Store----------------" << endl;
        // for (int i=0; i <16;i++)cout << "o: " << o_tensor[0][i] << " a: " << a_tensor[0][i] << endl;
        // cout << "-----------------------------------" << endl;

        // Write the results back into accumulator
        VLOG( << "----------Accum Write----------" << endl);
        write_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, a_tensor, acc_mem);
        // Write the results back in the output buffer
        VLOG( << "----------Out Write----------" << endl);
        write_tensor<bus_T, out_T, acc_idx_T, VTA_BUS_WIDTH, VTA_OUT_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, o_tensor, out_mem);
        DWAIT(32);
      }
      // Update offsets
      dst_offset_in += insn.gemm.dst_factor_in;
      src_offset_in += insn.gemm.src_factor_in;
      wgt_offset_in += insn.gemm.wgt_factor_in;
    }

    // Update offsets
    dst_offset_out += insn.gemm.dst_factor_out;
    src_offset_out += insn.gemm.src_factor_out;
    wgt_offset_out += insn.gemm.wgt_factor_out;
  }
  DWAIT(42);
  // cout << "+++++++++++++++++++++++++++++++++++++++++++" << endl;
}









void ACCNAME::compute(
  crf_type *crf_mem,
  crx_type *crx_mem,
	uop_T *uop_mem,
  bus_T *inp_mem,
  bus_T *wgt_mem,
  bus_T *out_mem,
  bus_T *acc_mem
  
  ) {

	unsigned long long temp_insn[2] ;

  // Pop GEMM instruction
  insn_T raw_insn = gemm_queue.read();
  insn_T raw_copy = raw_insn;
	temp_insn[0] = raw_copy.range(63,0).to_uint64(); // NEW
	temp_insn[1] = raw_copy.range(127,64).to_uint64(); // NEW

  // Cast to GenericInsn
  VTAInsn insn;
  insn.generic = *((VTAGenericInsn *) &temp_insn);

  // Pop dependence token if instructed
  if (insn.generic.pop_prev_dep) {
    l2g_dep_queue.read();
  }
  if (insn.generic.pop_next_dep) {
    s2g_dep_queue.read();
  }

  // Set done value
  done = 0;
  // Perform action based on opcode
  if (insn.generic.opcode == VTA_OPCODE_FINISH) {
    // Set done flag if we reach a FINISH instruction
    done = 1;
  } else if (insn.generic.opcode == VTA_OPCODE_LOAD) {
    // Initialize indices
    memop_sram_T sram_idx = insn.mem.sram_base;
    memop_dram_T dram_idx = insn.mem.dram_base;
    memop_sram_T x_width =
        (insn.mem.x_pad_0 + insn.mem.x_size + insn.mem.x_pad_1);
    memop_sram_T y_offset_0 = x_width * insn.mem.y_pad_0;
    memop_sram_T y_offset_1 = x_width * insn.mem.y_pad_1;

    DWAIT(13);
    if (insn.mem.memory_type == VTA_MEM_ID_UOP) {
      // Perform data transfer
      uops->burst_read(uops_addr + dram_idx, insn.mem.x_size,  &uop_mem[sram_idx]);
      DWAIT(9);
      for(int i=0;i< insn.mem.x_size;i++){
      	VLOG(<<"[mem wraper] data: read uops[0]["<<i<<"]= " <<  uop_mem[i] <<  " ||  address: " <<  &(uop_mem[i]) <<endl);
      }

    } else if (insn.mem.memory_type == VTA_MEM_ID_ACC) {
      sram_idx=0;
      // Perform data transfer from DRAM
    	VLOG2( << "------------- Loading Bias -------------" << endl);
      load_pad_2d<bus_T, ACC_MAT_AXI_RATIO, VTA_ACC_ELEM_BYTES>(
          data,bias_addr,
          acc_mem,
          0,
          dram_idx,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride,
          insn.mem.x_pad_0,
          insn.mem.x_pad_1,
          y_offset_0,
          y_offset_1);
      load_1ds<crf_type, 2, 4>(
          data,crf_addr,
          crf_mem,
          0,
          insn.mem.sram_base*4,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride);
      load_1ds<crx_type, 8, 1>(
          data,crx_addr,
          crx_mem,
          0,
          insn.mem.sram_base,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride);
    	VLOG2( << "---------------------------------------" << endl);
    }
  }else if (insn.generic.opcode == VTA_OPCODE_GEMM) {
    DWAIT(13);
    gemm(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
  }else if (insn.generic.opcode == VTA_OPCODE_ALU) {
    alu(crf_mem,crx_mem,raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
  }

  // Push dependence token if instructed
  if (insn.generic.push_prev_dep) {
    g2l_dep_queue.write(1);
  }
  if (insn.generic.push_next_dep) {
    g2s_dep_queue.write(1);
  }
}

#endif
