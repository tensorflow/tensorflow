#include "vta.h"

template <typename DATA_T, int MAT_AXI_RATIO>
void ACCNAME::reset_mem(
  memop_sram_T &sram_idx,
  memop_sram_T range,
  DATA_T *mem) {

  for (int i = 0; i < range; i ++) {
    for (int j = 0; j < MAT_AXI_RATIO; j ++) {
#pragma HLS UNROLL
      mem[j+ MAT_AXI_RATIO* sram_idx] = 0;
    }
    sram_idx ++;
  }
}



template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
void ACCNAME::load_2d(
	AXI4M_bus_port<unsigned long long> &src,
	sc_in<unsigned int> &addr,
  DATA_T *dst,
  memop_sram_T sram_idx,
  memop_dram_T dram_idx,
  memop_size_T y_size,
  memop_size_T x_size,
  memop_stride_T x_stride) {
#pragma HLS INLINE  

 unsigned int addr_base = addr.read();
 int isram_idx = sram_idx;
 int idram_idx = dram_idx;
 int iy_size = y_size;
 int ix_stride = x_stride;

  int load_length = (x_size*ELEM_BYTES)/sizeof(unsigned long long);
  int nx_size = x_size;
  int ny_size = y_size;
  for (int y = 0; y < y_size; y++) {
  	src->burst_read(addr + dram_idx * MAT_AXI_RATIO, load_length, (unsigned long long*) &dst[0 + sram_idx * MAT_AXI_RATIO]);
    sram_idx += x_size;
    dram_idx += x_stride;
    DWAIT(11);
  }
}



// void ACCNAME::load_2d(
// 	AXI4M_bus_port<unsigned long long> &src,
// 	sc_in<unsigned int> &addr,
//   memop_sram_T sram_idx,
//   memop_dram_T dram_idx,
//   memop_size_T y_size,
//   memop_size_T x_size,
//   memop_stride_T x_stride) {
// #pragma HLS INLINE  

//  unsigned int addr_base = addr.read();
//  int isram_idx = sram_idx;
//  int idram_idx = dram_idx;
//  int iy_size = y_size;
//  int ix_stride = x_stride;

//   int load_length = (x_size*ELEM_BYTES)/sizeof(unsigned long long);
//   int nx_size = x_size;
//   int ny_size = y_size;
//   for (int y = 0; y < y_size; y++) {
//     for (int x = 0; x < x_size; x++) {
//   	src->burst_read(addr + dram_idx + 1, 1, (unsigned long long*) &wgt_mem1[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 2, 1, (unsigned long long*) &wgt_mem2[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 3, 1, (unsigned long long*) &wgt_mem3[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 4, 1, (unsigned long long*) &wgt_mem4[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 5, 1, (unsigned long long*) &wgt_mem5[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 6, 1, (unsigned long long*) &wgt_mem6[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 7, 1, (unsigned long long*) &wgt_mem7[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 8, 1, (unsigned long long*) &wgt_mem8[0 + sram_idx + x]);

//     src->burst_read(addr + dram_idx + 9, 1, (unsigned long long*) &wgt_mem9[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 10, 1, (unsigned long long*) &wgt_mem10[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 11, 1, (unsigned long long*) &wgt_mem11[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 12, 1, (unsigned long long*) &wgt_mem12[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 13, 1, (unsigned long long*) &wgt_mem13[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 14, 1, (unsigned long long*) &wgt_mem14[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 15, 1, (unsigned long long*) &wgt_mem15[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 16, 1, (unsigned long long*) &wgt_mem16[0 + sram_idx + x]);

//   	src->burst_read(addr + dram_idx + 17, 1, (unsigned long long*) &wgt_mem17[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 18, 1, (unsigned long long*) &wgt_mem18[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 19, 1, (unsigned long long*) &wgt_mem19[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 20, 1, (unsigned long long*) &wgt_mem20[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 21, 1, (unsigned long long*) &wgt_mem21[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 22, 1, (unsigned long long*) &wgt_mem22[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 23, 1, (unsigned long long*) &wgt_mem23[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 24, 1, (unsigned long long*) &wgt_mem24[0 + sram_idx + x]);

//   	src->burst_read(addr + dram_idx + 25, 1, (unsigned long long*) &wgt_mem25[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 26, 1, (unsigned long long*) &wgt_mem26[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 27, 1, (unsigned long long*) &wgt_mem27[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 28, 1, (unsigned long long*) &wgt_mem28[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 29, 1, (unsigned long long*) &wgt_mem29[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 30, 1, (unsigned long long*) &wgt_mem30[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 31, 1, (unsigned long long*) &wgt_mem31[0 + sram_idx + x]);
//     src->burst_read(addr + dram_idx + 32, 1, (unsigned long long*) &wgt_mem32[0 + sram_idx + x]);
//     dram_idx += 32;
//     }
//     sram_idx += x_size;
//     DWAIT(11);
//   }
// }





template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
void ACCNAME::load_pad_2d(
	AXI4M_bus_port<unsigned long long> &src,
	sc_in<unsigned int> &addr,
	DATA_T *dst,
  memop_sram_T sram_idx,
  memop_dram_T dram_idx,
  memop_size_T y_size,
  memop_size_T x_size,
  memop_stride_T x_stride,
  memop_pad_T x_pad_0,
  memop_pad_T x_pad_1,
  memop_sram_T y_offset_0,
  memop_sram_T y_offset_1) {
#pragma HLS INLINE

   int load_length= (x_size*ELEM_BYTES)/sizeof(unsigned long long);
  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_0, dst);
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_0, dst);



    for(int i=0;i<load_length;i++){
//#pragma HLS PIPELINE
    	bus_T data = src->read(addr + dram_idx * MAT_AXI_RATIO + i);
    	int x_idx  = i/MAT_AXI_RATIO;
    	int y_idx  = i%MAT_AXI_RATIO;
    	// dst[sram_idx+x_idx][y_idx] = data;
      dst[(sram_idx+x_idx) * MAT_AXI_RATIO + y_idx] = data;
    }

    // src->burst_read(addr + dram_idx * MAT_AXI_RATIO, load_length, (unsigned long long *) &dst[0 + sram_idx * MAT_AXI_RATIO]);


    sram_idx += x_size;
    dram_idx += x_stride;
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_1, dst);
    DWAIT(13);
  }
  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_1, dst);
  DWAIT(2);
}



void ACCNAME::load() {
  unsigned long long temp_insn[2] ;
  wait();
  while(true){
    // Pop load instruction
    insn_T raw_insn = load_queue.read();
    insn_T raw_copy = raw_insn;
    temp_insn[0] = raw_copy.range(63,0).to_uint64(); // NEW
    temp_insn[1] = raw_copy.range(127,64).to_uint64(); // NEW

    // Cast to MemInsn
    VTAInsn insn;
    insn.generic = *((VTAGenericInsn *) &temp_insn);

    // Pop dependence token if instructed
    if (insn.mem.pop_next_dep) {
      g2l_dep_queue.read();
    }

    // Pre-processing
    memop_sram_T x_width = (insn.mem.x_pad_0 + insn.mem.x_size + insn.mem.x_pad_1);
    memop_sram_T y_offset_0 = x_width * insn.mem.y_pad_0;
    memop_sram_T y_offset_1 = x_width * insn.mem.y_pad_1;

    DWAIT(4);
    if (insn.mem.memory_type == VTA_MEM_ID_INP) {
      VLOG2( << "------------- Loading Inputs -------------" << endl);
      load_pad_2d<bus_T, INP_MAT_AXI_RATIO, VTA_INP_ELEM_BYTES>(
          data,input_addr,
          inp_mem,
          insn.mem.sram_base,
          insn.mem.dram_base,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride,
          insn.mem.x_pad_0,
          insn.mem.x_pad_1,
          y_offset_0,
          y_offset_1);
      VLOG2(<< "---------------------------------------" << endl);
    }else if (insn.mem.memory_type == VTA_MEM_ID_WGT) {
      load_2d<bus_T, WGT_MAT_AXI_RATIO, VTA_WGT_ELEM_BYTES>(
          data,weight_addr,
          wgt_mem,
          insn.mem.sram_base,
          insn.mem.dram_base,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride);
    }

    // Push dependence token if instructed
    if (insn.mem.push_next_dep) {
      l2g_dep_queue.write(1);
    }
    DWAIT();
  }
}
