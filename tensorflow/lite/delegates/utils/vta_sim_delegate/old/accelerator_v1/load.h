#include "vta.h"


#ifndef __SYNTHESIS__

#include <bitset>
using namespace std;

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

  for (int y = 0; y < y_size; y++) {
  	src->burst_read(addr + dram_idx * MAT_AXI_RATIO, (x_size*ELEM_BYTES)/sizeof(unsigned long long), (unsigned long long*) &dst[0 + sram_idx * MAT_AXI_RATIO]);
    sram_idx += x_size;
    dram_idx += x_stride;
  }
}

template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
void ACCNAME::load_1ds(
	AXI4M_bus_port<unsigned long long> &src,
	sc_in<unsigned int> &addr,
  DATA_T *dst,
  memop_sram_T sram_idx,
  memop_dram_T dram_idx,
  memop_size_T y_size,
  memop_size_T x_size,
  memop_stride_T x_stride) {
#pragma HLS INLINE
  
  int len = (y_size*ELEM_BYTES)/sizeof(unsigned long long);
  
  // src->burst_read(addr + dram_idx * MAT_AXI_RATIO, (y_size*ELEM_BYTES)/sizeof(unsigned long long), (unsigned long long*) &dst[0 + sram_idx * MAT_AXI_RATIO]);

  // for(int i=0;i<y_size;i++){
  //   cout <<"[mem wraper] data: read dst[" << i<< " + " << sram_idx << "*" <<MAT_AXI_RATIO << "]=" << (int) dst[i + sram_idx * MAT_AXI_RATIO] <<  " ||  address: " <<  (&(dst[i + sram_idx * MAT_AXI_RATIO])) <<endl;
  // }
  // for (int y = 0; y < read_len; y++) {
  // 	src->burst_read(addr + dram_idx * MAT_AXI_RATIO, (x_size*ELEM_BYTES)/sizeof(unsigned long long), (unsigned long long*) &dst[0 + sram_idx * MAT_AXI_RATIO]);
  //   sram_idx += x_size;
  //   dram_idx += x_stride;
  // }


  int read_len= (y_size*ELEM_BYTES)/sizeof(unsigned long long);
	int ele_size = ELEM_BYTES * 8;
	for(int i=0;i<read_len;i++){
		// bus_T data = src->read(addr + dram_idx);
    sc_uint<64> data = src->read(addr + dram_idx);

		for(int w=0;w<MAT_AXI_RATIO;w++){
      int r_left = (w + 1) * ele_size - 1;
      int r_right = w * ele_size;
      dst[sram_idx++]=data.range(r_left,r_right);
		}
    dram_idx++;
	}
}



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

  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_0, dst);
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_0, dst);
    VLOG2(<<  "------------------------------------"<< endl);
    VLOG2(<<"start dst["<<sram_idx<<"][0]  ||  address: " <<  &dst[0 + sram_idx * MAT_AXI_RATIO] <<endl);
    src->burst_read(addr + dram_idx * MAT_AXI_RATIO, (x_size*ELEM_BYTES)/sizeof(unsigned long long), (unsigned long long *) &dst[0 + sram_idx * MAT_AXI_RATIO]);

    for(int i=0;i<((x_size*ELEM_BYTES)/sizeof(unsigned long long));i++){
    	VLOG2(<<"[mem wraper] data: read dst[" << i<< " + " << sram_idx << "*" <<MAT_AXI_RATIO << "]=" <<  dst[i + sram_idx * MAT_AXI_RATIO] <<  " ||  address: " <<  &(dst[i + sram_idx * MAT_AXI_RATIO]) <<endl);
    }
    VLOG2(<<  "------------------------------------"<< endl);
    sram_idx += x_size;
    dram_idx += x_stride;
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_1, dst);
  }
  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_1, dst);
}



void ACCNAME::load(
  bus_T *inp_mem,
  bus_T *wgt_mem) {

  unsigned long long temp_insn[2] ;
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
}

#else
#endif
