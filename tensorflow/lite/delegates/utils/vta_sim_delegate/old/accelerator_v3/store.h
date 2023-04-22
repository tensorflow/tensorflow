#include "vta.h"

// int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
//   int nshift = shift;
//   int total_shift = 31 - shift;
//   sc_int<64> x_64 = x;
//   sc_int<64> quantized_multiplier_64(qm);
//   sc_int<64> one = 1;
//   sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
//   sc_int<64> result = x_64 * quantized_multiplier_64 + round;// ALU ADD + ALU MUL
//   result = result >> total_shift; // ALU SHRI
//   int nresult = result;
//   if (result > MAX32) result = MAX32; // ALU MIN
//   if (result < MIN32) result = MIN32; // ALU MAX
//   sc_int<32> result_32 = result; 
//   return result_32;
// }


void ACCNAME::store() {
  unsigned long long temp_insn[2] ;
  wait();
	while(true){
    // Pop store instruction
    insn_T raw_insn = store_queue.read();
    insn_T raw_copy = raw_insn;
    temp_insn[0] = raw_copy.range(63,0).to_uint64(); // NEW
    temp_insn[1] = raw_copy.range(127,64).to_uint64(); // NEW

    // Cast to MemInsn
    VTAInsn insn;
    insn.generic = *((VTAGenericInsn *) &temp_insn);

    // Pop dependence token if instructed
    if (insn.mem.pop_prev_dep) {
      g2s_dep_queue.read();
    }

    // Initialize indices
    memop_sram_T sram_idx = insn.mem.sram_base;
    memop_dram_T dram_idx = insn.mem.dram_base;

    // Copy along y dimension
    int write_length = (insn.mem.x_size *VTA_OUT_ELEM_BYTES)/ sizeof(unsigned long long);
    for (int y = 0; y < insn.mem.y_size; y++) {
      for(int i=0;i<write_length;i++){
        int x_idx  = i/OUT_MAT_AXI_RATIO;
        int y_idx  = i%OUT_MAT_AXI_RATIO;
        data->write(output_addr + dram_idx * OUT_MAT_AXI_RATIO + i,(unsigned long long*) &out_mem[(sram_idx+x_idx)* OUT_MAT_AXI_RATIO + y_idx]);
      }

      // data->burst_write(output_addr + dram_idx * OUT_MAT_AXI_RATIO, write_length, (unsigned long long*) &out_mem[0 + sram_idx * OUT_MAT_AXI_RATIO]);

      sram_idx += insn.mem.x_size;
      dram_idx += insn.mem.x_stride;
      DWAIT(6);
    }

    // Push dependence token if instructed
    if (insn.mem.push_prev_dep) {
      s2g_dep_queue.write(1);
    }
    DWAIT();
  }
}


