#include <bitset>
#include "vta.h"

void ACCNAME::fetch() {
#pragma HLS resource core = AXI4LiteS metadata = \
    "-bus_bundle CONTROL_BUS" variable = insn_count
#pragma HLS resource core = AXI4LiteS metadata = \
    "-bus_bundle CONTROL_BUS" variable = ins_addr

  unsigned long long temp_insn[2];
  int insn_pointer = pc*2;
  int store_count=0;
  int gemm_count=0;
  int load_count=0;
  int fin_count=0;


  int insn_qty = insn_count.read();
  // cout << " tmp_load_queue: " << tmp_load_queue.num_available()
  //      << " tmp_gemm_queue: " << tmp_gemm_queue.num_available()
  //      << " tmp_store_queue: " << tmp_store_queue.num_available() << endl;

   DPROF(instructions_count->value=insn_qty);

  for (; pc < insn_qty; pc++) {
#pragma HLS PIPELINE
    // Read instruction fields
    

    int curr_ins_addr = ins_addr;
    unsigned long long raw_insn1 = insns.read(curr_ins_addr + insn_pointer);
    unsigned long long raw_insn2 = insns.read(curr_ins_addr + insn_pointer + 1);
    insn_pointer += 2;
    sc_biguint<128> raw_insn;
    raw_insn.range(63, 0) = raw_insn1;
    raw_insn.range(127, 64) = raw_insn2;

    VTAInsn insn;
    temp_insn[0] = raw_insn.range(63, 0).to_uint64();
    temp_insn[1] = raw_insn.range(127, 64).to_uint64();
    insn.generic = *((VTAGenericInsn*)&temp_insn);
    // Do some partial decoding
    opcode_T opcode = insn.generic.opcode;
    memop_id_T memory_type = insn.mem.memory_type;
    // Push to appropriate instruction queue
    if (opcode == VTA_OPCODE_STORE) {
      if(tmp_store_queue.num_free()==0)break;
      tmp_store_queue.write(raw_insn);
      store_count++;
    } else if (opcode == VTA_OPCODE_LOAD) {
      if (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT) {
        if(tmp_load_queue.num_free()==0)break;
        tmp_load_queue.write(raw_insn);
        load_count++;

      } else {
        if(tmp_gemm_queue.num_free()==0)break;
        tmp_gemm_queue.write(raw_insn);
        gemm_count++;
      }
    } else {
      if(tmp_gemm_queue.num_free()==0)break;
      tmp_gemm_queue.write(raw_insn);
      fin_count++;
    }
    DWAIT(10);
  }
  wait();
  // cout << " tmp_load_queue: " << tmp_load_queue.num_available()
  //      << " tmp_gemm_queue: " << tmp_gemm_queue.num_available()
  //      << " tmp_store_queue: " << tmp_store_queue.num_available() << endl;

  // cout << "load_count: " << load_count
  // << " store_count: " << store_count
  // << " gemm_count: " << gemm_count
  // << " fin_count: " << fin_count  << endl;

  wait();
}
