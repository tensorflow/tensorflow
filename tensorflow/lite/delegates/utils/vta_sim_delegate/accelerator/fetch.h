#include "vta.h"

void ACCNAME::fetch() {
	unsigned long long temp_insn[2];
	insn_T reset_ins;
	reset_ins.range(63, 0) = 3;
	reset_ins.range(127, 64) = 0;

	fetch_run=false;
	fetch_resetted.write(false);
	start_count=0;
	done_count=0;
	wait();
	while(true){
		if(start.read() > start_count) {
			fetch_run = true;
    		start_count++;
    		done_count++;
    		fetch_resetted.write(false);

    		int insn_qty = insn_count.read();
    		unsigned int curr_ins_addr = ins_addr.read();
    		int insn_pointer = 0;
    		ra = ra_sig.read();
    		is_flipped = flipped.read();
    		DWAIT();
    		for (int pc=0;pc < insn_qty; pc++) {
    	#pragma HLS PIPELINE II=1
    			// Read instruction fields
    			unsigned long long raw_insn1 = insns.read(curr_ins_addr + insn_pointer);
    			unsigned long long raw_insn2 = insns.read(curr_ins_addr + insn_pointer + 1);
    			insn_pointer += 2;

    			insn_T raw_insn;
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
    				store_queue.write(raw_insn);
    			} else if (opcode == VTA_OPCODE_LOAD) {
    				if (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT) {
    					load_queue.write(raw_insn);
    				} else {
    					gemm_queue.write(raw_insn);
    				}
    			} else {
    				gemm_queue.write(raw_insn);
    			}
    		}
    }
    DWAIT();
    if(reset_vta && !fetch_resetted.read()){
			start_count = 0;
			done_count = 0;
			gemm_queue.write(reset_ins);
			fetch_resetted.write(true);
      DWAIT();
		}
    DWAIT();
  }
}


void ACCNAME::Counter() {
  wait();
  while (1) {
    per_batch_cycles->value++;
    // if (vtaS.read()==10)
    //   fetch_cycles->value++;
    // if (vtaS.read()==20)
    //   load_cycles->value++;
    // if (vtaS.read()==30)
    //   compute_cycles->value++;
    // if (vtaS.read()==40)
    //   store_cycles->value++;
    wait();
  }
}



  // cout << " tmp_load_queue: " << tmp_load_queue.num_available()
  //      << " tmp_gemm_queue: " << tmp_gemm_queue.num_available()
  //      << " tmp_store_queue: " << tmp_store_queue.num_available() << endl;

  // cout << " tmp_load_queue: " << tmp_load_queue.num_available()
  //      << " tmp_gemm_queue: " << tmp_gemm_queue.num_available()
  //      << " tmp_store_queue: " << tmp_store_queue.num_available() << endl;

  // cout << "load_count: " << load_count
  // << " store_count: " << store_count
  // << " gemm_count: " << gemm_count
  // << " fin_count: " << fin_count  << endl;
