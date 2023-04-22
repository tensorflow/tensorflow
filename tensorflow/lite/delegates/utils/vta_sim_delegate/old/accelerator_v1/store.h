#include "vta.h"


#ifndef __SYNTHESIS__

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



int ACCNAME::Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
	sc_int<64> pl;
	sc_int<32> pr;
	sc_int<32> msk;
	sc_int<32> sm;

	if(shift>0){
		pl = shift;
		pr = 0;
		msk = 0;
		sm = 0;
	}else{
		pl = 1;
		pr = -shift;
		msk = (1 << -shift)-1;
		sm = msk>>1;
	}
	sc_int<64> val = x*pl;
  if (val > MAX32) val = MAX32; // ALU MIN
  if (val < MIN32) val = MIN32; // ALU MAX
  sc_int<64> val_2 = val * qm;

	sc_int<32> temp_1;
	temp_1 = (val_2+POS)/DIVMAX;
	if(val_2<0)temp_1 = (val_2+NEG)/DIVMAX;

	sc_int<32> val_3 = temp_1;
	val_3 = val_3>>pr;

	sc_int<32> temp_2 = temp_1 & msk;
	sc_int<32> temp_3 = (temp_1 < 0) & 1;
	sc_int<32> temp_4 = sm + temp_3;
	sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);

  sc_int<32> result_32 = val_3 + temp_5;
	return result_32;
}

void ACCNAME::store(
  crf_type *crf_mem,
  crx_type *crx_mem,
  bus_T *acc_mem,
  bus_T *out_mem) {

  unsigned long long temp_insn[2] ;

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

  ra = ra_sig.read();

  int m = insn.mem.x_size *16;
  int n = insn.mem.y_size;
  int * a_data = (int *) &acc_mem[0];
  
  int8_t* out_store = (int8_t*) out_mem;
  for(int i=0;i<n ;i++) {
    for(int j=0;j<m ;j++) {

      int idx = j+i*m;
    	int i_r = idx % 2;
			int i_idx = idx-i_r;
			int i_left = (i_r + 1) * 32 - 1;
			int i_right = i_r * 32;
    	int i_x = ((i_idx/2) % 8);
    	int i_y = i_idx / 16;

      bus_T acc_data = acc_mem[j+ (i*m)];
    	sc_uint<64> sc_accum = acc_data;
      int accum = sc_accum.range(i_left,i_right);




      int value = Quantised_Multiplier(a_data[j+ (i*m)] , crf_mem[j],crx_mem[j]);
      sc_int<32> svalue = value + ra; 
      if (svalue > MAX8) svalue = MAX8;
      else if (svalue < MIN8)svalue = MIN8;
      out_store[j+ (i*m)] = svalue.range(7, 0);

      // int8_t asdasd = svalue.range(7, 0);
      // cout <<   crf_mem[j] << " " <<  (int) crx_mem[j]  
      // << " " << a_data[j+ (i*m)] << "-->" << (int) asdasd << endl;

  
    }
  }

  // int * a_data = (int *) &acc_mem[0];
  // int m = insn.mem.x_size *16;
  // int n = insn.mem.y_size;
  // sc_uint<64> packet = 0;
  // int w = 0;
  // int out_idx = 0;
  // for(int i=0;i<n ;i++) {
  //   for(int j=0;j<m ;j++) {
  //     int value = Quantised_Multiplier(a_data[j+ (i*m)] , crf_mem[j],crx_mem[j]);
  //     sc_int<32> svalue = value + ra; 
  //     if (svalue > MAX8) svalue = MAX8;
  //     else if (svalue < MIN8)svalue = MIN8;
  //     out_mem[j+ (i*m)] = svalue.range(7, 0);
  //     int r_left = (w + 1) * 8 - 1;
  //     int r_right = w * 8;
  //     packet.range(r_left,r_right) = svalue.range(7, 0);
  //     w++;
  //     if(w==8){
  //       out_mem[out_idx++]=packet;
  //       w= 0;
  //       packet=0;
  //     }
  //     // cout <<   crf_mem[j] << " " <<  (int) crx_mem[j]  
  //     // << " " << a_data[j+ (i*m)] << "-->" << (int) out_mem[j+ (i*m)] << endl;
  //   }
  //   // cout << "--------------------------" << endl;
  // }
  // cout << "--------------------------" << endl;
  

  // Copy along y dimension
  for (int y = 0; y < insn.mem.y_size; y++) {
  	VLOG2( <<  "------------------------------------"<< endl);
    data->burst_write(output_addr + dram_idx * OUT_MAT_AXI_RATIO, (insn.mem.x_size *VTA_OUT_ELEM_BYTES)/ sizeof(unsigned long long), (unsigned long long*) &out_mem[0 + sram_idx * OUT_MAT_AXI_RATIO]);
    for(int i=0;i<((insn.mem.x_size *VTA_OUT_ELEM_BYTES)/sizeof(unsigned long long));i++){
    	VLOG2(<<"[mem wraper] data: write out_mem[" << i<< " + " << sram_idx << "*" <<OUT_MAT_AXI_RATIO << "]=" <<  out_mem[i + sram_idx * OUT_MAT_AXI_RATIO] <<  " ||  address: " <<  &(out_mem[i + sram_idx * OUT_MAT_AXI_RATIO]) <<endl);
    }
    VLOG2(<<  "------------------------------------"<< endl);

    sram_idx += insn.mem.x_size;
    dram_idx += insn.mem.x_stride;
  }

  // Push dependence token if instructed
  if (insn.mem.push_prev_dep) {
    s2g_dep_queue.write(1);
  }
}

#else

void ACCNAME::store(
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]) {

  unsigned long long temp_insn[2] ;

  // Pop store instruction
  insn_T raw_insn = store_queue.read();
  // Cast to MemInsn
  insn_T raw_copy = raw_insn;
	temp_insn[0] = raw_copy.range(63,0).to_uint64(); // NEW
	temp_insn[1] = raw_copy.range(127,64).to_uint64(); // NEW
  VTAMemInsn insn = *((VTAMemInsn *) &temp_insn);

  // Pop dependence token if instructed
  if (insn.pop_prev_dep) {
    g2s_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = insn.sram_base;
  memop_dram_T dram_idx = insn.dram_base;

  // Copy along y dimension
  for (int y = 0; y < insn.y_size; y++) {
#pragma HLS PIPELINE
    // Perform data transfer
    data->burst_write(output_addr, insn.x_size/4, (unsigned long long*) &out_mem[sram_idx][0]);

#pragma HLS RESOURCE variable = sram_idx core = Mul_LUT
    sram_idx += insn.x_size;
    dram_idx += insn.x_stride;
  }

  // Push dependence token if instructed
  if (insn.push_prev_dep) {
    s2g_dep_queue.write(1);
  }
}

#endif


