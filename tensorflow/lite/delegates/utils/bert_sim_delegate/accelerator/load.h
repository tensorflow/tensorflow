void ACCNAME::load_weights() {
  wait();
  while (true) {
    while (!wgt_load)
      wait();
    opcode wgt_insn(wgt_insn1, wgt_insn2);
    sc_uint<16> wgt_idx = 0;
    sc_uint<16> m_inc = wgt_insn.y_size;
    sc_uint<16> load_length = wgt_insn.x_size;

    unsigned int mem_base = weight_addr.read();
    sc_uint<32> mem_idx = wgt_insn.doffset;
    sc_uint<16> dstride = wgt_insn.dstride;
    sc_uint<16> dstride_jump = dstride * 4;
    sc_uint<32> mem_idx1 = mem_idx;
    sc_uint<32> mem_idx2 = mem_idx + dstride * 1;
    sc_uint<32> mem_idx3 = mem_idx + dstride * 2;
    sc_uint<32> mem_idx4 = mem_idx + dstride * 3;

    for (int i = 0; i < m_inc / 4; i++) {
#pragma HLS pipeline II = 1
      weight_port->burst_read(mem_base + mem_idx1, load_length,
                              (unsigned long long *)&wgt_mem1[wgt_idx]);
      weight_port->burst_read(mem_base + mem_idx2, load_length,
                              (unsigned long long *)&wgt_mem2[wgt_idx]);
      weight_port->burst_read(mem_base + mem_idx3, load_length,
                              (unsigned long long *)&wgt_mem3[wgt_idx]);
      weight_port->burst_read(mem_base + mem_idx4, load_length,
                              (unsigned long long *)&wgt_mem4[wgt_idx]);

      mem_idx1 += dstride_jump;
      mem_idx2 += dstride_jump;
      mem_idx3 += dstride_jump;
      mem_idx4 += dstride_jump;
      wgt_idx += load_length;
    }
    wgt_load.write(false);
    while (loading.read())
      wait();
    wait();
  }
}

void ACCNAME::load_inputs() {
  wait();
  while (true) {
    while (!inp_load)
      wait();
    opcode inp_insn(inp_insn1, inp_insn2);
    sc_uint<16> inp_idx = 0;
    sc_uint<16> n_inc = inp_insn.y_size;
    sc_uint<16> load_length = inp_insn.x_size;

    unsigned int mem_base = input_addr.read();
    sc_uint<32> mem_idx = inp_insn.doffset;
    sc_uint<16> dstride = inp_insn.dstride;
    sc_uint<16> dstride_jump = dstride * 4;
    sc_uint<32> mem_idx1 = mem_idx;
    sc_uint<32> mem_idx2 = mem_idx + dstride * 1;
    sc_uint<32> mem_idx3 = mem_idx + dstride * 2;
    sc_uint<32> mem_idx4 = mem_idx + dstride * 3;

    for (int i = 0; i < n_inc / 4; i++) {
#pragma HLS pipeline II = 1
      input_port->burst_read(mem_base + mem_idx1, load_length,
                             (unsigned long long *)&inp_mem1[inp_idx]);
      input_port->burst_read(mem_base + mem_idx2, load_length,
                             (unsigned long long *)&inp_mem2[inp_idx]);
      input_port->burst_read(mem_base + mem_idx3, load_length,
                             (unsigned long long *)&inp_mem3[inp_idx]);
      input_port->burst_read(mem_base + mem_idx4, load_length,
                             (unsigned long long *)&inp_mem4[inp_idx]);

      mem_idx1 += dstride_jump;
      mem_idx2 += dstride_jump;
      mem_idx3 += dstride_jump;
      mem_idx4 += dstride_jump;
      inp_idx += load_length;
    }
    inp_load.write(false);
    while (loading.read())
      wait();
    wait();
  }
}

void ACCNAME::load_bias() {
  wait();
  while (true) {
    while (!bias_load)
      wait();
    opcode bias_insn(bias_insn1, bias_insn2);
    unsigned int mem_base = bias_addr.read();
    sc_uint<16> bias_idx = 0;
    sc_uint<32> mem_idx = bias_insn.doffset;
    sc_uint<16> dstride = bias_insn.dstride;
    sc_uint<16> m_inc = bias_insn.y_size;
    sc_uint<16> load_length = bias_insn.x_size;

    for (int i = 0; i < m_inc; i++) {
#pragma HLS pipeline II = 1
      bias_port->burst_read(mem_base + mem_idx, load_length,
                            (unsigned long long *)&acc_mem[bias_idx]);
#pragma HLS RESOURCE variable = bias_idx core = Mul_LUT
      bias_idx += load_length;
      mem_idx += dstride;
    }
    bias_load.write(false);
    while (loading.read())
      wait();
    wait();
  }
}