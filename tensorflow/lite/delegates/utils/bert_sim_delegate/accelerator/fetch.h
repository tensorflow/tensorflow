void ACCNAME::fetch() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=start_acc
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=done_acc
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=reset_acc
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=insn_count
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=insn_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=input_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=weight_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=bias_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=output_addr
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=crf
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=crx
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=ra
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=depth
  // clang-format on

  start_count = 0;
  done_count = 0;
  crf_val = 0;
  crx_val = 0;
  ra_val = 0;
  fetch_resetted.write(false);
  wgt_load.write(false);
  inp_load.write(false);
  bias_load.write(false);
  opcode insn(0, 0);
  wait();
  while (true) {
    wait();

    if (start_acc.read() > start_count) {
      cout << start_acc.read() << endl;
      start_count++;
      done_count++;
      fetch_resetted.write(false);
      crf_val = crf.read();
      crx_val = crx.read();
      ra_val = ra.read();

      if (crx_val > 0) {
        pl = crx_val;
        pr = 0;
        msk = 0;
        sm = 0;
      } else {
        pl = 1;
        pr = -crx_val;
        msk = (1 << -crx_val) - 1;
        sm = msk >> 1;
      }

      int insn_len = insn_count.read();
      int insn_idx = 0;
      DPROF(ins_count->value += insn_len;)
      for (int pc = 0; pc < insn_len; pc++) {
        sc_uint<64> in2 = insn_port.read(insn_addr + insn_idx);
        sc_uint<64> in1 = insn_port.read(insn_addr + insn_idx + 1);
        insn_idx += 2;
        opcode insn_read(in1, in2);
        unsigned long long p1 = in1;
        unsigned long long p2 = in2;
        insn = insn_read;

        if (insn.op == 1) {
          while (wgt_load.read())
            wait();
          wgt_insn1 = in1;
          wgt_insn2 = in2;
          loading.write(true);
          wgt_load.write(true);
          DWAIT();
          while (wgt_load.read())
            wait();
          loading.write(false);
          DWAIT();
        } else if (insn.op == 2) {
          while (inp_load.read())
            wait();
          inp_insn1 = in1;
          inp_insn2 = in2;
          loading.write(true);
          inp_load.write(true);
          DWAIT();
          while (inp_load.read())
            wait();
          loading.write(false);
          DWAIT();
        } else if (insn.op == 3) {
          while (bias_load.read())
            wait();
          bias_insn1 = in1;
          bias_insn2 = in2;
          loading.write(true);
          bias_load.write(true);
          DWAIT();
          while (bias_load.read())
            wait();
          loading.write(false);
          DWAIT();
        } else {
          while (wgt_load.read())
            wait();
          while (inp_load.read())
            wait();
          while (bias_load.read())
            wait();

          store_doffset.write(insn.doffset);
          store_dstride.write(insn.dstride);
          inp_block.write(insn.y_size);
          wgt_block.write(insn.x_size);
          depth_val.write(depth.read());
          schedule.write(true);
          DWAIT();
          while (schedule.read())
            wait();
        }
      }
      wait();
      done_acc.write(done_count);
    }

    if (reset_acc && !fetch_resetted.read()) {
      start_count = 0;
      done_count = 0;
      fetch_resetted.write(true);
      done_acc.write(done_count);
      wait();
    }

#ifndef __SYNTHESIS__
    DWAIT();
    sc_pause();
#endif
    wait();
  }
}