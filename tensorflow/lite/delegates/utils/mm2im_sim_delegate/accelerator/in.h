#define COMPUTE -1
#define OUTPUT -2

void ACCNAME::Input_Handler() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=rmax
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=lmax
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=outS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=schS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=p1S
  // clang-format on

  inS.write(0);
  int r_max;
  int l_max;
  DATA last = {5000, 1};
  wait();
  while (1) {
    inS.write(1);
    ACC_DTYPE<32> header_lite = din1.read().data.to_int();
    if (header_lite != COMPUTE && header_lite != OUTPUT) {
      bool rhs_take = header_lite.range(0, 0);
      bool lhs_take = header_lite.range(1, 1);
      depth = header_lite.range(31, 18);

      inS.write(2);
      ACC_DTYPE<32> hcounts = din1.read().data.to_int();
      int rhs_count = hcounts.range(15, 0);
      int lhs_count = hcounts.range(31, 16);
      out_len = rhs_count * lhs_count / 4;

      inS.write(3);
      ACC_DTYPE<32> lengths = din1.read().data.to_int();
      int rhs_length = lengths.range(15, 0);
      int lhs_length = lengths.range(31, 16);

      d_in1.write(1);
      read_inputs.write(1);
      rtake.write(rhs_take);
      ltake.write(lhs_take);
      rlen.write(rhs_length);
      llen.write(lhs_length);
      DWAIT();

      inS.write(4);
      while (d_in1.read()) wait();
      read_inputs.write(0);
      inS.write(5);
      wait();

      inS.write(6);
      if (lhs_take) {
        r_max = rhs_count;
        l_max = lhs_count;
      }
      DWAIT(3);

    } else if (header_lite == COMPUTE) {
      while (schedule.read()) wait();
      rhs_block_max.write(r_max);
      lhs_block_max.write(l_max);
      rmax.write(r_max);
      lmax.write(l_max);
      schedule.write(1);
      out_check.write(1);
      inS.write(7);
      wait();
      while (schedule.read()) wait();
    } else if (header_lite == OUTPUT) {
      out_c = din1.read().data.to_int();
      out_r = din1.read().data.to_int();
      out_int8_len = din1.read().data.to_int();
      out_int8_lenr = din1.read().data.to_int();
      ra = din1.read().data.to_int();

      for (int i = 0; i < out_c; i++) {
        bias[i] = din1.read().data.to_int();
        crf1[i] = din1.read().data.to_int();
        crx[i] = din1.read().data.to_int();
      }

      bias_quantize.write(1);
      wait();
      while (bias_quantize.read()) wait();

      while (send_output.read()) wait();
      send_output.write(1);
      out_check.write(1);
      wait();
      while (send_output.read()) wait();
    }
  }
}
