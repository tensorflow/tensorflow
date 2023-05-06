void ACCNAME::Data_In() {
  int llength = 0;
  int rlength = 0;

  wait();
  while (1) {
    while (!read_inputs.read())
      wait();
    llength = llen.read();
    rlength = rlen.read();
    int la = 0;
    int lb = 0;
    int ra = 0;
    int rb = 0;

    DWAIT(3);
    if (ltake.read()) {
      for (int i = 0; i < llength / 4; i++) {
        ACC_DTYPE<32> data1 = din1.read().data.to_int();
        ACC_DTYPE<32> data2 = din2.read().data.to_int();
        ACC_DTYPE<32> data3 = din3.read().data.to_int();
        ACC_DTYPE<32> data4 = din4.read().data.to_int();
        lb++;
        lhsdata1a[la] = data1;
        lhsdata1b[la] = data1;
        lhsdata1c[la] = data1;
        lhsdata1d[la] = data1;
        lhsdata2a[la] = data2;
        lhsdata2b[la] = data2;
        lhsdata2c[la] = data2;
        lhsdata2d[la] = data2;
        lhsdata3a[la] = data3;
        lhsdata3b[la] = data3;
        lhsdata3c[la] = data3;
        lhsdata3d[la] = data3;
        lhsdata4a[la] = data4;
        lhsdata4b[la] = data4;
        lhsdata4c[la] = data4;
        lhsdata4d[la] = data4;
        la = lb;
#ifndef __SYNTHESIS__
        inputbuf_p->value = la;
        DWAIT();
#endif
      }
      for (int i = 0; i < rlength; i++) {
        ACC_DTYPE<32> wsums1 = din1.read().data.to_int();
        ACC_DTYPE<32> wsums2 = din2.read().data.to_int();
        ACC_DTYPE<32> wsums3 = din3.read().data.to_int();
        ACC_DTYPE<32> wsums4 = din4.read().data.to_int();
        ACC_DTYPE<32> rfs1 = din1.read().data.to_int();
        ACC_DTYPE<32> rfs2 = din2.read().data.to_int();
        ACC_DTYPE<32> rfs3 = din3.read().data.to_int();
        ACC_DTYPE<32> rfs4 = din4.read().data.to_int();
        ACC_DTYPE<32> exs = din1.read().data.to_int();
        rb++;
        lhs_sum1[ra] = wsums1;
        lhs_sum2[ra] = wsums2;
        lhs_sum3[ra] = wsums3;
        lhs_sum4[ra] = wsums4;
        crf1[ra] = rfs1;
        crf2[ra] = rfs2;
        crf3[ra] = rfs3;
        crf4[ra] = rfs4;
        crx[ra] = exs;
        ra = rb;
        DWAIT();
      }
    }

    DWAIT();
    if (rtake.read()) {
      for (int i = 0; i < rlength / 4; i++) {
        ACC_DTYPE<32> data1 = din1.read().data.to_int();
        ACC_DTYPE<32> data2 = din2.read().data.to_int();
        ACC_DTYPE<32> data3 = din3.read().data.to_int();
        ACC_DTYPE<32> data4 = din4.read().data.to_int();
        rb++;
        rhsdata1[ra] = data1;
        rhsdata2[ra] = data2;
        rhsdata3[ra] = data3;
        rhsdata4[ra] = data4;
        ra = rb;
#ifndef __SYNTHESIS__
        gweightbuf_p->value = ra;
        DWAIT();
#endif
      }
      for (int i = 0; i < llength; i++) {
        ACC_DTYPE<32> isums1 = din1.read().data.to_int();
        ACC_DTYPE<32> isums2 = din2.read().data.to_int();
        ACC_DTYPE<32> isums3 = din3.read().data.to_int();
        ACC_DTYPE<32> isums4 = din4.read().data.to_int();
        lb++;
        rhs_sum1[la] = isums1;
        rhs_sum2[la] = isums2;
        rhs_sum3[la] = isums3;
        rhs_sum4[la] = isums4;
        la = lb;
        DWAIT();
      }
    }
    d_in1.write(0);
    while (read_inputs.read())
      wait();
  }
}
