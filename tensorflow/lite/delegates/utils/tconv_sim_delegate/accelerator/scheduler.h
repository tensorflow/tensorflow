
void ACCNAME::load_weights(int r_pointer, int d) {
  for (int i = 0; i < d; i++) {
    ACC_DTYPE<32> w1 = rhsdata1[r_pointer];
    ACC_DTYPE<32> w2 = rhsdata2[r_pointer];
    ACC_DTYPE<32> w3 = rhsdata3[r_pointer];
    ACC_DTYPE<32> w4 = rhsdata4[r_pointer];
    rhs1a_1[i] = w1;
    rhs1b_1[i] = w2;
    rhs1c_1[i] = w3;
    rhs1d_1[i] = w4;
    // rhs2a_1[i]=w1;
    // rhs2b_1[i]=w2;
    // rhs2c_1[i]=w3;
    // rhs2d_1[i]=w4;
    // rhs3a_1[i]=w1;
    // rhs3b_1[i]=w2;
    // rhs3c_1[i]=w3;
    // rhs3d_1[i]=w4;
    // rhs4a_1[i]=w1;
    // rhs4b_1[i]=w2;
    // rhs4c_1[i]=w3;
    // rhs4d_1[i]=w4;
    r_pointer++;
#ifndef __SYNTHESIS__
    weightbuf_p->value =
        (i + 1) > weightbuf_p->value ? (i + 1) : weightbuf_p->value;
    schS.write(20);
    DWAIT(3);
#endif
  }
}

void ACCNAME::schedule_gemm_unit(int unit_counter, int l_pointer, int l,
                                 int r) {
  int x1 = lhs_sum1[l];
  int x2 = lhs_sum2[l];
  int x3 = lhs_sum3[l];
  int x4 = lhs_sum4[l];

  DWAIT(14);
  if (unit_counter == 0) {
    schS.write(41);
    while (!gemm_unit_1_ready.read()) wait();
    gemm_unit_1_ready.write(0);
    gemm_unit_1_l_pointer = l_pointer;
    WRQ1.write(x1);
    WRQ1.write(x2);
    WRQ1.write(x3);
    WRQ1.write(x4);
    wait();
  }

  // if(unit_counter==1){
  // 	schS.write(42);
  // 	while(!gemm_unit_2_ready.read())wait();
  // 	gemm_unit_2_ready.write(0);
  // 	gemm_unit_2_l_pointer=l_pointer;
  // 	WRQ2.write(x1);
  // 	WRQ2.write(x2);
  // 	WRQ2.write(x3);
  // 	WRQ2.write(x4);
  // 	wait();
  // }

  // if(unit_counter==2){
  // 	schS.write(43);
  // 	while(!gemm_unit_3_ready.read())wait();
  // 	gemm_unit_3_ready.write(0);
  // 	gemm_unit_3_l_pointer=l_pointer;
  // 	WRQ3.write(x1);
  // 	WRQ3.write(x2);
  // 	WRQ3.write(x3);
  // 	WRQ3.write(x4);
  // 	wait();
  // }

  // if(unit_counter==3){
  // 	schS.write(44);
  // 	while(!gemm_unit_4_ready.read())wait();
  // 	gemm_unit_4_ready.write(0);
  // 	gemm_unit_4_l_pointer=l_pointer;
  // 	WRQ4.write(x1);
  // 	WRQ4.write(x2);
  // 	WRQ4.write(x3);
  // 	WRQ4.write(x4);
  // 	wait();
  // }
}

void ACCNAME::overwrite_weights_check() {
  while (!gemm_unit_1_ready.read() || gemm_unit_1_iwuse.read() != 0) wait();
  while (!gemm_unit_2_ready.read() || gemm_unit_2_iwuse.read() != 0) wait();
  while (!gemm_unit_3_ready.read() || gemm_unit_3_iwuse.read() != 0) wait();
  while (!gemm_unit_4_ready.read() || gemm_unit_4_iwuse.read() != 0) wait();
}

void ACCNAME::Scheduler() {
  int unit_counter = 0;
  gemm_unit_1_ready.write(1);
  gemm_unit_2_ready.write(1);
  gemm_unit_3_ready.write(1);
  gemm_unit_4_ready.write(1);
  gemm_unit_1_l_pointer.write(0);
  gemm_unit_2_l_pointer.write(0);
  gemm_unit_3_l_pointer.write(0);
  gemm_unit_4_l_pointer.write(0);
  schS.write(0);
  wait();
  while (1) {
    schS.write(10);
    while (!schedule.read()) wait();

    dex_i = 0;
    schS.write(1);
    int dm = depth / 4;
    int rmax = rhs_block_max;
    int lmax = lhs_block_max;
    for (int r = 0; r < rhs_block_max; r += 4) {
      int r4 = r / 4;
      int r_pointer = r4 * dm;
      schS.write(2);
      overwrite_weights_check();
      load_weights(r_pointer, dm);
      DWAIT(15);
      schS.write(4);
      for (int l = 0; l < lhs_block_max; l += 4) {
        schS.write(5);
        int l4 = l / 4;
        int l_pointer = l4 * dm;
        schedule_gemm_unit(0, l_pointer, l4, r4);
        // unit_counter=((unit_counter+1)%4);
        schS.write(6);
        wait();
        DWAIT(10);
      }
    }
    schS.write(7);
    schedule.write(0);
    schS.write(8);
    wait();
  }
}
