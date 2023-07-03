void ACCNAME::Data_In() {
  #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    data_inS

  data_inS.write(0);
  wait();
  while (1) {
    data_inS.write(1);
    while (!load_data.read())
      wait();

    if (load_wgt) {
      data_inS.write(2);
      // wgt_packet wp =
      //     wgt_packet(din1.read().data.to_uint(), din1.read().data.to_uint());
      wgt_packet wp = wgt_packet(&din1);
      nfilters = din1.read().data;
      int temp = nfilters;
      int dex = 0;
      for (int i = 0; i < wp.wgt_rows; i++) {
        for (int j = 0; j < wp.wgt_depth; j++) {
          for (int u = 0; u < UF; u += 4) {
            sc_uint<32> data = din1.read().data;
            acc_dt tmp1 = data.range(7, 0).to_int();
            acc_dt tmp2 = data.range(15, 8).to_int();
            acc_dt tmp3 = data.range(23, 16).to_int();
            acc_dt tmp4 = data.range(31, 24).to_int();
            wgt_buf[dex][u + 0] = tmp1;
            wgt_buf[dex][u + 1] = tmp2;
            wgt_buf[dex][u + 2] = tmp3;
            wgt_buf[dex][u + 3] = tmp4;
            DWAIT();
          }
          dex++;
        }
        sc_uint<32> data = din1.read().data;
        wgt_sum_buf[i] = data;
      }
      for (int i = 0; i < nfilters; i++) {
        sc_uint<32> data = din1.read().data;
        sc_uint<32> data1 = din1.read().data;
        sc_uint<32> data2 = din1.read().data;
        bias_buf[i] = data;
        crf_buf[i] = data1;
        crx_buf[i] = data2;
      }
    }

    if (load_inp) {
      data_inS.write(3);
      inp_packet ip =
          inp_packet(&din1);
      int dex = 0;
      for (int i = 0; i < ip.inp_rows; i++) {
        for (int j = 0; j < ip.inp_depth; j++) {
          for (int u = 0; u < UF; u += 4) {
            sc_uint<32> data = din1.read().data;
            acc_dt tmp1 = data.range(7, 0).to_int();
            acc_dt tmp2 = data.range(15, 8).to_int();
            acc_dt tmp3 = data.range(23, 16).to_int();
            acc_dt tmp4 = data.range(31, 24).to_int();
            inp_buf[dex][u + 0] = tmp1;
            inp_buf[dex][u + 1] = tmp2;
            inp_buf[dex][u + 2] = tmp3;
            inp_buf[dex][u + 3] = tmp4;
            DWAIT();
          }
          dex++;
        }
      }
    }

    if (load_map) {
      data_inS.write(4);
      int out_dex = 0;
      for (int i = 0; i < row_size; i++) {
        int output_size = din1.read().data;
        out_starts[i] = out_dex;
        out_size[i] = output_size;
        for (int j = 0; j < output_size; j++) {
          outmap_buf[out_dex++] = din1.read().data;
          DWAIT();
        }
      }
    }

    if (load_col_map) {
      data_inS.write(5);
      int out_dex = 0;
      number_of_rows = din1.read().data;
      for (int i = 0; i < number_of_rows; i++) {
        int col_dex_size = din1.read().data;
        col_indice_starts[i] = out_dex;
        col_indice_lens[i] = col_dex_size;
        for (int j = 0; j < col_dex_size; j++) {
          col_indices[out_dex] = din1.read().data;
          out_indices[out_dex++] = din1.read().data;
          DWAIT();
        }
      }
    }

    load_data.write(false);
    wait();
  }
}
