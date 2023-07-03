

void ACCNAME::start_compute(int col_indice_len) {
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].col_size.write(col_indice_len);
    vars[i].compute.write(true);
    vars[i].out.write(true);
  }
}

void ACCNAME::stop_compute() {
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].compute.write(false);
    vars[i].reset_compute.write(true);
    vars[i].out.write(false);
  }
}

void ACCNAME::init_PE_signals() {
#pragma HLS inline OFF

  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].online.write(false);
    vars[i].compute.write(false);
    vars[i].col_size.write(0);
    vars[i].start_addr_p.write(0);
    vars[i].send_len_p.write(0);
    vars[i].bias_data.write(0);
    vars[i].crf_data.write(0);
    vars[i].crx_data.write(0);
    vars[i].ra_data.write(0);

    vars[i].send.write(false);
    vars[i].out.write(false);
    vars[i].cols_per_filter.write(cols_per_filter);
    vars[i].depth.write(depth);
  }
}

bool ACCNAME::wgt_loaded() {
#pragma HLS inline OFF
  bool loop = true;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    loop = loop && vars[i].wgt_loaded;
  }
  return loop;
}

bool ACCNAME::compute_done() {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    if (!vars[i].compute_done || !vars[i].out_done) {
      loop = true;
      break;
    }
  }
  return loop;
}

bool ACCNAME::compute_resetted() {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    loop = loop || vars[i].compute_done || vars[i].out_done;
  }
  return loop;
}

bool ACCNAME::store_done() {
#pragma HLS inline OFF
  bool loop = false;
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    if (!vars[i].send_done) loop = true;
  }
  return loop;
}

void ACCNAME::activate_PEs() {
#pragma HLS inline OFF
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].cols_per_filter.write(cols_per_filter);
    vars[i].depth.write(depth);
    vars[i].online.write(true);
  }
}

void ACCNAME::deactivate_PEs() {
#pragma HLS inline OFF
  for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
    vars[i].online.write(false);
  }
}

bool ACCNAME::out_fifo_filled() {
#pragma HLS inline OFF
  bool loop = false;
  //   for (int i = 0; i < PE_COUNT; i++) {
  // #pragma HLS unroll
  //     // cout << "out_fifo_filled: " << i << " " <<
  //     // vars[i].out_fifo.num_available()
  //     //      << endl;
  //     if (vars[i].out_fifo.num_available()) {
  //       loop = true;
  //       break;
  //     }
  //   }

  loop = loop || vars.vars_0.out_fifo.num_available();
  loop = loop || vars.vars_1.out_fifo.num_available();
  loop = loop || vars.vars_2.out_fifo.num_available();
  loop = loop || vars.vars_3.out_fifo.num_available();
  loop = loop || vars.vars_4.out_fifo.num_available();
  loop = loop || vars.vars_5.out_fifo.num_available();
  loop = loop || vars.vars_6.out_fifo.num_available();
  loop = loop || vars.vars_7.out_fifo.num_available();
  return loop;
}
