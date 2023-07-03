
#ifndef PE_MODULE_H
#define PE_MODULE_H

#include "acc_config.h"

SC_MODULE(PE) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<int> col_dexs_fifo_in;
  sc_fifo_in<int> dex_fifo_in;
  sc_fifo_in<char> wgt_fifo_in;
  sc_fifo_in<char> inp_fifo_in;
  // sc_fifo_out<int> out_fifo_out;
  sc_fifo_out<DATA> out_fifo_out;
  sc_fifo_in<int> temp_fifo_in;
  sc_fifo_out<int> temp_fifo_out;

  sc_in<bool> online;
  sc_in<bool> compute;
  sc_in<bool> reset_compute;
  sc_in<int> col_size;
  sc_in<int> start_addr_p;
  sc_in<int> send_len_p;
  sc_in<int> bias_data;
  sc_in<int> crf_data;
  sc_in<int> crx_data;
  sc_in<int> ra_data;

  sc_in<bool> send;
  sc_in<bool> out;
  sc_in<int> cols_per_filter;
  sc_in<int> depth;

  sc_out<bool> compute_done;
  sc_out<bool> wgt_loaded;
  sc_out<bool> out_done;
  sc_out<bool> send_done;

  sc_out<int> computeS;
  sc_out<int> sendS;
  int compute_count = 0;
  int outcount = 0;

  int Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
    sc_int<64> pl;
    sc_int<32> pr;
    sc_int<32> msk;
    sc_int<32> sm;
    if (shift > 0) {
      pl = shift;
      pr = 0;
      msk = 0;
      sm = 0;
    } else {
      // pl = 1;
      pl = 0;
      pr = -shift;
      msk = (1 << -shift) - 1;
      sm = msk >> 1;
    }
    sc_int<64> val = x * (1 << pl);
    if (val > MAX) val = MAX;
    if (val < MIN) val = MIN;
    sc_int<64> val_2 = val * qm;
    sc_int<32> temp_1;
    temp_1 = (val_2 + POS) / DIVMAX;
    if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
    sc_int<32> val_3 = temp_1;
    val_3 = val_3 >> pr;
    sc_int<32> temp_2 = temp_1 & msk;
    sc_int<32> temp_3 = (temp_1 < 0) & 1;
    sc_int<32> temp_4 = sm + temp_3;
    sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
    sc_int<32> result_32 = val_3 + temp_5;
    int res = result_32;
    return result_32;
  }

  void Compute() {

    // wgt_cols_buf needs to support ks * ks * depth / UF
    acc_dt wgt_cols_buf[PE_WGTCOLBUF_SIZE][UF];

    // wgt_col_sum needs to support ks * ks
    int wgt_col_sum[PE_WGTCOLSUMBUF_SIZE];

    // single row of input , 32 is a limiting factor
    // x rows of weights (x * depth) (x = ks * ks)
    // inp_row_buf needs to support depth / UF
    acc_dt inp_row_buf[PE_INPROWBUF_SIZE][UF];

    // Outbuf needs to support ir * ks * ks gemm outputs where ir is the number
    // of input rows
    int out_buf[PE_OUTBUF_SIZE];

    // pouts_needed is the number of output needed to be computed using current
    // input row max value is ks * ks
    int pouts_needed;
    // pout_dex is the indexes of the output needed to be computed using current
    // input row
    int pout_dex[PE_POUTDEXBUF_SIZE];

    // temy inp and wgt buffers, supports UF unrolling
    acc_dt inp_temp[UF];
    // acc_dt wgt_ud[UF];

#pragma HLS ARRAY_PARTITION variable = inp_temp complete
    // #pragma HLS ARRAY_PARTITION variable = wgt_ud complete

    compute_done.write(false);
    wgt_loaded.write(false);
    computeS.write(0);
    wait();
    while (1) {

      computeS.write(1);
      wgt_loaded.write(false);
      DWAIT();
      while (!online.read()) wait();

      // load weights
      int i = 0;
      computeS.write(2);
      for (int c = 0; c < cols_per_filter; c++) {
        wgt_col_sum[c] = col_dexs_fifo_in.read();
        for (int d = 0; d < depth; d++) {
          for (int u = 0; u < UF; u++) {
            wgt_cols_buf[i][u] = wgt_fifo_in.read();
          }
          i++;
        }
      }
      wait();
      computeS.write(3);
      wgt_loaded.write(true);
      wait();

      computeS.write(4);
      // PE is active
      while (online) {
        computeS.write(5);

        while (!compute) {
          computeS.write(51);
          if (!online) break;
          wait();
        }
        if (!online) break;

        // loads inputs
        for (int d = 0; d < depth; d++) {
          for (int u = 0; u < UF; u++) {
            inp_row_buf[d][u] = inp_fifo_in.read();
          }
        }

        computeS.write(6);
        pouts_needed = col_size.read();
        int pouts_count = pouts_needed;
        for (int i = 0; i < pouts_needed; i++)
          pout_dex[i] = col_dexs_fifo_in.read();

        computeS.write(7);
        for (int d = 0; d < depth; d++) {
          // load input
          for (int u = 0; u < UF; u++) {
            inp_temp[u] = inp_row_buf[d][u];
          }
          for (int i = 0; i < pouts_needed; i++) {
            int ocol = pout_dex[i];
            int ocol_s = ocol * depth;
            int sum = 0;
            for (int u = 0; u < UF; u++) {
              acc_dt wt1 = wgt_cols_buf[ocol_s + d][u];
              sum += wt1 * inp_temp[u];
            }
            out_buf[i] += sum;
            // save to temp output
          }
        }

        computeS.write(8);
        for (int i = 0; i < pouts_needed; i++) {
          int ocol = pout_dex[i];
          int output = out_buf[i] + wgt_col_sum[ocol];
          temp_fifo_out.write(output);
        }
        wait();
        for (int i = 0; i < pouts_needed; i++) {
          out_buf[i] = 0;
        }

        compute_done.write(true);
        while (!reset_compute) {
          computeS.write(11);
          DWAIT();
        }

        compute_done.write(false);
        computeS.write(12);
      }
      computeS.write(13);
      DWAIT();
    }
  }

  void Out() {
    DATA last = {4000, 1};
    DATA d = {0, 0};
    int acc_store[PE_ACC_BUF_SIZE];
    int pouts_count = 0;
    int start_addr = 0;
    int send_len = 0;
    int bias;
    int crf;
    sc_int<32> crx;
    int ra = 0;
    out_done.write(false);
    send_done.write(false);
    sendS.write(0);
    outcount = 0;
    wait();
    while (1) {
      while (!out.read() && !send.read()) {
        // sendS.write(1);
        DWAIT();
      }
      // sendS.write(2);
      if (send) {
        // sendS.write(3);
        start_addr = start_addr_p.read();
        send_len = send_len_p.read();
        bias = bias_data.read();
        crf = crf_data.read();
        crx = crx_data.read();
        ra = ra_data.read();
        // sendS.write(31);
        sendS.write(send_len);
        for (int i = 0; i < send_len; i++) {
          int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
          int qm_ret = ra + Quantised_Multiplier(acc_store[dex] + bias, crf,
                                                 crx.range(7, 0));
          if (qm_ret > MAX8) qm_ret = MAX8;
          else if (qm_ret < MIN8) qm_ret = MIN8;
          d.data = qm_ret;
          // d.data = acc_store[dex];
          out_fifo_out.write(d);
          if (i + 1 == send_len) {
            out_fifo_out.write(last);
          }
          sendS.write(32);
        }
        for (int i = 0; i < send_len; i++) {
          int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
          acc_store[dex] = 0;
        }
        // sendS.write(33);
        send_done.write(true);
      }

      if (out) {
        // sendS.write(4);
        pouts_count = col_size.read();
        for (int i = 0; i < pouts_count; i++) {
          int dex = dex_fifo_in.read() % PE_ACC_BUF_SIZE;
          int out = temp_fifo_in.read();
          acc_store[dex] += out;
        }
        out_done.write(true);
      }

      wait();
      while (out || send) {
        // sendS.write(5);
        DWAIT();
      }
      out_done.write(false);
      send_done.write(false);
      DWAIT();
    }
  }

  void init(sc_in<bool> & clock, sc_in<bool> & reset, PE_vars & vars) {
    this->clock(clock);
    this->reset(reset);
    this->col_dexs_fifo_in(vars.col_dexs_fifo);
    this->dex_fifo_in(vars.dex_fifo);
    this->inp_fifo_in(vars.inp_fifo);
    this->wgt_fifo_in(vars.wgt_fifo);
    this->out_fifo_out(vars.out_fifo);
    this->temp_fifo_in(vars.temp_fifo);
    this->temp_fifo_out(vars.temp_fifo);

    this->online(vars.online);
    this->compute(vars.compute);
    this->reset_compute(vars.reset_compute);
    this->col_size(vars.col_size);
    this->start_addr_p(vars.start_addr_p);
    this->send_len_p(vars.send_len_p);
    this->bias_data(vars.bias_data);
    this->crf_data(vars.crf_data);
    this->crx_data(vars.crx_data);
    this->ra_data(vars.ra_data);

    this->send(vars.send);
    this->out(vars.out);
    this->cols_per_filter(vars.cols_per_filter);
    this->depth(vars.depth);
    this->compute_done(vars.compute_done);
    this->wgt_loaded(vars.wgt_loaded);
    this->out_done(vars.out_done);
    this->send_done(vars.send_done);

    this->computeS(vars.computeS);
    this->sendS(vars.sendS);
  }

  SC_HAS_PROCESS(PE);

  PE(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Out, clock);
    reset_signal_is(reset, true);
  }
};

// struct var_array {
//   PE_vars vars_0;
//   PE_vars vars_1;
//   PE_vars vars_2;
//   PE X1;
//   PE X2;
//   PE X3;

// #ifndef __SYNTHESIS__
//   var_array()
//       : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), X1("X1"), X2("X2"),
//         X3("X3") {}
// #else
//   var_array()
//       : vars_0(16), vars_1(16), vars_2(16), X1("X1"), X2("X2"), X3("X3") {}
// #endif
//   PE_vars &operator[](int index) {
//     if (index == 0) return vars_0;
//     else if (index == 1) return vars_1;
//     else if (index == 2) return vars_2;

//     return vars_0;
//   }

//   void init(sc_in<bool> &clock, sc_in<bool> &reset) {
//     X1.init(clock, reset, vars_0);
//     X2.init(clock, reset, vars_1);
//     X3.init(clock, reset, vars_2);
//   }
// };

// create var_array with 8 PEs
struct var_array {
  PE_vars vars_0;
  PE_vars vars_1;
  PE_vars vars_2;
  PE_vars vars_3;
  PE_vars vars_4;
  PE_vars vars_5;
  PE_vars vars_6;
  PE_vars vars_7;
  PE X1;
  PE X2;
  PE X3;
  PE X4;
  PE X5;
  PE X6;
  PE X7;
  PE X8;

#ifndef __SYNTHESIS__
  var_array()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3),
        vars_4(16, 4), vars_5(16, 5), vars_6(16, 6), vars_7(16, 7), X1("X1"),
        X2("X2"), X3("X3"), X4("X4"), X5("X5"), X6("X6"), X7("X7"), X8("X8") {}
#else
  var_array()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), vars_4(16), vars_5(16),
        vars_6(16), vars_7(16), X1("X1"), X2("X2"), X3("X3"), X4("X4"),
        X5("X5"), X6("X6"), X7("X7"), X8("X8") {}
#endif

  PE_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else if (index == 2) return vars_2;
    else if (index == 3) return vars_3;
    else if (index == 4) return vars_4;
    else if (index == 5) return vars_5;
    else if (index == 6) return vars_6;
    else if (index == 7) return vars_7;

    return vars_0;
  }

  DATA get(int index) {
    if (index == 0) return vars_0.out_fifo.read();
    else if (index == 1) return vars_1.out_fifo.read();
    else if (index == 2) return vars_2.out_fifo.read();
    else if (index == 3) return vars_3.out_fifo.read();
    else if (index == 4) return vars_4.out_fifo.read();
    else if (index == 5) return vars_5.out_fifo.read();
    else if (index == 6) return vars_6.out_fifo.read();
    else if (index == 7) return vars_7.out_fifo.read();
    return vars_0.out_fifo.read();
  };

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    X1.init(clock, reset, vars_0);
    X2.init(clock, reset, vars_1);
    X3.init(clock, reset, vars_2);
    X4.init(clock, reset, vars_3);
    X5.init(clock, reset, vars_4);
    X6.init(clock, reset, vars_5);
    X7.init(clock, reset, vars_6);
    X8.init(clock, reset, vars_7);
  }
};

#endif // PE_MODULE_H

// while (out_data_stored)
//   wait();
// computeS.write(9);
// for (int i = 0; i < pouts_needed; i++) {
//   int ocol = pout_dex[i];
//   out_buf[i] += wgt_col_sum[ocol];
// }
// compute_data_stored.write(true);
// computeS.write(10);
// while (!out_data_stored)
//   wait();
// compute_data_stored.write(false);

// while (!compute_data_stored)
//   wait();
// sendS.write(42);
// wait();
// for (int i = 0; i < pouts_count; i++) {
//   int dex = dex_fifo_in.read() % PE_OUT_BUF_LEN;
//   acc_store[dex] += out_buf[i];
//   out_buf[i] = 0;
// }
// wait();
// sendS.write(43);
// out_data_stored.write(true);
// while (compute_data_stored)
//   wait();
// sendS.write(44);
// out_data_stored.write(false);
