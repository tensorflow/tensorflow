#ifndef AXIS_ENGINE_H
#define AXIS_ENGINE_H

#include "sysc_types.h"

SC_MODULE(AXIS_ENGINE) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_fifo_in<DATA> dout1;
  sc_fifo_out<DATA> din1;
  bool send;
  bool recv;
  int id;

  void DMA_MMS2() {
    int initial_free = din1.num_free();
    while (1) {
      while (!send) wait();
      for (int i = 0; i < input_len; i++) {
        int d = DMA_input_buffer[i + input_offset];
        din1.write({d, 1});
        wait();
      }
      while (initial_free != din1.num_free()) wait();
      send = false;
      sc_pause();
      wait();
    }
  };

  void DMA_S2MM() {
    while (1) {
      while (!recv) wait();
      bool last = false;
      int i = 0;
      do {
        DATA d = dout1.read();
        while (i >= output_len) wait();
        last = d.tlast;
        DMA_output_buffer[output_offset + i++] = d.data;
        wait();
      } while (!last);
      output_len = i;
      recv = false;
      // // To ensure wait_send() does not evoke the sc_pause
      // while (send)
      //   wait(2);
      wait();
      sc_pause();
      wait();
    }
  };

  SC_HAS_PROCESS(AXIS_ENGINE);

  AXIS_ENGINE(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(DMA_MMS2, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(DMA_S2MM, clock.pos());
    reset_signal_is(reset, true);
  }

  int* DMA_input_buffer;
  int* DMA_output_buffer;

  int input_len;
  int input_offset;

  int output_len;
  int output_offset;
};

#endif