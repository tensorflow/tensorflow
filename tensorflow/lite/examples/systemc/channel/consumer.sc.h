// Created by Nicolas Agostini

#ifndef CONSUMER_H
#define CONSUMER_H

#include <systemc/systemc.h>
#include "stack_if.sc.h"

class consumer : public sc_module {
  public:
    sc_port<stack_read_if>  in;
    sc_in<bool> Clock;

    void do_reads() {
      int i =0;
      char TestString [32];
      while (true) {
        wait(); // for clock
        if (in->nb_read(TestString[i]))
          std::cout<< "R " << TestString[i] << " at "
            << sc_time_stamp() << endl;
        i = (i+1) % 32;
      }
    }

    SC_CTOR(consumer) {
      SC_THREAD(do_reads);
      sensitive << Clock.pos();
    }
};

#endif // CONSUMER_H
