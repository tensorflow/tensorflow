// Created by Nicolas Agostini

#ifndef PRODUCER_H
#define PRODUCER_H

#include <systemc/systemc.h>
#include "stack_if.sc.h"

class producer : public sc_module {
  public:
    sc_port<stack_write_if> out;
    sc_in<bool> Clock;

    void do_writes() {
      int i =0;
      const char * TestString = "Hallo,    This will be Reversed";
      while (true) {
        wait(); // for clock
        if (out->nb_write(TestString[i]))
          std::cout<< "W " << TestString[i] << " at "
            << sc_time_stamp() << endl;
        i = (i+1) % 32;
      }
    }

    SC_CTOR(producer) {
      SC_THREAD(do_writes);
      sensitive << Clock.pos();
    }
};

#endif // PRODUCER_H
