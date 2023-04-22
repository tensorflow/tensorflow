#ifndef VTA_Driver_H
#define VTA_Driver_H
#include <systemc.h>
#include "../vta.h"


SC_MODULE(VTA_Driver) {
  sc_in<bool> clock;
  sc_in <bool>  reset;

  sc_in<unsigned int> vtaS;
  sc_out<unsigned int> insn_count;
  sc_out<unsigned int> ins_addr;
  sc_out<unsigned int> uops_addr;
  sc_out<unsigned int> input_addr;
  sc_out<unsigned int> weight_addr;
  sc_out<unsigned int> bias_addr;
  sc_out<unsigned int> output_addr;


  AXI4M_bus_port<unsigned long long> insns; // only 64 bits should be 128, 1 << VTA_LOG_INS_WIDTH
  AXI4M_bus_port<unsigned int> uops; // (1 << VTA_LOG_UOP_WIDTH)
  AXI4M_bus_port<unsigned long long> data; // (1 << VTA_LOG_BUS_WIDTH)



  void Driver(){
  	bool on = false;
  	while(true){
  		on = !on;
//  		insn_count.write(1);
//  		ins_addr.write(0);
//  		uops_addr.write(0);
//  		input_addr.write(0);
//  		weight_addr.write(0);
//  		bias_addr.write(0);
//  		output_addr.write(0);
  		wait();
  	}
  };

  SC_HAS_PROCESS(VTA_Driver);
  VTA_Driver(sc_module_name name_) :sc_module(name_)
  , insns ("insns") , uops ("uops") , data ("data")

  {

    SC_CTHREAD(Driver, clock.pos());
    reset_signal_is(reset,true);

  }
};

#endif /* VTA_Driver_H */
