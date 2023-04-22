/* -*- sysc -*-*/
/*
#-  (c) Copyright 2011-2018 Xilinx, Inc. All rights reserved.
#-
#-  This file contains confidential and proprietary information
#-  of Xilinx, Inc. and is protected under U.S. and
#-  international copyright and other intellectual property
#-  laws.
#-
#-  DISCLAIMER
#-  This disclaimer is not a license and does not grant any
#-  rights to the materials distributed herewith. Except as
#-  otherwise provided in a valid license issued to you by
#-  Xilinx, and to the maximum extent permitted by applicable
#-  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
#-  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
#-  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
#-  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
#-  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
#-  (2) Xilinx shall not be liable (whether in contract or tort,
#-  including negligence, or under any other theory of
#-  liability) for any loss or damage of any kind or nature
#-  related to, arising under or in connection with these
#-  materials, including for any direct, or any indirect,
#-  special, incidental, or consequential loss or damage
#-  (including loss of data, profits, goodwill, or any type of
#-  loss or damage suffered as a result of any action brought
#-  by a third party) even if such damage or loss was
#-  reasonably foreseeable or Xilinx had been advised of the
#-  possibility of the same.
#-
#-  CRITICAL APPLICATIONS
#-  Xilinx products are not designed or intended to be fail-
#-  safe, or for use in any application requiring fail-safe
#-  performance, such as life-support or safety devices or
#-  systems, Class III medical devices, nuclear facilities,
#-  applications related to the deployment of airbags, or any
#-  other applications that could lead to death, personal
#-  injury, or severe property or environmental damage
#-  (individually and collectively, "Critical
#-  Applications"). Customer assumes the sole risk and
#-  liability of any use of Xilinx products in Critical
#-  Applications, subject only to applicable laws and
#-  regulations governing limitations on product liability.
#-
#-  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
#-  PART OF THIS FILE AT ALL TIMES. 
#- ************************************************************************

 *
 *
 */

#ifndef _AP_BUS_IF
#define _AP_BUS_IF

#define _hls_bus_addrt unsigned

template<typename _VHLS_DT>
class hls_bus_if : public sc_interface
{
public:
   hls_bus_if(const char* name_ = ""){
   }

   virtual _VHLS_DT& read(_hls_bus_addrt addr)=0;
   virtual void read(_hls_bus_addrt addr, _VHLS_DT *data)=0;
   virtual void burst_read(_hls_bus_addrt addr, int size, _VHLS_DT *data)=0;
   virtual void write(_hls_bus_addrt addr, _VHLS_DT *data)=0;
   virtual void burst_write(_hls_bus_addrt addr, int size, _VHLS_DT *data)=0;
};


//----------------------------------------------------------
// hls_bus_port
//----------------------------------------------------------
template<typename _VHLS_DT>
class hls_bus_port : public sc_port<hls_bus_if<_VHLS_DT> >
{
    // typedefs
    //typedef _VHLS_DT                                             _VHLS_DT;
    typedef hls_bus_if<_VHLS_DT>       if_type;
    typedef sc_port<if_type, 1,SC_ONE_OR_MORE_BOUND>      base_type;
    typedef hls_bus_port<_VHLS_DT>     this_type;

    typedef if_type                                       in_if_type;
    typedef base_type                                     in_port_type;

public:

    hls_bus_port() {
     #ifndef __RTL_SIMULATION__
        cout<<"@W [SIM] Please add name for your hls_bus_port, or RTL simulation will fail."<<endl; 
     #endif
    }
    explicit hls_bus_port( const char* name_ ) { }

    void reset() {}

    _VHLS_DT &read(_hls_bus_addrt addr) {
       return (*this)->read(addr);
    }

    void burst_read(_hls_bus_addrt addr, int size, _VHLS_DT *data) {
       (*this)->burst_read(addr, size, data);
    }

    void write(_hls_bus_addrt addr, _VHLS_DT *data) {
        (*this)->write(addr, data);
    }

    void burst_write(_hls_bus_addrt addr, int size, _VHLS_DT *data) {
       (*this)->burst_write(addr, size, data);
    }
};


//-----------------------------
// bus channel
//-----------------------------
template <typename data_type>
class hls_bus_chn
  :public hls_bus_if<data_type>
  ,public sc_module
{
  private:
    //data_type  mem[END_ADDR - START_ADDR];
    data_type  *mem;
    unsigned int m_start_addr;
    unsigned int m_end_addr;
    std::string name;

  public:
    SC_HAS_PROCESS(hls_bus_chn);
    hls_bus_chn(sc_module_name _name
              ,unsigned int start_addr = 0
              ,unsigned int end_addr = 1024)
    : sc_module(_name)
    , name(_name)
    , m_start_addr(start_addr)
    , m_end_addr(end_addr)
    {
       sc_assert(m_start_addr <= m_end_addr);

       unsigned int size = (m_end_addr-m_start_addr+1);

       mem = new data_type [size];
       for (unsigned int i = 0; i < size; ++i)
         mem[i] = i;
    }

    ///
    /// bus read
    ///
    data_type& read(_hls_bus_addrt addr)
    {
    #ifdef DUMP_BUS_OP
        cout <<"[bus wraper] "<<name<<": read mem["<<addr<<"] = "<< mem[addr] <<endl;
    #endif
        return mem[addr];
    }

    void read(_hls_bus_addrt addr, data_type *data)
    {
        *data = mem[addr];
    #ifdef DUMP_BUS_OP
        cout <<"[bus wraper] "<<name<<": read mem["<<addr<<"] = "<< mem[addr] <<endl;
    #endif
    }

    ///
    /// burst read
    ///
    void burst_read(_hls_bus_addrt addr, int size, data_type *data)
    {
       for(int i=0; i<size; i++)
       {
          data[i] = mem[addr+i];
        //  cout<<"[bus wraper] "<< name <<": read mem["<<addr+i<<"] = "<< data[i] <<  " ||  address: " <<  &(data[i]) <<endl;
    #ifdef DUMP_BUS_OP
          cout<<"[bus wraper] "<< name <<": read mem["<<addr+i<<"] = "<< data[i] <<endl;
    #endif
       }
    }

    ///
    /// bus write
    ///
    void write(_hls_bus_addrt addr, data_type *data)
    {
        mem[addr] = *data;
    #ifdef DUMP_BUS_OP
        cout <<"[bus wraper] "<< name <<": write mem["<<addr<<"] = "<< mem[addr] <<endl;
    #endif
    }

    ///
    /// burst write
    ///
    void burst_write(_hls_bus_addrt addr, int size, data_type *data)
    {
       for(int i=0; i<size; i++)
       {
          mem[addr+i] = data[i];
        //  cout <<"[bus wraper] "<<name<<": write mem["<<addr+i<<"] = "<< mem[addr+i] <<endl;
    #ifdef DUMP_BUS_OP
          cout <<"[bus wraper] "<<name<<": write mem["<<addr+i<<"] = "<< mem[addr+i] <<endl;
    #endif
       }
    }
};

#undef _hls_bus_addrt
#endif


