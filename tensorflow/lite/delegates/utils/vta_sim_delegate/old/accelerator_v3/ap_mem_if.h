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

#ifndef _AP_MEM_IF_H
#define _AP_MEM_IF_H

#include <systemc.h>

//#define USER_DEBUG_MEMORY

/* ap_mem_port Types */
enum ap_mem_port_type {
    RAM_1P = 0,
    RAM1P = RAM_1P,
    RAM_2P = 1,
    RAM2P = RAM_2P,
    RAM_T2P = 2,
    RAMT2P = RAM_T2P,
    ROM_1P = 3,
    ROM1P = ROM_1P,
    ROM_2P = 4,
    ROM2P = ROM_2P,
};

//=================================================================
//========================  ap_mem ports  =========================
//=================================================================
//----------------------------------------------------------
// ap_mem_if
//----------------------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_mem_if : public sc_interface
{
public:

    // read the current value
    virtual _hls_mem_dt& read(const _hls_mem_addrt addr)  = 0;
    virtual void write(const _hls_mem_addrt addr, const _hls_mem_dt data)  = 0;

protected:

    // constructor
    ap_mem_if()
	{}

private:
    // disabled
    ap_mem_if( const ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& );
    ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& operator = ( const ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& );
};

//----------------------------------------------------------
// ap_mem_port
//----------------------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_mem_port : public sc_port<ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType> >
{
    // typedefs
    //typedef T                                             _hls_mem_dt;
    typedef ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>       if_type;
    typedef sc_port<if_type,1,SC_ONE_OR_MORE_BOUND>       base_type;
    typedef ap_mem_port<_hls_mem_dt, _hls_mem_addrt, t_size, portType>     this_type;

    typedef if_type                                       in_if_type;
    typedef base_type                                     in_port_type;
    _hls_mem_addrt ADDR_tmp;

public:

    ap_mem_port() {
     #ifndef __RTL_SIMULATION__
        cout<<"@W [SIM] Please add name for your ap_mem_port, or RTL simulation will fail."<<endl; 
     #endif
    }
    explicit ap_mem_port( const char* name_ ) { }

    void reset() {}

    _hls_mem_dt &read(const _hls_mem_addrt addr) {
       return (*this)->read(addr);
    }

    void write(const _hls_mem_addrt addr, const _hls_mem_dt data) {
        (*this)->write(addr, data);
    }

    ap_mem_port& operator [] (const _hls_mem_addrt addr) {
        //return (*this)->read(addr);
        ADDR_tmp = addr;
        return *this;
    }

    void operator = (_hls_mem_dt data)
    {
        (*this)->write(ADDR_tmp, data);
    }

    operator _hls_mem_dt ()
    {
        return (*this)->read(ADDR_tmp);
    }
};

//----------------------------------------------------------
// ap_mem_chn
//----------------------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_mem_chn :
    public ap_mem_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>,
    public sc_prim_channel
{
private:
    _hls_mem_dt mArray[2][t_size];
    int mInitiatorBank;
    int mTargetCurBank;
    int mCurBank;
    std::string mem_name;
    
public:
    explicit ap_mem_chn( const char* name="ap_mem_chn" ) : mCurBank(0), mem_name(name) { }

    virtual ~ap_mem_chn() { };
    
    _hls_mem_dt& read(const _hls_mem_addrt addr) {
#ifdef USER_DEBUG_MEMORY
        cout<<"mem read : "<< mem_name <<"["<<addr<<"] = "<<mArray[mCurBank][addr]<<endl;
#endif
        return mArray[mCurBank][addr];
    }

    void write(const _hls_mem_addrt addr, const _hls_mem_dt data) {
        mArray[mCurBank][addr] = data;
#ifdef USER_DEBUG_MEMORY
        cout<<"mem write: "<< mem_name <<"["<<addr<<"] = "<<mArray[mCurBank][addr]<<endl;
#endif
    }

    _hls_mem_dt& operator [] (const _hls_mem_addrt& addr) {
        return mArray[mCurBank][addr];
    }

    //int put() { mCurBank++; mCurBank %= 2; }
    //int get() { }
};

//=================================================================
//======================== ap_pingpong_if =========================
//=================================================================
//--------------------------------------------
// ap_pingpong_if
//--------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_pingpong_if : public sc_interface
{
public:
    virtual bool put_is_ready() = 0;
    virtual void put_request()  = 0;
    virtual void put_release()  = 0;
    virtual _hls_mem_dt put_read(const _hls_mem_addrt addr)  = 0;
    virtual void put_write(const _hls_mem_addrt addr, const _hls_mem_dt data)  = 0;
    virtual bool get_is_ready() = 0;
    virtual void get_request()  = 0;
    virtual void get_release()  = 0;
    virtual _hls_mem_dt get_read(const _hls_mem_addrt addr)  = 0;
    virtual void get_write(const _hls_mem_addrt addr, const _hls_mem_dt data)  = 0;

protected:

    // constructor
    ap_pingpong_if()
	{}

private:

    // disabled
    ap_pingpong_if( const ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& );
    ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& operator = ( const ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>& );
};

//--------------------------------------------
// ap_pingpong_get
//--------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_pingpong_get: public sc_port<ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType> >
{
    //typedef sc_port_b<ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType> > Get_Base;
public:
    explicit ap_pingpong_get( const char* name_ ) { }
    
    /// functions
    void reset() {  }
    
    bool is_ready()
    {
       return (*this)->get_is_ready();
    }
    
    void request()
    {
       (*this)->get_request();
    }

    void release()
    {
       (*this)->get_release();
    }

    _hls_mem_dt read(_hls_mem_addrt addr) {
        return (*this)->get_read(addr);
    }

    void write(_hls_mem_addrt addr, _hls_mem_dt data) {
        (*this)->get_write(addr, data);
    }
};


//--------------------------------------------
// ap_pingpong_put
//--------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_pingpong_put: public sc_port<ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType> >
{
public:
    explicit ap_pingpong_put( const char* name_ ) { }

    /// functions
    void reset() {  }

    bool is_ready()
    {
       return (*this)->put_is_ready();
    }
    
    void request()
    {
       (*this)->put_request();
    }

    void release()
    {
       (*this)->put_release();
    }

    _hls_mem_dt read(_hls_mem_addrt addr) {
        return (*this)->put_read(addr);
    }

    void write(_hls_mem_addrt addr, _hls_mem_dt data) {
        (*this)->put_write(addr, data);
    }
};

//--------------------------------------------
// ap_pingpong_chn
//--------------------------------------------
template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_pingpong_chn
  :public ap_pingpong_if<_hls_mem_dt, _hls_mem_addrt, t_size, portType>
  ,public sc_prim_channel
{
private:
    _hls_mem_dt mArray[2][t_size];
    int mInitiatorBank;
    int mTargetBank;
    std::string mem_name;
    bool cur_bankflag[2];
    bool new_bankflag[2];

protected:
    virtual void update()
    {
       cur_bankflag[0] = new_bankflag[0];
       cur_bankflag[1] = new_bankflag[1];
    }
    
public:
    explicit ap_pingpong_chn( const char* name="ap_pingpong_chn" )
    : mem_name(name)
    , mInitiatorBank(0)
    , mTargetBank(0)
    {
       cur_bankflag[0] = 0;
       cur_bankflag[1] = 0;
       new_bankflag[0] = 0;
       new_bankflag[1] = 0;
    }

    virtual ~ap_pingpong_chn() { };
    
    /// Initiator/put APIs
    bool put_is_ready()
    {
       return (!cur_bankflag[mInitiatorBank]);
    }

    void put_request() { do { wait(); } while(!put_is_ready() ); }

    void put_release() 
    {
        new_bankflag[mInitiatorBank] = 1;
        mInitiatorBank++;
        mInitiatorBank %= 2;
        request_update();
    }

    _hls_mem_dt put_read(const _hls_mem_addrt addr) {
#ifdef USER_DEBUG_MEMORY
        cout<<"Initor read : "<< mem_name <<"["<<addr<<"] = "<<mArray[mInitiatorBank][addr]<<endl;
#endif
        return mArray[mInitiatorBank][addr];
    }

    void put_write(const _hls_mem_addrt addr, const _hls_mem_dt data) {
        mArray[mInitiatorBank][addr] = data;
#ifdef USER_DEBUG_MEMORY
        cout<<"Initor write: "<< mem_name <<"["<<addr<<"] = "<<mArray[mInitiatorBank][addr]<<endl;
#endif
    }

    /// Target/get APIs
    bool get_is_ready()
    {
       return (cur_bankflag[mTargetBank]);
    }

    void get_request()
    {
       do { wait(); } while(!get_is_ready() );
    }

    void get_release()
    {
        new_bankflag[mTargetBank] = 0;
        mTargetBank++;
        mTargetBank %= 2;
        request_update();
    }

    _hls_mem_dt get_read(const _hls_mem_addrt addr) {
        if(!get_is_ready()) { return 0; }
#ifdef USER_DEBUG_MEMORY
        cout<<"Target read : "<< mem_name <<"["<<addr<<"] = "<<mArray[mTargetBank][addr]<<endl;
#endif
        return mArray[mTargetBank][addr];
    }

    void get_write(const _hls_mem_addrt addr, const _hls_mem_dt data) {
        if(!get_is_ready()) { return; }
        mArray[mTargetBank][addr] = data;
#ifdef USER_DEBUG_MEMORY
        cout<<"Target write: "<< mem_name <<"["<<addr<<"] = "<<mArray[mTargetBank][addr]<<endl;
#endif
    }
};

template<typename _hls_mem_dt, typename _hls_mem_addrt, int t_size, ap_mem_port_type portType=RAM1P>
class ap_pingpong
{
  public:
    typedef ap_pingpong_put<_hls_mem_dt, _hls_mem_addrt, t_size, portType> put;
    typedef ap_pingpong_get<_hls_mem_dt, _hls_mem_addrt, t_size, portType> get;
    typedef ap_pingpong_chn<_hls_mem_dt, _hls_mem_addrt, t_size, portType> chn;
};
#endif /* Header Guard */



