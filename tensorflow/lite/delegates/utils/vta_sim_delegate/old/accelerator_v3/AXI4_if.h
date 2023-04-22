#include "hls_bus_if.h"

template<typename _VHLS_DT>
class AXI4M_bus_port: public hls_bus_port<_VHLS_DT>
{
    typedef hls_bus_port<_VHLS_DT> Base;
public:

    AXI4M_bus_port() { 
    }

    explicit AXI4M_bus_port( const char* name_ ):Base(name_)
    {
    }

};


