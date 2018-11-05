import numpy as np
import re

# This should be run from make/targets/apollo3evb

def new_data_to_array(fn):
    vals = []
    with open(fn) as f:
        for n, line in enumerate(f):
            if n is not 0:
                vals.extend([str(int(v, 16)) for v in line.split()])
            
    return ','.join(vals)

def replace_data(fn_old, new_data):
    patt = '(?<=\{).+?(?=\})'
    with open(fn_old,'r') as f:
        str_old = f.read()
        str_new = re.sub(patt, new_data, str_old, flags=re.DOTALL)
    with open(fn_old,'w') as f:
        f.write(str_new)
        
            
yes_old = '../../../../examples/micro_speech/CMSIS/yes_power_spectrum_data.cc'
no_old = '../../../../examples/micro_speech/CMSIS/no_power_spectrum_data.cc'
yes_new = 'yes_calculated_data.txt'
no_new = 'no_calculated_data.txt'

yes_new_vals = new_data_to_array(yes_new)
no_new_vals = new_data_to_array(no_new)

replace_data(yes_old, yes_new_vals)
replace_data(no_old, no_new_vals)
