import os
import re

def parse_all_ops(path):
            
    if not os.path.exists(path):
        raise Exception("cannot find all_ops_resolver.cc at path {}".format(path))
        
    function_list =[]
    with open(path, 'r') as fid:
        for line in fid.readlines():
            if "BuiltinOperator" in line:
                s = line.lstrip()[27:].split(',')
                function_list.extend([s[0]])
    
    return function_list

def get_name(layer):
    name = layer.get_config()['name']
    if "dense" in name:
        return 'FULLY_CONNECTED'
    
    return name.upper()
    
def get_activation(layer):
    activation = layer.get_config().get('activation', None)
    
    if activation:    
        return activation.upper()
    
    return None
    
    
def gen_model_functions(function_list):
    
    template = "  resolver.AddBuiltin(tflite::BuiltinOperator_{0}, tflite::ops::micro::Register_{0}());"
    M = []
    
    for function in function_list:
        M.append(template.format(function))
    
    return M
        
    
def get_model_functions(model, function_list):
    used_functions = set()
    for layer in model.layers:
        name = get_name(layer)
        activation = get_activation(layer)
        
        if name in function_list:
            used_functions.add(name)
        else:
            raise Exception("model uses {} which is not a supported function in tf micro".format(name))
            
        if activation in function_list:
            used_functions.add(activation)
        else:
            raise Exception("model uses {} which is not a supported function in tf micro".format(name))
            
    
    return used_functions

def gen_micro_mutable_ops_resolver_add(model, all_ops_path):
    function_list = parse_all_ops(all_ops_path)
    used_functions = get_model_functions(model, function_list)
    return gen_model_functions(used_functions)


def fill_template_file(model=None, template_path='./micro_api.cc.tpl', output='./micro_api.cc', all_ops_path='../../kernels/all_ops_resolver.cc'):
    
    default_template = {"micro_mutable_ops_resolver":[" static tflite::ops::micro::AllOpsResolver resolver;"],
                        "micro_mutable_ops_resolver_header":['#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"']}
    
    if model:
        default_template["micro_mutable_ops_resolver"] = gen_micro_mutable_ops_resolver_add(model, all_ops_path)
        default_template["micro_mutable_ops_resolver_header"] = ['#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"']
        
    with open(template_path, 'r') as fid:
        output_str = ''.join(fid.readlines())
        for key, value in default_template.items():
            output_str = re.sub( r"//FILL_{}\b".format(key.upper()),
                                "\n".join(value),
                                output_str,
                               )
            
    with open(output, 'w') as out:
        out.write(output_str)


if __name__ == "__main__":
    fill_template_file()

    