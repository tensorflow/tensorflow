# -*- python -*-
"""
 add a repo_generator rule for tensorrt

"""

_TENSORRT_INSTALLATION_PATH="TENSORRT_INSTALL_PATH"
_TF_TENSORRT_VERSION="TF_TENSORRT_VERSION"

def _is_trt_enabled(repo_ctx):
    if "TF_NEED_TENSORRT" in repo_ctx.os.environ:
        enable_trt = repo_ctx.os.environ["TF_NEED_TENSORRT"].strip()
        return enable_trt == "1"
    return False

def _dummy_repo(repo_ctx):

    repo_ctx.template("BUILD",Label("//third_party/tensorrt:BUILD.tpl"),
                      {"%{tensorrt_lib}":"","%{tensorrt_genrules}":""},
                      False)
    repo_ctx.template("build_defs.bzl",Label("//third_party/tensorrt:build_defs.bzl.tpl"),
                      {"%{trt_configured}":"False"},False)
    repo_ctx.file("include/NvUtils.h","",False)
    repo_ctx.file("include/NvInfer.h","",False)

def _trt_repo_impl(repo_ctx):
    """
    Implements local_config_tensorrt
    """

    if not _is_trt_enabled(repo_ctx):
        _dummy_repo(repo_ctx)
        return
    trt_libdir=repo_ctx.os.environ[_TENSORRT_INSTALLATION_PATH]
    trt_ver=repo_ctx.os.environ[_TF_TENSORRT_VERSION]
# if deb installation
# once a standardized installation between tar and deb
# is done, we don't need this
    if trt_libdir == '/usr/lib/x86_64-linux-gnu':
        incPath='/usr/include/x86_64-linux-gnu'
        incname='/usr/include/x86_64-linux-gnu/NvInfer.h'
    else:
        incPath=str(repo_ctx.path("%s/../include"%trt_libdir).realpath)
        incname=incPath+'/NvInfer.h'
    if len(trt_ver)>0:
        origLib="%s/libnvinfer.so.%s"%(trt_libdir,trt_ver)
    else:
        origLib="%s/libnvinfer.so"%trt_libdir        
    objdump=repo_ctx.which("objdump")
    if objdump == None:
        if len(trt_ver)>0:
            targetlib="lib/libnvinfer.so.%s"%(trt_ver[0])
        else:
            targetlib="lib/libnvinfer.so"
    else:
        soname=repo_ctx.execute([objdump,"-p",origLib])
        for l in soname.stdout.splitlines():
            if "SONAME" in l:
                lib=l.strip().split(" ")[-1]
                targetlib="lib/%s"%(lib)
    
    if len(trt_ver)>0:
        repo_ctx.symlink(origLib,targetlib)
    else:
        repo_ctx.symlink(origLib,targetlib)
    grule=('genrule(\n    name = "trtlinks",\n'+
           '    outs = [\n    "%s",\n    "include/NvInfer.h",\n    "include/NvUtils.h",\n     ],\n'%targetlib +
           '    cmd="""ln -sf %s $(@D)/%s '%(origLib,targetlib) +
           '&&\n    ln -sf %s $(@D)/include/NvInfer.h '%(incname) +
           '&&\n    ln -sf %s/NvUtils.h $(@D)/include/NvUtils.h""",\n)\n'%(incPath))
    repo_ctx.template("BUILD",Label("//third_party/tensorrt:BUILD.tpl"),
                      {"%{tensorrt_lib}":'"%s"'%targetlib,"%{tensorrt_genrules}":grule},
                      False)
    repo_ctx.template("build_defs.bzl",Label("//third_party/tensorrt:build_defs.bzl.tpl"),
                      {"%{trt_configured}":"True"},False)

trt_repository=repository_rule(
    implementation= _trt_repo_impl,
    local=True,
    environ=[
        "TF_NEED_TENSORRT",
        _TF_TENSORRT_VERSION,
        _TENSORRT_INSTALLATION_PATH,
        ],
    )
