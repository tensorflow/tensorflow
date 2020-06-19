// automatically generated
/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_HIP_OSTREAM_OPS_H_
#define INC_HIP_OSTREAM_OPS_H_
#ifdef __cplusplus
#include <iostream>

#include "roctracer.h"

#include "hip/hip_runtime_api.h"
#include "hip/hcc_detail/hip_vector_types.h"


namespace roctracer {
namespace hip_support {
// begin ostream ops for HIP 
// HIP basic ostream ops
template <typename T>
  inline std::ostream& operator<<(std::ostream& out, const T& v) {
     using std::operator<<;
     static bool recursion = false;
     if (recursion == false) { recursion = true; out << v; recursion = false; }
     return out; }
inline std::ostream& operator<<(std::ostream& out, void* v) { using std::operator<<; out << std::hex << v; return out; }
inline std::ostream& operator<<(std::ostream& out, const void* v) { using std::operator<<; out << std::hex << v; return out; }
inline std::ostream& operator<<(std::ostream& out, bool v) { using std::operator<<; out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint8_t v) { using std::operator<<; out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint16_t v) { using std::operator<<; out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint32_t v) { using std::operator<<; out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint64_t v) { using std::operator<<; out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, bool* v) {  using std::operator<<; out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint8_t* v) { using std::operator<<; out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint16_t* v) { using std::operator<<; out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint32_t* v) { using std::operator<<; out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }
inline std::ostream& operator<<(std::ostream& out, uint64_t* v) { using std::operator<<; out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }

// End of HIP basic ostream ops

inline std::ostream& operator<<(std::ostream& out, hipLaunchParams& v)
{
   roctracer::hip_support::operator<<(out, v.stream);
   roctracer::hip_support::operator<<(out, v.sharedMem);
   roctracer::hip_support::operator<<(out, v.blockDim);
   roctracer::hip_support::operator<<(out, v.gridDim);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ushort1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ushort2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ushort3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipDeviceProp_t& v)
{
   roctracer::hip_support::operator<<(out, v.isLargeBar);
   roctracer::hip_support::operator<<(out, v.cooperativeMultiDeviceUnmatchedSharedMem);
   roctracer::hip_support::operator<<(out, v.cooperativeMultiDeviceUnmatchedBlockDim);
   roctracer::hip_support::operator<<(out, v.cooperativeMultiDeviceUnmatchedGridDim);
   roctracer::hip_support::operator<<(out, v.cooperativeMultiDeviceUnmatchedFunc);
   roctracer::hip_support::operator<<(out, v.tccDriver);
   roctracer::hip_support::operator<<(out, v.ECCEnabled);
   roctracer::hip_support::operator<<(out, v.kernelExecTimeoutEnabled);
   roctracer::hip_support::operator<<(out, v.texturePitchAlignment);
   roctracer::hip_support::operator<<(out, v.textureAlignment);
   roctracer::hip_support::operator<<(out, v.memPitch);
   roctracer::hip_support::operator<<(out, v.hdpRegFlushCntl);
   roctracer::hip_support::operator<<(out, v.hdpMemFlushCntl);
   roctracer::hip_support::operator<<(out, v.maxTexture3D);
   roctracer::hip_support::operator<<(out, v.maxTexture2D);
   roctracer::hip_support::operator<<(out, v.maxTexture1D);
   roctracer::hip_support::operator<<(out, v.cooperativeMultiDeviceLaunch);
   roctracer::hip_support::operator<<(out, v.cooperativeLaunch);
   roctracer::hip_support::operator<<(out, v.integrated);
   roctracer::hip_support::operator<<(out, v.gcnArch);
   roctracer::hip_support::operator<<(out, v.canMapHostMemory);
   roctracer::hip_support::operator<<(out, v.isMultiGpuBoard);
   roctracer::hip_support::operator<<(out, v.maxSharedMemoryPerMultiProcessor);
   roctracer::hip_support::operator<<(out, v.pciDeviceID);
   roctracer::hip_support::operator<<(out, v.pciBusID);
   roctracer::hip_support::operator<<(out, v.pciDomainID);
   roctracer::hip_support::operator<<(out, v.concurrentKernels);
   roctracer::hip_support::operator<<(out, v.arch);
   roctracer::hip_support::operator<<(out, v.clockInstructionRate);
   roctracer::hip_support::operator<<(out, v.computeMode);
   roctracer::hip_support::operator<<(out, v.maxThreadsPerMultiProcessor);
   roctracer::hip_support::operator<<(out, v.l2CacheSize);
   roctracer::hip_support::operator<<(out, v.multiProcessorCount);
   roctracer::hip_support::operator<<(out, v.minor);
   roctracer::hip_support::operator<<(out, v.major);
   roctracer::hip_support::operator<<(out, v.totalConstMem);
   roctracer::hip_support::operator<<(out, v.memoryBusWidth);
   roctracer::hip_support::operator<<(out, v.memoryClockRate);
   roctracer::hip_support::operator<<(out, v.clockRate);
   roctracer::hip_support::operator<<(out, v.maxGridSize);
   roctracer::hip_support::operator<<(out, v.maxThreadsDim);
   roctracer::hip_support::operator<<(out, v.maxThreadsPerBlock);
   roctracer::hip_support::operator<<(out, v.warpSize);
   roctracer::hip_support::operator<<(out, v.regsPerBlock);
   roctracer::hip_support::operator<<(out, v.sharedMemPerBlock);
   roctracer::hip_support::operator<<(out, v.totalGlobalMem);
   roctracer::hip_support::operator<<(out, v.name);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, double2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, double3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulong4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulong3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulong2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulong1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_ARRAY_DESCRIPTOR& v)
{
   roctracer::hip_support::operator<<(out, v.NumChannels);
   roctracer::hip_support::operator<<(out, v.Format);
   roctracer::hip_support::operator<<(out, v.Height);
   roctracer::hip_support::operator<<(out, v.Width);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipPitchedPtr& v)
{
   roctracer::hip_support::operator<<(out, v.ysize);
   roctracer::hip_support::operator<<(out, v.xsize);
   roctracer::hip_support::operator<<(out, v.pitch);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uchar1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uchar3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uchar2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uchar4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_MEMCPY3D& v)
{
   roctracer::hip_support::operator<<(out, v.Depth);
   roctracer::hip_support::operator<<(out, v.Height);
   roctracer::hip_support::operator<<(out, v.WidthInBytes);
   roctracer::hip_support::operator<<(out, v.dstHeight);
   roctracer::hip_support::operator<<(out, v.dstPitch);
   roctracer::hip_support::operator<<(out, v.dstArray);
   roctracer::hip_support::operator<<(out, v.dstDevice);
   roctracer::hip_support::operator<<(out, v.dstMemoryType);
   roctracer::hip_support::operator<<(out, v.dstLOD);
   roctracer::hip_support::operator<<(out, v.dstZ);
   roctracer::hip_support::operator<<(out, v.dstY);
   roctracer::hip_support::operator<<(out, v.dstXInBytes);
   roctracer::hip_support::operator<<(out, v.srcHeight);
   roctracer::hip_support::operator<<(out, v.srcPitch);
   roctracer::hip_support::operator<<(out, v.srcArray);
   roctracer::hip_support::operator<<(out, v.srcDevice);
   roctracer::hip_support::operator<<(out, v.srcMemoryType);
   roctracer::hip_support::operator<<(out, v.srcLOD);
   roctracer::hip_support::operator<<(out, v.srcZ);
   roctracer::hip_support::operator<<(out, v.srcY);
   roctracer::hip_support::operator<<(out, v.srcXInBytes);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, float4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, float1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, float2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, float3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, max_align_t& v)
{
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_RESOURCE_DESC& v)
{
   roctracer::hip_support::operator<<(out, v.flags);
   roctracer::hip_support::operator<<(out, v.resType);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, long4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipExtent& v)
{
   roctracer::hip_support::operator<<(out, v.depth);
   roctracer::hip_support::operator<<(out, v.height);
   roctracer::hip_support::operator<<(out, v.width);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ushort4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, surfaceReference& v)
{
   roctracer::hip_support::operator<<(out, v.surfaceObject);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipDeviceArch_t& v)
{
   roctracer::hip_support::operator<<(out, v.hasDynamicParallelism);
   roctracer::hip_support::operator<<(out, v.has3dGrid);
   roctracer::hip_support::operator<<(out, v.hasSurfaceFuncs);
   roctracer::hip_support::operator<<(out, v.hasSyncThreadsExt);
   roctracer::hip_support::operator<<(out, v.hasThreadFenceSystem);
   roctracer::hip_support::operator<<(out, v.hasFunnelShift);
   roctracer::hip_support::operator<<(out, v.hasWarpShuffle);
   roctracer::hip_support::operator<<(out, v.hasWarpBallot);
   roctracer::hip_support::operator<<(out, v.hasWarpVote);
   roctracer::hip_support::operator<<(out, v.hasDoubles);
   roctracer::hip_support::operator<<(out, v.hasSharedInt64Atomics);
   roctracer::hip_support::operator<<(out, v.hasGlobalInt64Atomics);
   roctracer::hip_support::operator<<(out, v.hasFloatAtomicAdd);
   roctracer::hip_support::operator<<(out, v.hasSharedFloatAtomicExch);
   roctracer::hip_support::operator<<(out, v.hasSharedInt32Atomics);
   roctracer::hip_support::operator<<(out, v.hasGlobalFloatAtomicExch);
   roctracer::hip_support::operator<<(out, v.hasGlobalInt32Atomics);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipArray& v)
{
   roctracer::hip_support::operator<<(out, v.textureType);
   roctracer::hip_support::operator<<(out, v.NumChannels);
   roctracer::hip_support::operator<<(out, v.Format);
   roctracer::hip_support::operator<<(out, v.depth);
   roctracer::hip_support::operator<<(out, v.height);
   roctracer::hip_support::operator<<(out, v.width);
   roctracer::hip_support::operator<<(out, v.type);
   roctracer::hip_support::operator<<(out, v.desc);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, short4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, short1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, short2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, short3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_RESOURCE_VIEW_DESC& v)
{
   roctracer::hip_support::operator<<(out, v.reserved);
   roctracer::hip_support::operator<<(out, v.lastLayer);
   roctracer::hip_support::operator<<(out, v.firstLayer);
   roctracer::hip_support::operator<<(out, v.lastMipmapLevel);
   roctracer::hip_support::operator<<(out, v.firstMipmapLevel);
   roctracer::hip_support::operator<<(out, v.depth);
   roctracer::hip_support::operator<<(out, v.height);
   roctracer::hip_support::operator<<(out, v.width);
   roctracer::hip_support::operator<<(out, v.format);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipFuncAttributes& v)
{
   roctracer::hip_support::operator<<(out, v.sharedSizeBytes);
   roctracer::hip_support::operator<<(out, v.ptxVersion);
   roctracer::hip_support::operator<<(out, v.preferredShmemCarveout);
   roctracer::hip_support::operator<<(out, v.numRegs);
   roctracer::hip_support::operator<<(out, v.maxThreadsPerBlock);
   roctracer::hip_support::operator<<(out, v.maxDynamicSharedSizeBytes);
   roctracer::hip_support::operator<<(out, v.localSizeBytes);
   roctracer::hip_support::operator<<(out, v.constSizeBytes);
   roctracer::hip_support::operator<<(out, v.cacheModeCA);
   roctracer::hip_support::operator<<(out, v.binaryVersion);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipMemcpy3DParms& v)
{
   roctracer::hip_support::operator<<(out, v.kind);
   roctracer::hip_support::operator<<(out, v.extent);
   roctracer::hip_support::operator<<(out, v.dstPtr);
   roctracer::hip_support::operator<<(out, v.dstPos);
   roctracer::hip_support::operator<<(out, v.dstArray);
   roctracer::hip_support::operator<<(out, v.srcPtr);
   roctracer::hip_support::operator<<(out, v.srcPos);
   roctracer::hip_support::operator<<(out, v.srcArray);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, __locale_struct& v)
{
   roctracer::hip_support::operator<<(out, v.__names);
   roctracer::hip_support::operator<<(out, v.__ctype_toupper);
   roctracer::hip_support::operator<<(out, v.__ctype_tolower);
   roctracer::hip_support::operator<<(out, v.__ctype_b);
   roctracer::hip_support::operator<<(out, v.__locales);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipResourceViewDesc& v)
{
   roctracer::hip_support::operator<<(out, v.lastLayer);
   roctracer::hip_support::operator<<(out, v.firstLayer);
   roctracer::hip_support::operator<<(out, v.lastMipmapLevel);
   roctracer::hip_support::operator<<(out, v.firstMipmapLevel);
   roctracer::hip_support::operator<<(out, v.depth);
   roctracer::hip_support::operator<<(out, v.height);
   roctracer::hip_support::operator<<(out, v.width);
   roctracer::hip_support::operator<<(out, v.format);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipIpcMemHandle_t& v)
{
   roctracer::hip_support::operator<<(out, v.reserved);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uint4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uint1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_TEXTURE_DESC& v)
{
   roctracer::hip_support::operator<<(out, v.reserved);
   roctracer::hip_support::operator<<(out, v.borderColor);
   roctracer::hip_support::operator<<(out, v.maxMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.minMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.mipmapLevelBias);
   roctracer::hip_support::operator<<(out, v.mipmapFilterMode);
   roctracer::hip_support::operator<<(out, v.maxAnisotropy);
   roctracer::hip_support::operator<<(out, v.flags);
   roctracer::hip_support::operator<<(out, v.filterMode);
   roctracer::hip_support::operator<<(out, v.addressMode);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uint3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, uint2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, textureReference& v)
{
   roctracer::hip_support::operator<<(out, v.format);
   roctracer::hip_support::operator<<(out, v.numChannels);
   roctracer::hip_support::operator<<(out, v.textureObject);
   roctracer::hip_support::operator<<(out, v.maxMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.minMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.mipmapLevelBias);
   roctracer::hip_support::operator<<(out, v.mipmapFilterMode);
   roctracer::hip_support::operator<<(out, v.maxAnisotropy);
   roctracer::hip_support::operator<<(out, v.sRGB);
   roctracer::hip_support::operator<<(out, v.channelDesc);
   roctracer::hip_support::operator<<(out, v.filterMode);
   roctracer::hip_support::operator<<(out, v.readMode);
   roctracer::hip_support::operator<<(out, v.normalized);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, int4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipResourceDesc& v)
{
   roctracer::hip_support::operator<<(out, v.resType);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, int1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, int3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, int2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, longlong1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, longlong3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, longlong2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, longlong4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, dim3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipChannelFormatDesc& v)
{
   roctracer::hip_support::operator<<(out, v.f);
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, double4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulonglong4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulonglong1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulonglong3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, ulonglong2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, char1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, char3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, char2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, char4& v)
{
   roctracer::hip_support::operator<<(out, v.w);
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, double1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipPos& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, HIP_ARRAY3D_DESCRIPTOR& v)
{
   roctracer::hip_support::operator<<(out, v.Flags);
   roctracer::hip_support::operator<<(out, v.NumChannels);
   roctracer::hip_support::operator<<(out, v.Format);
   roctracer::hip_support::operator<<(out, v.Depth);
   roctracer::hip_support::operator<<(out, v.Height);
   roctracer::hip_support::operator<<(out, v.Width);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipTextureDesc& v)
{
   roctracer::hip_support::operator<<(out, v.maxMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.minMipmapLevelClamp);
   roctracer::hip_support::operator<<(out, v.mipmapLevelBias);
   roctracer::hip_support::operator<<(out, v.mipmapFilterMode);
   roctracer::hip_support::operator<<(out, v.maxAnisotropy);
   roctracer::hip_support::operator<<(out, v.normalizedCoords);
   roctracer::hip_support::operator<<(out, v.borderColor);
   roctracer::hip_support::operator<<(out, v.sRGB);
   roctracer::hip_support::operator<<(out, v.readMode);
   roctracer::hip_support::operator<<(out, v.filterMode);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipMipmappedArray& v)
{
   roctracer::hip_support::operator<<(out, v.depth);
   roctracer::hip_support::operator<<(out, v.height);
   roctracer::hip_support::operator<<(out, v.width);
   roctracer::hip_support::operator<<(out, v.desc);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, long3& v)
{
   roctracer::hip_support::operator<<(out, v.z);
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, long2& v)
{
   roctracer::hip_support::operator<<(out, v.y);
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, long1& v)
{
   roctracer::hip_support::operator<<(out, v.x);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hip_Memcpy2D& v)
{
   roctracer::hip_support::operator<<(out, v.Height);
   roctracer::hip_support::operator<<(out, v.WidthInBytes);
   roctracer::hip_support::operator<<(out, v.dstPitch);
   roctracer::hip_support::operator<<(out, v.dstArray);
   roctracer::hip_support::operator<<(out, v.dstDevice);
   roctracer::hip_support::operator<<(out, v.dstMemoryType);
   roctracer::hip_support::operator<<(out, v.dstY);
   roctracer::hip_support::operator<<(out, v.dstXInBytes);
   roctracer::hip_support::operator<<(out, v.srcPitch);
   roctracer::hip_support::operator<<(out, v.srcArray);
   roctracer::hip_support::operator<<(out, v.srcDevice);
   roctracer::hip_support::operator<<(out, v.srcMemoryType);
   roctracer::hip_support::operator<<(out, v.srcY);
   roctracer::hip_support::operator<<(out, v.srcXInBytes);
   return out;
}
inline std::ostream& operator<<(std::ostream& out, hipPointerAttribute_t& v)
{
   roctracer::hip_support::operator<<(out, v.allocationFlags);
   roctracer::hip_support::operator<<(out, v.isManaged);
   roctracer::hip_support::operator<<(out, v.device);
   roctracer::hip_support::operator<<(out, v.memoryType);
   return out;
}
// end ostream ops for HIP 
};};

inline std::ostream& operator<<(std::ostream& out, const hipLaunchParams& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ushort1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ushort2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ushort3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipDeviceProp_t& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const double2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const double3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulong4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulong3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulong2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulong1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipPitchedPtr& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_MEMCPY3D& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const float4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const float1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const float2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const float3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const max_align_t& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_DESC& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const long4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipExtent& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ushort4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const surfaceReference& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipDeviceArch_t& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipArray& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const short4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const short1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const short2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const short3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_VIEW_DESC& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipFuncAttributes& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipMemcpy3DParms& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const __locale_struct& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipResourceViewDesc& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipIpcMemHandle_t& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uint4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uint1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_TEXTURE_DESC& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uint3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const uint2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const textureReference& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const int4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipResourceDesc& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const int1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const int3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const int2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const longlong1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const longlong3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const longlong2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const longlong4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const dim3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipChannelFormatDesc& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const double4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulonglong4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulonglong1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulonglong3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const ulonglong2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const char1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const char3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const char2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const char4& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const double1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipPos& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipTextureDesc& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipMipmappedArray& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const long3& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const long2& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const long1& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hip_Memcpy2D& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

inline std::ostream& operator<<(std::ostream& out, const hipPointerAttribute_t& v)
{
   roctracer::hip_support::operator<<(out, v);
   return out;
}

#endif //__cplusplus
#endif // INC_HIP_OSTREAM_OPS_H_
 
