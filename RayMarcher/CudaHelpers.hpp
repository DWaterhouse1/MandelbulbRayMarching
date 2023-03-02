#ifndef RAYMARCHER_CUDAHELPERS_HPP
#define RAYMARCHER_CUDAHELPERS_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * Copied from <https://stackoverflow.com/a/14038590>
 */
#ifndef NDEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#else
#define gpuErrchk(ans) ans
#endif // NDEBUG
#endif // RAYMARCHER_CUDAHELPERS_HPP