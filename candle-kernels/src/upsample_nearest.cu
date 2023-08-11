#include "cuda_utils.cuh"
#include <stdint.h>

template <typename T>
__device__ void upsample_nearest2d(
    const size_t numel,
    const size_t *info,
    const size_t dst_h,
    const size_t dst_w,
    // Trust me bro, I did the math
    const double scale_h,
    const double scale_w,
    const T *inp,
    T *out
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t *dims = info;
    const size_t *strides = info + 4;

    const size_t b_sz = dims[0];
    const size_t c = dims[1];
    const size_t src_h = dims[2];
    const size_t src_w = dims[3];
    const size_t stride_b = strides[0];
    const size_t stride_c = strides[1];
    const size_t stride_h = strides[2];
    const size_t stride_w = strides[3];

    const size_t dst_h_idx = dst_i / (b_sz * c * dst_w);
    const size_t dst_w_idx = dst_i / (b_sz * c * dst_h);

    size_t _src_h_idx = src_h - 1;
    if (dst_h_idx < src_h) {
        const double _tmp_dbl = (double) dst_h_idx / scale_h;
        _src_h_idx = (size_t) _tmp_dbl;
    }
    const size_t src_h_idx = _src_h_idx;

    size_t _src_w_idx = src_w - 1;
    if (dst_h_idx < src_h) {
        const double _tmp_dbl = (double) dst_h_idx / scale_w;
        _src_w_idx = (size_t) _tmp_dbl;
    }
    const size_t src_w_idx = _src_w_idx;

    T d = 0;
    for(size_t b_idx = 0; b_idx < b_sz; b_idx++) {
        for(size_t c_idx = 0; c_idx < c; c_idx++) {
            const size_t src_idx = b_idx * stride_b + c_idx * stride_c + src_h_idx * stride_h + src_w_idx * stride_w;
            d = inp[src_idx];
        }
    }
    out[dst_i] = d;
}

#define UPSAMPLE_NEAREST2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel,  \
    const size_t *info, \
    const size_t dst_h, \
    const size_t dst_w, \
    const double scale_h, \
    const double scale_w, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    upsample_nearest2d<TYPENAME>(numel, info, dst_h, dst_w, scale_h, scale_w, inp, out); \
} \

#if __CUDA_ARCH__ >= 800
UPSAMPLE_NEAREST2D_OP(__nv_bfloat16, upsample_nearest_2d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
UPSAMPLE_NEAREST2D_OP(__half, upsample_nearest_2d_f16)
#endif

UPSAMPLE_NEAREST2D_OP(float, upsample_nearest2d_f32)
UPSAMPLE_NEAREST2D_OP(double, upsample_nearest2d_f64)
UPSAMPLE_NEAREST2D_OP(uint8_t, upsample_nearest2d_u8)
UPSAMPLE_NEAREST2D_OP(uint32_t, upsample_nearest2d_u32)
