#include "cuda_utils.cuh"
#include<stdint.h>

template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride, 
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  // src: (b_size, c_in, l_in)
  // k: (c_out, c_in, k_size)
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k_size = k_dims[2];
  const size_t k_over_2 = k_size / 2;
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  // TODO
  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  const size_t dst_l = dst_i % l_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t offset = 0; offset < k_size; ++offset) {
    const size_t src_l_plus = stride * dst_l + offset;
    if (k_over_2 <= src_l_plus && src_l_plus < l_in + k_over_2) {
      const size_t src_l = src_l_plus - k_over_2;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
        const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
        const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
        d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
    }
  }
  dst[dst_i] = static_cast<T>(d);
}


template <typename T, typename A>
__device__ void conv2d(
  const size_t src_numel,
  const size_t out_h,
  const size_t out_w,
  const size_t stride,
  const size_t *info,
  const T *src,
  const T *kernel,
  T *dst
) {
  // src: (b_size, c_in, i_h, i_w)
  // k: (c_out, c_in, k_h, k_w)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t *k_dims = info + 8;
  const size_t *k_s = info + 12;
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k_h = k_dims[2];
  const size_t k_w = k_dims[3];
  const size_t k_h_over_2 = k_h / 2;
  const size_t k_w_over_2 = k_w / 2;

  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t i_h = src_dims[2];
  const size_t i_w = src_dims[3];

  const size_t b_idx = dst_i / (out_h * out_w * c_out);
  const size_t dst_c_idx = (dst_i / (out_h * out_w)) % c_out;
  const size_t dst_l = dst_i % (out_h * out_w);
  const size_t dst_h = dst_l / out_w;
  const size_t dst_w = dst_l / out_h;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t offset_h = 0; offset_h < k_h; offset_h++) {
    const size_t src_h_plus = stride * dst_h + offset_h;
    if (k_h_over_2 <= src_h_plus && src_h_plus < i_h + k_h_over_2) {
      const size_t src_h = src_h_plus - k_h_over_2;
      for (size_t offset_w = 0; offset_w < k_w; offset_w++) {
        const size_t src_w_plus = stride * dst_w + offset_w;
        if (k_w_over_2 <= src_w_plus && src_w_plus < i_w + k_w_over_2) {
          const size_t src_w = src_w_plus - k_w_over_2;
          for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
            const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_h * src_s[2] + src_w * src_s[3];
            const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset_h * k_s[2] + offset_w * k_s[3];
            d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
          }
        }
      }
    }
  }
  dst[dst_i] = static_cast<T>(d);
}


#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t stride, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d<TYPENAME, TYPEACC>(src_numel, num_dims, stride, info, src, kernel, dst); \
} \

#if __CUDA_ARCH__ >= 800
CONV1D_OP(__nv_bfloat16, float, conv1d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, float, conv1d_f16)
#endif

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)

#define CONV2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t out_h, \
    const size_t out_w, \
    const size_t stride, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv2d<TYPENAME, TYPEACC>(src_numel, out_h, out_w, stride, info, src, kernel, dst); \
} \

#if __CUDA_ARCH__ >= 800
CONV2D_OP(__nv_bfloat16, float, conv2d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV2D_OP(__half, float, conv2d_f16)
#endif

CONV2D_OP(float, float, conv2d_f32)
CONV2D_OP(double, double, conv2d_f64)
CONV2D_OP(uint8_t, uint8_t, conv2d_u8)
CONV2D_OP(uint32_t, uint32_t, conv2d_u32)