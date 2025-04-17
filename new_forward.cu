#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {

/*Self-defined Parameters*/
#define TILE_WIDTH_A 18
#define TILE_WIDTH_B 32
#define MAX_K 7

__constant__ float const_kernel[24*12*MAX_K*MAX_K];

/*Shared Memory Version*/
__global__ void forward_kernel_1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define ck4d(i3, i2, i1, i0) const_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH_A + K - 1;
    // Reverted shared memory to float
    __shared__ float shared_input[TILE_WIDTH_A + MAX_K - 1][TILE_WIDTH_A + MAX_K - 1];

    int W_grid = (W_out + TILE_WIDTH_A - 1) / TILE_WIDTH_A;

    int m = blockIdx.x;
    int h_base = (blockIdx.y / W_grid) * TILE_WIDTH_A;
    int w_base = (blockIdx.y % W_grid) * TILE_WIDTH_A;
    int h = h_base + threadIdx.y;
    int w = w_base + threadIdx.x;
    int b = blockIdx.z;

    // Accumulator is float
    float acc = 0.0f;

    const int c = 0; 

    // Load input tile into shared memory (as float)
    // Removed #pragma unroll from loading loops
    for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
        for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
            int row_in = h_base + i;
            int col_in = w_base + j;
            if (row_in < H &&  col_in < W) { // Add boundary checks
                shared_input[i][j] = x4d(b, c, row_in, col_in);
            }
        }
    }

    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (h < H_out && w < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * ck4d(m, c, 0, 0);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * ck4d(m, c, 0, 1);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * ck4d(m, c, 0, 2);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * ck4d(m, c, 0, 3);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * ck4d(m, c, 0, 4);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * ck4d(m, c, 0, 5);
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * ck4d(m, c, 0, 6);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * ck4d(m, c, 1, 0);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * ck4d(m, c, 1, 1);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * ck4d(m, c, 1, 2);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * ck4d(m, c, 1, 3);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * ck4d(m, c, 1, 4);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * ck4d(m, c, 1, 5);
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * ck4d(m, c, 1, 6);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * ck4d(m, c, 2, 0);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * ck4d(m, c, 2, 1);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * ck4d(m, c, 2, 2);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * ck4d(m, c, 2, 3);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * ck4d(m, c, 2, 4);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * ck4d(m, c, 2, 5);
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * ck4d(m, c, 2, 6);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * ck4d(m, c, 3, 0);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * ck4d(m, c, 3, 1);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * ck4d(m, c, 3, 2);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * ck4d(m, c, 3, 3);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * ck4d(m, c, 3, 4);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * ck4d(m, c, 3, 5);
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * ck4d(m, c, 3, 6);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * ck4d(m, c, 4, 0);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * ck4d(m, c, 4, 1);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * ck4d(m, c, 4, 2);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * ck4d(m, c, 4, 3);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * ck4d(m, c, 4, 4);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * ck4d(m, c, 4, 5);
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * ck4d(m, c, 4, 6);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * ck4d(m, c, 5, 0);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * ck4d(m, c, 5, 1);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * ck4d(m, c, 5, 2);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * ck4d(m, c, 5, 3);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * ck4d(m, c, 5, 4);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * ck4d(m, c, 5, 5);
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * ck4d(m, c, 5, 6);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * ck4d(m, c, 6, 0);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * ck4d(m, c, 6, 1);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * ck4d(m, c, 6, 2);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * ck4d(m, c, 6, 3);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * ck4d(m, c, 6, 4);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * ck4d(m, c, 6, 5);
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * ck4d(m, c, 6, 6);

        y4d(b, m, h, w) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef ck4d
}



__global__ void forward_kernel_2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int X_tile_width = TILE_WIDTH_B + K - 1;
    // Reverted shared memory to float
    __shared__ float shared_kernel[MAX_K][MAX_K];
    __shared__ float shared_input[TILE_WIDTH_B + MAX_K - 1][TILE_WIDTH_B + MAX_K - 1];

    const int W_grid = (W_out + TILE_WIDTH_B - 1) / TILE_WIDTH_B;

    int m = blockIdx.x;

    int b = blockIdx.z;

    // Accumulator is float
    float acc = 0.0f;

    // Removed #pragma unroll from c loop
    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 0, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 0, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 0, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 0, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 0, threadIdx.y, threadIdx.x);
    }

    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data



    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 1, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 1, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 1, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 1, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 1, threadIdx.y, threadIdx.x);
    }


    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data



    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 2, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 2, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 2, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 2, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 2, threadIdx.y, threadIdx.x);
    }

    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data



    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 3, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 3, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 3, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 3, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 3, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data



    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 4, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 4, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 4, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 4, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 4, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data



    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 5, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 5, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 5, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 5, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 5, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data




    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 6, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 6, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 6, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 6, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 6, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data




    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 7, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 7, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 7, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 7, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 7, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data





    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 8, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 8, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 8, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 8, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 8, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data




    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 9, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 9, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 9, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 9, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 9, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data





    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 10, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 10, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 10, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 10, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 10, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data





    // Load input tile into shared memory (as float)

    shared_input[threadIdx.y][threadIdx.x] = x4d(b, 11, threadIdx.y, threadIdx.x);
    if(threadIdx.y == 0){
        shared_input[threadIdx.x][32] = x4d(b, 11, threadIdx.x, 32);
        shared_input[32][threadIdx.x] = x4d(b, 11, 32, threadIdx.x);
        shared_input[32][32] = x4d(b, 11, 32, 32);
    }


    // Load kernel tile into shared memory (as float)
    // Check if m and c are valid before loading kernel weights
    if (threadIdx.y < K && threadIdx.x < K) {
        shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, 11, threadIdx.y, threadIdx.x);
    }
    
    __syncthreads(); // Sync after loading shared memory

    // Perform computation within the tile
    // Check if the current thread's output pixel (h, w) is within bounds
    if (threadIdx.y < H_out && threadIdx.x < W_out) {
        // Explicitly unrolled loops assuming K=7
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 0] * shared_kernel[0][0];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 1] * shared_kernel[0][1];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 2] * shared_kernel[0][2];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 3] * shared_kernel[0][3];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 4] * shared_kernel[0][4];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 5] * shared_kernel[0][5];
        acc += shared_input[threadIdx.y + 0][threadIdx.x + 6] * shared_kernel[0][6];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 0] * shared_kernel[1][0];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 1] * shared_kernel[1][1];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 2] * shared_kernel[1][2];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 3] * shared_kernel[1][3];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 4] * shared_kernel[1][4];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 5] * shared_kernel[1][5];
        acc += shared_input[threadIdx.y + 1][threadIdx.x + 6] * shared_kernel[1][6];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 0] * shared_kernel[2][0];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 1] * shared_kernel[2][1];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 2] * shared_kernel[2][2];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 3] * shared_kernel[2][3];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 4] * shared_kernel[2][4];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 5] * shared_kernel[2][5];
        acc += shared_input[threadIdx.y + 2][threadIdx.x + 6] * shared_kernel[2][6];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 0] * shared_kernel[3][0];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 1] * shared_kernel[3][1];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 2] * shared_kernel[3][2];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 3] * shared_kernel[3][3];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 4] * shared_kernel[3][4];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 5] * shared_kernel[3][5];
        acc += shared_input[threadIdx.y + 3][threadIdx.x + 6] * shared_kernel[3][6];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 0] * shared_kernel[4][0];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 1] * shared_kernel[4][1];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 2] * shared_kernel[4][2];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 3] * shared_kernel[4][3];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 4] * shared_kernel[4][4];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 5] * shared_kernel[4][5];
        acc += shared_input[threadIdx.y + 4][threadIdx.x + 6] * shared_kernel[4][6];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 0] * shared_kernel[5][0];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 1] * shared_kernel[5][1];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 2] * shared_kernel[5][2];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 3] * shared_kernel[5][3];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 4] * shared_kernel[5][4];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 5] * shared_kernel[5][5];
        acc += shared_input[threadIdx.y + 5][threadIdx.x + 6] * shared_kernel[5][6];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 0] * shared_kernel[6][0];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 1] * shared_kernel[6][1];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 2] * shared_kernel[6][2];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 3] * shared_kernel[6][3];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 4] * shared_kernel[6][4];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 5] * shared_kernel[6][5];
        acc += shared_input[threadIdx.y + 6][threadIdx.x + 6] * shared_kernel[6][6];

    }
    __syncthreads(); // Sync before loading next channel's data





    // Write final accumulated result (float)
    if(threadIdx.y< H_out && threadIdx.x < W_out){
        y4d(b, m, threadIdx.y, threadIdx.x) = acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}


/*Gemm Version*/
// __global__ void unroll_kernel(const float *x, const float *k, const int C, const int H, const int W, const int K, float* x_unrolled)
// {

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     int tid = threadIdx.x;
//     int total = C * K * K * H_out * W_out;  // total number of x_unroll
//     int stride = blockDim.x * gridDim.x;   //  total number of threads in one grid
//     int b = blockIdx.z;

//     for (int idx = tid + blockIdx.x * blockDim.x; idx < total; idx += stride) {

//         int out_col = idx % (H_out * W_out);    //x index in output
//         int out_row = (idx / (H_out * W_out));      //y index in output

//         int k_col = out_row % K;                //x index in kernal
//         int k_row = (out_row / K) % K;         //y index in kernal
//         int ch = out_row / (K * K);         //channel index in kernal

//         int h_out = out_col / W_out;        //y index of the first element of the kernal
//         int w_out = out_col % W_out;        //x index of the first element of the kernal

//         int h_in = h_out + k_row;
//         int w_in = w_out + k_col;

//         int in_idx = b * (C * H * W) + ch * H * W + h_in * W + w_in;
//         int out_idx = b * (C * K * K * H_out * W_out) + out_row * (H_out * W_out) + out_col;

//         x_unrolled[out_idx] = x[in_idx];
//     }
    
// }

// __global__ void gemm_kernel(float *y, const float *k, const int M, const int C, const int H, const int W, const int K, float* x_unrolled)
// {

//     __shared__ float sub_k[32][32];
//     __shared__ float sub_x_unrolled[32][32];

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int row = blockIdx.y * blockDim.y + threadIdx.y; // M
//     int col = blockIdx.x * blockDim.x + threadIdx.x; // H_out * W_out

//     float val = 0.0f;
//     int row_unroll = K * K * C; 
//     int col_unroll = H_out * W_out;

//     int b = blockIdx.z;

//     // if (row < M && col < H_out * W_out) {
//     //     for (int i = 0; i < row_unroll; ++i) {
//     //         val += k[row * row_unroll + i] * x_unrolled[(b * row_unroll * col_unroll) + i * H_out * W_out + col];
//     //     }
//     //     y[b * (M * H_out * W_out) + row * H_out * W_out + col] = val;
//     // }


//     for(int numTile = 0; numTile < (row_unroll + 31) / 32; ++numTile){
//         if((numTile * 32 + tx) < row_unroll && row < M){
//             sub_k[ty][tx] = k[row * row_unroll + numTile * 32 + tx];
//         }
//             else{
//             sub_k[ty][tx] = 0.0f;
//         }

//             if((numTile * 32 + ty) < row_unroll && col < col_unroll){  
//             sub_x_unrolled[ty][tx] = x_unrolled[(b * row_unroll * col_unroll) + (numTile * 32 + ty) * col_unroll + col];
//         }
//             else{
//             sub_x_unrolled[ty][tx] = 0.0f;
//         }

//         __syncthreads();

//         for(int k=0 ; k<32 ; k++){
//             val += sub_k[ty][k] * sub_x_unrolled[k][tx]; 
//         }
        
//         __syncthreads();
//     }

//     if(row < M && col < H_out * W_out){
//         y[b * (M * H_out * W_out) + row * H_out * W_out + col] = val;
//     }
// }

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {

    /*Current Version*/
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    // printf("B:%d, C:%d, H:%d, W:%d, M:%d, K:%d\n",B,C,H,W,M,K);

     /*******************Tiling Version**********************/
    // int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of horizontal tiles per output map
    // int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of vertical tiles per output map
    // int Y = W_grid * H_grid;            //number of tiles per output map

    // dim3 gridDim(M,Y,B);
    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    // // C10_CUDA_CHECK(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(), w.data_ptr<float>(), B, M, C, H, W, K);
    // // C10_CUDA_CHECK(cudaDeviceSynchronize());

    /********************Shared Mem**************************/
    
    if(C == 1){
        const int W_grid = (W_out + TILE_WIDTH_A - 1) / TILE_WIDTH_A;        //number of horizontal tiles per output map
        const int H_grid = (H_out + TILE_WIDTH_A - 1) / TILE_WIDTH_A;        //number of vertical tiles per output map
        const int Y = W_grid * H_grid;            //number of tiles per output map

        cudaMemcpyToSymbol(const_kernel, w.data_ptr<float>(), M * C * K * K * sizeof(float));

        dim3 gridDim(M,Y,B);
        dim3 blockDim(TILE_WIDTH_A,TILE_WIDTH_A,1);

        // C10_CUDA_CHECK(cudaDeviceSynchronize());
        forward_kernel_1<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(), w.data_ptr<float>(), B, M, C, H, W, K);
        // C10_CUDA_CHECK(cudaDeviceSynchronize());
    }else{
        const int W_grid = (W_out + TILE_WIDTH_B - 1) / TILE_WIDTH_B;        //number of horizontal tiles per output map
        const int H_grid = (H_out + TILE_WIDTH_B - 1) / TILE_WIDTH_B;        //number of vertical tiles per output map
        const int Y = W_grid * H_grid;            //number of tiles per output map

        dim3 gridDim(M,Y,B);
        dim3 blockDim(TILE_WIDTH_B,TILE_WIDTH_B,1);

        // C10_CUDA_CHECK(cudaDeviceSynchronize());
        forward_kernel_2<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(), w.data_ptr<float>(), B, M, C, H, W, K);
        // C10_CUDA_CHECK(cudaDeviceSynchronize());
    }

    /*******************Gemm Version**********************/
    // int W_unroll = C * K * K;
    // int H_unroll = H_out * W_out;
    // float* X_unrolled;

    // // dim3 gridDim_unroll((H_unroll * W_unroll + 1023) / 1024, 1, B);
    // // dim3 blockDim_unroll(1024, 1, 1);

    // // dim3 gridDim_gemm((H_unroll + 31) / 32, (M + 31)/ 32, B);
    // // dim3 blockDim_gemm(32,32,1);

    // cudaMalloc(&X_unrolled, B_TILE * W_unroll * H_unroll * sizeof(float));

    // for(int b_start = 0; b_start < B; b_start += B_TILE){
    //     int b_eff = std::min(B_TILE, B - b_start);

    //     dim3 gridDim_unroll((H_unroll * W_unroll + 1023) / 1024, 1, b_eff);
    //     dim3 blockDim_unroll(1024, 1, 1);
    
    //     dim3 gridDim_gemm((H_unroll + 31) / 32, (M + 31)/ 32, b_eff);
    //     dim3 blockDim_gemm(32,32,1);

    //     unroll_kernel<<<gridDim_unroll,blockDim_unroll>>>(x.data_ptr<float>() + b_start * C * H * W, w.data_ptr<float>(), C, H, W, K, X_unrolled);

    //     gemm_kernel<<<gridDim_gemm,blockDim_gemm>>>(y.data_ptr<float>() + b_start * M * H_out * W_out, w.data_ptr<float>(), M, C, H, W, K, X_unrolled);
    // }
    // cudaFree(X_unrolled);


    return y;
}

}; // namespace eecs471
