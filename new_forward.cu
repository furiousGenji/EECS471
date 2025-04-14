#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {

/*Self-defined Parameters*/
#define TILE_WIDTH 32
#define MAX_K 7
#define BLOCK_DEPTH 1024 / (TILE_WIDTH * TILE_WIDTH)
#define BLOCK_SIZE TILE_WIDTH * TILE_WIDTH
#define B_TILE 500

// Original Version
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {

//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int b = blockDim.x * blockIdx.x + threadIdx.x;

//     if (b < B) // for each image in the batch
//     {
//         /*Original Version*/

//         for (int m = 0; m < M; m++)         // for each output feature maps
//             for (int h = 0; h < H_out; h++) // for each output element
//                 for (int w = 0; w < W_out; w++)
//                 {
//                     y4d(b, m, h, w) = 0;
//                     for (int c = 0; c < C; c++)     // sum over all input feature maps
//                         for (int p = 0; p < K; p++) // KxK filter
//                             for (int q = 0; q < K; q++)
//                                 y4d(b, m, h, w) += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
//                 }

//     }

// #undef y4d
// #undef x4d
// #undef k4d
// }


// /*Working Version*/
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {

//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of horizontal tiles per output map

//     /*Self-defined Parameters*/
//     int m = blockIdx.x;                 // for each output feature maps
//     int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;   //thread height in output featured map
//     int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;   //thread width in output featured map
//     int b = blockIdx.z;
//     float acc;

//     /*Current Version*/
//     /*Every thread caculate part of one element in one output featured map (one input map)*/
//     if(m < M && h < H_out && w < W_out){
//         acc = 0;

//         for (int c = 0; c < C; c++){    // sum over all input feature maps
//             for (int p = 0; p < K; p++){ // KxK filter
//                 for (int q = 0; q < K; q++){
//                     int input_h = h + p;
//                     int input_w = w + q;
//                     acc += x4d(b, c, input_h, input_w) * k4d(m, c, p, q);
//                 }
//             }
//         }
//         y4d(b , m , h , w) = acc;
//     }


// #undef y4d
// #undef x4d
// #undef k4d
// }


/*Current Version*/
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {

//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.
//     We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     */

//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

// // An example use of these macros:
// // float a = y4d(0,0,0,0)
// // y4d(0,0,0,0) = a
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     const int X_tile_width = TILE_WIDTH + K - 1;
    
//     // extern __shared__ float shmem[];
//     // float* shared_input = &shmem[0];
//     // float* shared_kernal = &shmem[C * X_tile_width * X_tile_width];
//     __shared__ float shared_kernel[12][MAX_K][MAX_K]; 
//     __shared__ float shared_input[12][TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1];

//     int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of horizontal tiles per output map

//     /*Self-defined Parameters*/
//     int m = blockIdx.x;                                         // for each output feature maps
//     int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;            //first thread height in TB
//     int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;            //first thread width in TB
//     int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;   //thread height in output featured map
//     int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;   //thread width in output featured map
//     int b = blockIdx.z;
//     float acc;

//     //Input data into the Shared Memory (This will cause divergence!!!)
//     int tid = threadIdx.y * blockDim.x + threadIdx.x;

//     for(int i = 0 ; i < C; i += 1){    
//         for (int j = threadIdx.y; j < X_tile_width; j += blockDim.y) {
//             for (int k = threadIdx.x; k < X_tile_width; k += blockDim.x) {
//                 int row_in = h_base + j;
//                 int col_in = w_base + k;
//                 if (row_in < H && col_in < W) {
//                     shared_input[i][j][k] = x4d(b, i, row_in, col_in);
//                 }
//             }
//         }
//     }
//     __syncthreads();

//     //Kernal into the shared memory
//     for(int i = 0 ; i < C; i += 1){
//         if (threadIdx.x < K && threadIdx.y < K) {
//             shared_kernel[i][threadIdx.y][threadIdx.x] = k4d(m, i, threadIdx.y, threadIdx.x);
//         }
//     }
//     __syncthreads();

    
//     /*Every thread caculate part of one element in one output featured map (one input map)*/
//     if(m < M && h < H_out && w < W_out){
//         acc = 0;

//         for (int c = 0; c < C; c++){    // sum over all input feature maps
//             for (int p = 0; p < K; p++){ // KxK filter
//                 for (int q = 0; q < K; q++){
//                     int input_h = h + p;
//                     int input_w = w + q;
//                     acc += shared_input[c][threadIdx.y + p][threadIdx.x + q] * shared_kernel[c][p][q];
//                 }
//             }
//         }
//         y4d(b , m , h , w) = acc;


//     }


// #undef y4d
// #undef x4d
// #undef k4d
// }

/*Gemm Version*/
__global__ void unroll_kernel(const float *x, const float *k, const int C, const int H, const int W, const int K, float* x_unrolled)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int tid = threadIdx.x;
    int total = C * K * K * H_out * W_out;  // total number of x_unroll
    int stride = blockDim.x * gridDim.x;   //  total number of threads in one grid
    int b = blockIdx.z;

    for (int idx = tid + blockIdx.x * blockDim.x; idx < total; idx += stride) {

        int out_col = idx % (H_out * W_out);    //x index in output
        int out_row = (idx / (H_out * W_out));      //y index in output

        int k_col = out_row % K;                //x index in kernal
        int k_row = (out_row / K) % K;         //y index in kernal
        int ch = out_row / (K * K);         //channel index in kernal

        int h_out = out_col / W_out;        //y index of the first element of the kernal
        int w_out = out_col % W_out;        //x index of the first element of the kernal

        int h_in = h_out + k_row;
        int w_in = w_out + k_col;

        int in_idx = b * (C * H * W) + ch * H * W + h_in * W + w_in;
        int out_idx = b * (C * K * K * H_out * W_out) + out_row * (H_out * W_out) + out_col;

        x_unrolled[out_idx] = x[in_idx];
    }
    
}

__global__ void gemm_kernel(float *y, const float *k, const int M, const int C, const int H, const int W, const int K, float* x_unrolled)
{

    __shared__ float sub_k[32][32];
    __shared__ float sub_x_unrolled[32][32];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // H_out * W_out

    float val = 0.0f;
    int row_unroll = K * K * C; 
    int col_unroll = H_out * W_out;

    int b = blockIdx.z;

    // if (row < M && col < H_out * W_out) {
    //     for (int i = 0; i < row_unroll; ++i) {
    //         val += k[row * row_unroll + i] * x_unrolled[(b * row_unroll * col_unroll) + i * H_out * W_out + col];
    //     }
    //     y[b * (M * H_out * W_out) + row * H_out * W_out + col] = val;
    // }


    for(int numTile = 0; numTile < (row_unroll + 31) / 32; ++numTile){
        if((numTile * 32 + tx) < row_unroll && row < M){
            sub_k[ty][tx] = k[row * row_unroll + numTile * 32 + tx];
        }
            else{
            sub_k[ty][tx] = 0.0f;
        }

            if((numTile * 32 + ty) < row_unroll && col < col_unroll){  
            sub_x_unrolled[ty][tx] = x_unrolled[(b * row_unroll * col_unroll) + (numTile * 32 + ty) * col_unroll + col];
        }
            else{
            sub_x_unrolled[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int k=0 ; k<32 ; k++){
            val += sub_k[ty][k] * sub_x_unrolled[k][tx]; 
        }
        
        __syncthreads();
    }

    if(row < M && col < H_out * W_out){
        y[b * (M * H_out * W_out) + row * H_out * W_out + col] = val;
    }
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    /*Original Version*/
    // const int B = x.size(0);
    // const int C = x.size(1);
    // const int H = x.size(2);
    // const int W = x.size(3);
    // const int K = w.size(3);
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // auto y = torch::empty({B, M, H_out, W_out}, x.options());

    // dim3 gridDim((B + 511) / 512);
    // dim3 blockDim(512);
    

    // // C10_CUDA_CHECK(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(),
    //                                       w.data_ptr<float>(), B, M, C, H, W, K);
    // // C10_CUDA_CHECK(cudaDeviceSynchronize());

    // return y;

    /*Current Version*/
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    printf("B:%d, C:%d, H:%d, W:%d, M:%d, K:%d\n",B,C,H,W,M,K);

     /*******************Tiling Version**********************/

    // /*Self-defined Parameters*/
    // int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of horizontal tiles per output map
    // int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;        //number of vertical tiles per output map
    // int Y = W_grid * H_grid;            //number of tiles per output map

    // dim3 gridDim(M,Y,B);
    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    // // C10_CUDA_CHECK(cudaDeviceSynchronize());
    // forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(),
    // w.data_ptr<float>(), B, M, C, H, W, K);
    // // C10_CUDA_CHECK(cudaDeviceSynchronize());

    /*******************Gemm Version**********************/
    int W_unroll = C * K * K;
    int H_unroll = H_out * W_out;
    float* X_unrolled;

    // dim3 gridDim_unroll((H_unroll * W_unroll + 1023) / 1024, 1, B);
    // dim3 blockDim_unroll(1024, 1, 1);

    // dim3 gridDim_gemm((H_unroll + 31) / 32, (M + 31)/ 32, B);
    // dim3 blockDim_gemm(32,32,1);

    cudaMalloc(&X_unrolled, B_TILE * W_unroll * H_unroll * sizeof(float));

    for(int b_start = 0; b_start < B; b_start += B_TILE){
        int b_eff = std::min(B_TILE, B - b_start);

        dim3 gridDim_unroll((H_unroll * W_unroll + 1023) / 1024, 1, b_eff);
        dim3 blockDim_unroll(1024, 1, 1);
    
        dim3 gridDim_gemm((H_unroll + 31) / 32, (M + 31)/ 32, b_eff);
        dim3 blockDim_gemm(32,32,1);

        unroll_kernel<<<gridDim_unroll,blockDim_unroll>>>(x.data_ptr<float>() + b_start * C * H * W, w.data_ptr<float>(), C, H, W, K, X_unrolled);

        gemm_kernel<<<gridDim_gemm,blockDim_gemm>>>(y.data_ptr<float>() + b_start * M * H_out * W_out, w.data_ptr<float>(), M, C, H, W, K, X_unrolled);
    }
    

    cudaFree(X_unrolled);


    return y;
}






/*ZTY Version*/
/*Working Version*/
// // __constant__ float k[MAXKernelLength];


// // Use half for input/output pointers, keep dimensions as int
// __global__ void forward_kernel(half *y, const half *x, const half *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

// // Adjust macros for half type pointer arithmetic if necessary (should work as is)
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     const int X_tile_width = TILE_WIDTH + K - 1;
//     // Use half for shared memory
//     __shared__ half shared_kernel[MAX_K][MAX_K];
//     // Add padding for alignment (+1) ? Let's check. Usually good practice.
//     // Check if TILE_WIDTH + MAX_K - 1 needs extra padding for half. Let's keep it for now.
//     __shared__ half shared_input[TILE_WIDTH + MAX_K - 1][TILE_WIDTH + MAX_K - 1];

//     int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;

//     int m = blockIdx.x;
//     int h_base = (blockIdx.y / W_grid) * TILE_WIDTH;
//     int w_base = (blockIdx.y % W_grid) * TILE_WIDTH;
//     int h = h_base + threadIdx.y;
//     int w = w_base + threadIdx.x;
//     int b = blockIdx.z;

//     // Accumulator remains float for precision
//     float acc = 0.0f;

//     for(int c = 0; c < C; c++){
//         // Load input tile into shared memory (as half)
//         for (int i = threadIdx.y; i < X_tile_width; i += blockDim.y) {
//             for (int j = threadIdx.x; j < X_tile_width; j += blockDim.x) {
//                 int row_in = h_base + i;
//                 int col_in = w_base + j;
//                 if (row_in >= 0 && row_in < H && col_in >= 0 && col_in < W) { // Add boundary checks
//                     shared_input[i][j] = x4d(b, c, row_in, col_in);
//                 } else {
//                     // Pad with zero
//                     shared_input[i][j] = (half)0.0f;
//                 }
//             }
//         }

//         // Load kernel tile into shared memory (as half)
//         // Ensure only valid threads load kernel weights
//         if (threadIdx.y < K && threadIdx.x < K) {
//              // Check if m and c are valid before loading kernel weights
//              if (m < M && c < C) {
//                 shared_kernel[threadIdx.y][threadIdx.x] = k4d(m, c, threadIdx.y, threadIdx.x);
//              } else {
//                  // Handle cases where m or c might be out of bounds if gridDim.x > M or C is not uniform?
//                  // Or assume valid m, c based on grid setup. Let's assume valid for now.
//                  // If needed, add padding: shared_kernel[threadIdx.y][threadIdx.x] = (half)0.0f;
//              }
//         }
//         __syncthreads(); // Sync after loading shared memory

//         // Perform computation within the tile
//         // Check if the current thread's output pixel (h, w) is within bounds
//         if (h < H_out && w < W_out) {
//             for (int p = 0; p < K; p++){
//                 for (int q = 0; q < K; q++){
//                     // Multiply half, convert product to float, then accumulate
//                     acc += __half2float(shared_input[threadIdx.y + p][threadIdx.x + q] * shared_kernel[p][q]);
//                 }
//             }
//         }
//         __syncthreads(); // Sync before loading next channel's data
//     }

//     // Write final accumulated result (convert float acc to half)
//     if(m < M && h < H_out && w < W_out){
//         y4d(b, m, h, w) = __float2half(acc);
//     }

//     #undef y4d
//     #undef x4d
//     #undef k4d
// }

// torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
//     // Inputs x and w are expected to be float

//     const int B = x.size(0);
//     const int C = x.size(1);
//     const int H = x.size(2);
//     const int W = x.size(3);
//     const int K = w.size(3); // Kernel size from weights tensor

//     // Check if input types are float
//     TORCH_CHECK(x.scalar_type() == torch::kFloat, "Input tensor x must be float32");
//     TORCH_CHECK(w.scalar_type() == torch::kFloat, "Weight tensor w must be float32");


//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     // Convert inputs to half precision
//     auto x_half = x.to(torch::kHalf);
//     auto w_half = w.to(torch::kHalf);

//     // Create output tensor in half precision
//     auto y_half = torch::empty({B, M, H_out, W_out}, x_half.options()); // options will be kHalf

//     // Define grid and block dimensions
//     int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
//     int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
//     int Y = W_grid * H_grid; // Total tiles in H-W plane

//     // Grid: (Output Channels M, Tiles in H-W plane, Batch B)
//     dim3 gridDim(M, Y, B);
//     // Block: (Tile Width, Tile Height, 1)
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

//     // Launch the kernel with half precision pointers
//     // Note: Use at::Half for data_ptr with kHalf tensors
//     // Cast c10::Half* to CUDA's half*
//     forward_kernel<<<gridDim, blockDim>>>(
//         reinterpret_cast<half*>(y_half.data_ptr<at::Half>()),
//         reinterpret_cast<const half*>(x_half.data_ptr<at::Half>()),
//         reinterpret_cast<const half*>(w_half.data_ptr<at::Half>()),
//         B, M, C, H, W, K);

//     // Synchronize device after kernel launch
//     C10_CUDA_CHECK(cudaDeviceSynchronize());

//     // Convert the half precision result back to float precision
//     auto y = y_half.to(torch::kFloat);

//     return y; // Return float tensor
// }





}; // namespace eecs471
