#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define TILE_DIM 16

// Tiled matrix multiplication kernel using shared memory
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ float s_a[TILE_DIM][TILE_DIM];
    __shared__ float s_b[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile from A into shared memory
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_b[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int main(int argc, char** argv) {
    // Query and print device information
    int deviceId = 0;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));

    std::cout << "======================================================" << std::endl;
    std::cout << "  NVIDIA GPU Device Info" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Device Name       : " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory      : " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "SM Count          : " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp Size         : " << prop.warpSize << std::endl;
    std::cout << "Max Shared Mem/SM : " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "======================================================" << std::endl;

    // Matrix dimensions M x K and K x N -> M x N
    int M = 2048;
    int N = 2048;
    int K = 2048;
    if (argc >= 2) M = atoi(argv[1]);
    if (argc >= 3) N = atoi(argv[2]);
    if (argc >= 4) K = atoi(argv[3]);

    std::cout << "\nRunning GEMM test: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    // Host allocations
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f;

    // Device allocations
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Warm-up run
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    int num_iterations = 20;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < num_iterations; ++iter) {
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
    float ms_avg = ms_total / num_iterations;

    // FLOP calculation: 2 * M * N * K operations per GEMM
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_avg * 1e-3)) / 1e12;

    // Verify result
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float expected = (float)K * 1.0f * 2.0f;
    bool correct = true;
    for (int i = 0; i < std::min(M * N, 100); ++i) {
        if (std::abs(h_C[i] - expected) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": got " << h_C[i] 
                      << ", expected " << expected << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << "Verification      : " << (correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Average Latency   : " << ms_avg << " ms" << std::endl;
    std::cout << "Achieved TFLOPS   : " << tflops << " TFLOPS (FP32)" << std::endl;
    std::cout << "======================================================\n" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return correct ? 0 : 1;
}
