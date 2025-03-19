#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrixMultiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CUDA kernel for Softmax operation
__global__ void softmax(float* input, float* output, int N, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N) {
        // Find maximum value in the row for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, input[row * N + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            float exp_val = expf((input[row * N + i] - max_val) / sqrtf((float)d_k));
            output[row * N + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < N; i++) {
            output[row * N + i] /= sum;
        }
    }
}

// CUDA kernel for matrix concatenation
__global__ void concatHeads(float** head_outputs, float* output, int N, int d_v, int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = col / d_v;
    int head_col = col % d_v;
    
    if (row < N && col < d_v * h) {
        output[row * (d_v * h) + col] = head_outputs[head_idx][row * d_v + head_col];
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    // Calculate dimensions for each head
    int d_k = d_model / h;
    int d_v = d_model / h; // Assuming d_k = d_v
    
    // Device memory pointers
    float *d_Q, *d_K, *d_V, *d_output;
    float **d_head_outputs, **head_outputs;
    
    // Allocate device memory
    cudaMalloc(&d_Q, N * d_model * sizeof(float));
    cudaMalloc(&d_K, N * d_model * sizeof(float));
    cudaMalloc(&d_V, N * d_model * sizeof(float));
    cudaMalloc(&d_output, N * d_model * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_Q, Q, N * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for head outputs
    head_outputs = (float**)malloc(h * sizeof(float*));
    cudaMalloc(&d_head_outputs, h * sizeof(float*));
    
    for (int i = 0; i < h; i++) {
        cudaMalloc(&head_outputs[i], N * d_v * sizeof(float));
    }
    cudaMemcpy(d_head_outputs, head_outputs, h * sizeof(float*), cudaMemcpyHostToDevice);
    
    // Temporary matrices for computations
    float *d_QK, *d_softmax_output;
    cudaMalloc(&d_QK, N * N * sizeof(float));
    cudaMalloc(&d_softmax_output, N * N * sizeof(float));
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    // Process each head
    for (int i = 0; i < h; i++) {
        float *d_Q_i, *d_K_i, *d_V_i;
        
        // Split the matrices for this head
        cudaMalloc(&d_Q_i, N * d_k * sizeof(float));
        cudaMalloc(&d_K_i, N * d_k * sizeof(float));
        cudaMalloc(&d_V_i, N * d_v * sizeof(float));
        
        // Extract Q_i, K_i, V_i (simple offset-based approach)
        // In a real implementation, you would use linear projections with weight matrices
        for (int j = 0; j < N; j++) {
            cudaMemcpy(d_Q_i + j * d_k, d_Q + j * d_model + i * d_k, d_k * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_K_i + j * d_k, d_K + j * d_model + i * d_k, d_k * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_V_i + j * d_v, d_V + j * d_model + i * d_v, d_v * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        // Compute Q_i * K_i^T
        matrixMultiply<<<gridDim, blockDim>>>(d_Q_i, d_K_i, d_QK, N, d_k, N);
        
        // Apply softmax
        softmax<<<gridDim, blockDim>>>(d_QK, d_softmax_output, N, d_k);
        
        // Compute softmax(Q_i * K_i^T / sqrt(d_k)) * V_i
        matrixMultiply<<<gridDim, blockDim>>>(d_softmax_output, d_V_i, head_outputs[i], N, N, d_v);
        
        // Free temporary memory
        cudaFree(d_Q_i);
        cudaFree(d_K_i);
        cudaFree(d_V_i);
    }
    
    // Concatenate the head outputs
    dim3 concatBlockDim(16, 16);
    dim3 concatGridDim((d_model + concatBlockDim.x - 1) / concatBlockDim.x, 
                       (N + concatBlockDim.y - 1) / concatBlockDim.y);
                       
    // In a simplified approach, just copy each head's output to the appropriate position
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < N; j++) {
            cudaMemcpy(d_output + j * d_model + i * d_v, 
                      head_outputs[i] + j * d_v, 
                      d_v * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
        }
    }
    
    // Copy result back to host
    cudaMemcpy(output, d_output, N * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    cudaFree(d_QK);
    cudaFree(d_softmax_output);
    
    for (int i = 0; i < h; i++) {
        cudaFree(head_outputs[i]);
    }
    cudaFree(d_head_outputs);
    free(head_outputs);
}
 

