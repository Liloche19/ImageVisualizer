#include "../include/visualizer_cuda.cuh"
#include <pthread.h>

void *load_drivers(void *data)
{
    void *ptr = NULL;

    cudaMalloc(&ptr, 1);
    cudaDeviceSynchronize();
    cudaFree(ptr);
    return NULL;
}

extern "C" void init_gpu(pthread_t *gpu_loader)
{
    int device = 0;
    cudaError_t err = cudaGetDeviceCount(&device);

    if (err != cudaSuccess || device == 0)
        return;
    if (pthread_create(gpu_loader, NULL, &load_drivers, NULL) != 0) {
        fprintf(stderr, "Error while creating thread!\n");
        exit(1);
    }
    return;
}
