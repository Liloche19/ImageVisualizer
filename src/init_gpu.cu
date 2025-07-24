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

extern "C" void init_gpu(Screen *screen)
{
    int device = 0;
    cudaError_t err = cudaGetDeviceCount(&device);

    if (err != cudaSuccess || device == 0)
        return;
    if (pthread_create(&screen->gpu_loader, NULL, &load_drivers, NULL) != 0) {
        fprintf(stderr, "Error while creating thread!\n");
        exit(1);
    }
    screen->use_gpu = true;
    return;
}
