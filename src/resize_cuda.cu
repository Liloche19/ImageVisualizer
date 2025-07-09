#include "../include/visualizer_cuda.cuh"

extern __global__ void resize_image_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y);

extern "C" int resize_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y)
{
    int device = 0;
    cudaError_t err = cudaGetDeviceCount(&device);
    int nb_pixels = 0;
    int nb_blocks = 0;

    if (err != cudaSuccess || device == 0)
        return 1;
    nb_pixels = screen->cols * screen->rows;
    nb_blocks = (nb_pixels + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    resize_image_cuda<<<nb_blocks, CUDA_BLOCK_SIZE>>>(screen, image, ratio_x, ratio_y);
    return 0;
}
