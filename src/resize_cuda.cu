#include "../include/visualizer_cuda.cuh"

extern __global__ void resize_image_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y);

extern "C" int resize_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y)
{
    int device = 0;
    cudaError_t err = cudaGetDeviceCount(&device);
    int nb_pixels = 0;
    int nb_blocks = 0;
    int image_size = 0;
    int screen_buffer_size = 0;

    if (err != cudaSuccess || device == 0)
        return 1;
    nb_pixels = screen->cols * screen->rows;
    nb_blocks = (nb_pixels + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    screen_buffer_size = sizeof(char) * (21 * screen->cols * screen->rows + (sizeof(RESET) + 1) * screen->rows);
    image_size = sizeof(unsigned char) * image->channels * image->height * image->width;
    cudaMalloc(&screen->gpu_print_buffer, screen_buffer_size);
    cudaMalloc(&image->gpu_pixels, image_size);
    cudaMemcpy(image->gpu_pixels, image->pixels, image_size, cudaMemcpyHostToDevice);
    resize_image_cuda<<<nb_blocks, CUDA_BLOCK_SIZE>>>(screen, image, ratio_x, ratio_y);
    cudaDeviceSynchronize();
    cudaMemcpy(screen->print_buffer, screen->gpu_print_buffer, image_size, cudaMemcpyDeviceToHost);
    cudaFree(screen->gpu_print_buffer);
    cudaFree(image->gpu_pixels);
    return 0;
}
