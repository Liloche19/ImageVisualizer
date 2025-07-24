#include "../include/visualizer_cuda.cuh"

extern __global__ void resize_image_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y);

extern "C" int resize_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y)
{
    int device = 0;
    cudaError_t err = cudaGetDeviceCount(&device);
    int nb_pixels = 0;
    int nb_blocks = 0;
    int image_size = 0;
    Screen *gpu_screen;
    Image *gpu_image;

    if (err != cudaSuccess || device == 0)
        return 1;
    nb_pixels = screen->cols * screen->rows;
    nb_blocks = (nb_pixels + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    screen->buffer_size = sizeof(char) * (sizeof(PIXEL_TEMPLATE) * screen->cols * screen->rows + (sizeof(RESET) + 1) * screen->rows);
    image_size = sizeof(unsigned char) * image->channels * image->height * image->width;
    if (pthread_join(screen->gpu_loader, NULL) != 0) {
        fprintf(stderr, "Error while waiting thread!\n");
        exit(1);
    }
    if (cudaMalloc(&gpu_screen, sizeof(Screen)) != cudaSuccess || cudaMalloc(&gpu_image, sizeof(Image)) != cudaSuccess) {
        fprintf(stderr, "Error initialising structures!\n");
        exit(1);
    }
    if (cudaMalloc(&(screen->gpu_print_buffer), screen->buffer_size) != cudaSuccess || cudaMalloc(&(image->gpu_pixels), image_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(1);
    }
    if (cudaMemcpy(image->gpu_pixels, image->pixels, image_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed!\n");
        exit(1);
    }
    if (cudaMemcpy(gpu_screen, screen, sizeof(Screen), cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(gpu_image, image, sizeof(Image), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error copying data to GPU!\n");
        exit(1);
    }
    resize_image_cuda<<<nb_blocks, CUDA_BLOCK_SIZE>>>(gpu_screen, gpu_image, ratio_x, ratio_y);
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        fprintf(stderr, "sync failed!\n%s\n", cudaGetErrorString(err));
        exit(1);
    }
    if ((err = cudaMemcpy(screen->print_buffer, screen->gpu_print_buffer, screen->buffer_size, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed!\n%s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaFree(screen->gpu_print_buffer);
    cudaFree(image->gpu_pixels);
    cudaFree(gpu_screen);
    cudaFree(gpu_image);
    return 0;
}
