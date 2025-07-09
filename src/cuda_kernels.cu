#include "../include/visualizer_cuda.cuh"

__device__ rgb_t avg_rgb_cuda(unsigned char *img, float ratio_x, float ratio_y, int x, int y, int width, int height, int channels)
{
    rgb_t rgb;
    int r = 0;
    int g = 0;
    int b = 0;
    int coord = 0;
    int nb_pixels = 0;
    int coord_max = height * width * channels;

    for (int offset_x = 0; offset_x < ratio_x; offset_x++) {
        for (int offset_y = 0; offset_y < ratio_y; offset_y++) {
            if (((int) (x * ratio_x) + offset_x) >= width || ((int) (y * ratio_y) + offset_y) >= height)
                break;
            coord = (((int) (x * ratio_x) + offset_x) + ((int) (y * ratio_y) + offset_y) * width) * channels;
            nb_pixels++;
            r += img[coord];
            g += img[coord + 1];
            b += img[coord + 2];
        }
    }
    rgb.rgb[0] = nb_pixels == 0 ? 0 : r / nb_pixels;
    rgb.rgb[1] = nb_pixels == 0 ? 0 : g / nb_pixels;
    rgb.rgb[2] = nb_pixels == 0 ? 0 : b / nb_pixels;
    return rgb;
}

__device__ void apply_color_at_coord_on_buffer_cuda(Screen *screen, int x, int y, rgb_t color)
{
    char pixel[] = "\033[48;2;000;000;000m ";
    int start_index = 21 * x + (sizeof(RESET) + 1 + 21 * screen->cols) * y;

    pixel[7] = (color.rgb[0] / 100) + 48;
    pixel[8] = ((color.rgb[0] / 10) % 10) + 48;
    pixel[9] = (color.rgb[0] % 10) + 48;
    pixel[11] = (color.rgb[1] / 100) + 48;
    pixel[12] = ((color.rgb[1] / 10) % 10) + 48;
    pixel[13] = (color.rgb[1] % 10) + 48;
    pixel[15] = (color.rgb[2] / 100) + 48;
    pixel[16] = ((color.rgb[2] / 10) % 10) + 48;
    pixel[17] = (color.rgb[2] % 10) + 48;
    pixel[19] = CHAR;
    for (int i = 0; i < (int) sizeof(pixel); i++)
        screen->gpu_print_buffer[start_index + i] = pixel[i];
    if (x == screen->cols - 1) {
        for (int i = 0; i < (int) sizeof(RESET); i++)
            screen->gpu_print_buffer[start_index + sizeof(pixel) + i] = RESET[i];
        screen->gpu_print_buffer[start_index + sizeof(pixel) + sizeof(RESET)] = '\n';
        if (start_index + sizeof(pixel) + sizeof(RESET) > screen->buffer_size) {
                printf("Thread (%d,%d): Déborde du buffer à l'index %d (taille: %d)\n",
                       x, y, start_index, screen->buffer_size);
                return;
            }
    }
    return;
}

__global__ void resize_image_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int y = index / screen->cols;
    int x = index - (y * screen->cols);

    if (index >= screen->cols * screen->rows)
        return;
    apply_color_at_coord_on_buffer_cuda(screen, x, y, avg_rgb_cuda(image->gpu_pixels, ratio_x, ratio_y, x, y, image->width, image->height, image->channels));
    return;
}
