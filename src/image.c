#include "../include/visualizer.h"

rgb_t avg_rgb(unsigned char *img, float ratio_x, float ratio_y, int x, int y, int width, int height, int channels)
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
            coord = (((int) (x * ratio_x) + offset_x) + ((int) (y * ratio_y) + offset_y) * width) * channels;
            if (coord + 2 >= coord_max || coord < 0)
                break;
            nb_pixels++;
            r += img[coord];
            g += img[coord + 1];
            b += img[coord + 2];
        }
    }
    rgb.rgb[0] = r / nb_pixels;
    rgb.rgb[1] = g / nb_pixels;
    rgb.rgb[2] = b / nb_pixels;
    return rgb;
}

void apply_color_at_coord(unsigned char *image, rgb_t rgb, int x, int y, int width)
{
    int index = x * 3 + y * width * 3;

    image[index] = rgb.rgb[0];
    image[index + 1] = rgb.rgb[1];
    image[index + 2] = rgb.rgb[2];
    return;
}

unsigned char *resize_image(Image *image, Screen *screen)
{
    float img_ratio = 0;
    float ratio_x = 0.0;
    float ratio_y = 0.0;
    unsigned char *resized = NULL;

    img_ratio = (float) image->width / image->height;
    if (img_ratio > ((float) screen->cols / screen->rows) / screen->char_ratio)
        screen->rows = screen->cols / (img_ratio * screen->char_ratio);
    else
        screen->cols = screen->rows * img_ratio * screen->char_ratio;
    resized = malloc(sizeof(unsigned char) * screen->cols * screen->rows * 3);
    if (image == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    ratio_x = (float) image->width / screen->cols;
    ratio_y = (float) image->height / screen->rows;
    for (int x = 0; x < screen->cols; x++)
        for (int y = 0; y < screen->rows; y++)
            apply_color_at_coord(resized, avg_rgb(image->pixels, ratio_x, ratio_y, x, y, image->width, image->height, image->channels), x, y, screen->cols);
    return resized;
}

void load_image(char *filename, Image *settings)
{
    settings->pixels = stbi_load(filename, &settings->width, &settings->height, &settings->channels, 0);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Can't open image!\n");
        exit(1);
    }
    return;
}
