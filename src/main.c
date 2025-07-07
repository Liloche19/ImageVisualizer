#define STB_IMAGE_IMPLEMENTATION
#include "../include/visualizer.h"
#include "../include/stb_image.h"

int help(char *prog, int status)
{
    printf("USAGE:\n");
    printf("\t%s [-h] filemane\n\n", prog);
    printf("DESCRIPTION:\n");
    printf("\tfilename\tThe filepath to the image you want to display on your terminal\n");
    printf("\t-h\t\tPrints this help\n");
    return status;
}

void print_pixel(unsigned char r, unsigned char g, unsigned char b)
{
    char *pixel = NULL;

    asprintf(&pixel, "\033[48;2;%u;%u;%um%c%s", r, g, b, CHAR, RESET);
    if (pixel == NULL) {
        perror("Malloc failed!\n");
        exit(1);
    }
    write(1, pixel, strlen(pixel));
    return;
}

int avg_rgb(unsigned char *img, float ratio_x, float ratio_y, int x, int y, int width, int height)
{
    int rgb = 0;
    int r = 0;
    int g = 0;
    int b = 0;
    int coord = 0;
    int nb_pixels = 0;
    int coord_max = height * width * 3;

    for (int offset_x = 0; offset_x < ratio_x; offset_x++) {
        for (int offset_y = 0; offset_y < ratio_y; offset_y++) {
            coord = (((int) (x * ratio_x) + offset_x) + ((int) (y * ratio_y) + offset_y) * width) * 3;
            if (coord + 2 >= coord_max || coord < 0)
                break;
            nb_pixels++;
            r += img[coord];
            g += img[coord + 1];
            b += img[coord + 2];
        }
    }
    r /= nb_pixels;
    g /= nb_pixels;
    b /= nb_pixels;
    rgb = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
    return rgb;
}

void apply_color_at_coord(unsigned char *image, int rgb, int x, int y, int width)
{
    int index = x * 3 + y * width * 3;

    image[index] = (char) (rgb >> 16) & 0xFF;
    image[index + 1] = (char) (rgb >> 8) & 0xFF;
    image[index + 2] = (char) (rgb) & 0xFF;
    return;
}

unsigned char *resize_image(unsigned char *img, int *width, int *height, int *target_width, int *target_height, float char_ratio)
{
    unsigned char *image = NULL;
    float img_ratio = 0;
    float ratio_x = 0.0;
    float ratio_y = 0.0;

    img_ratio = (float) *width / *height;
    if (img_ratio > ((float) *target_width / *target_height) * char_ratio)
        *target_height = *target_width / (img_ratio / char_ratio);
    else
        *target_width = *target_height * img_ratio * char_ratio;
    image = malloc(sizeof(unsigned char) * *target_width * *target_height * 3);
    if (image == NULL) {
        perror("Malloc failed!\n");
        exit(1);
    }
    ratio_x = (float) *width / *target_width;
    ratio_y = (float) *height / *target_height;
    for (int x = 0; x < *target_width; x++)
        for (int y = 0; y < *target_height; y++)
            apply_color_at_coord(image, avg_rgb(img, ratio_x, ratio_y, x, y, *width, *height), x, y, *target_width);
    return image;
}

void display_image(unsigned char *img, int width, int height, int target_width, int target_height, float char_ratio)
{
    int index = 0;

    img = resize_image(img, &width, &height, &target_width, &target_height, char_ratio);
    for (int i = 0; i < target_height; i++) {
        for (int j = 0; j < target_width; j++) {
            index = (i * target_width + j) * 3;
            print_pixel(img[index + 0], img[index + 1], img[index + 2]);
        }
        write(1, "\n", 1);
    }
    free(img);
    return;
}

int main(int argc, char **argv)
{
    float char_ratio = CHAR_RATIO;
    char *image = NULL;
    unsigned char *img = NULL;
    int width, height, channels;
    struct winsize w;

    if (argc != 2)
        return help(argv[0], 1);
    if (strcmp(argv[1], "-h") == 0)
        return help(argv[0], 0);
    image = argv[1];
    if (image == NULL || char_ratio == 0)
        return 0;
    img = stbi_load(image, &width, &height, &channels, 0);
    if (img == NULL)
        return 1;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    char_ratio = (float) w.ws_ypixel / w.ws_row / ((float) w.ws_xpixel / w.ws_col);
    display_image(img, width, height, w.ws_col, w.ws_row, char_ratio);
    stbi_image_free(img);
    return 0;
}
