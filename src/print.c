#include "../include/visualizer.h"

void print_pixel(unsigned char r, unsigned char g, unsigned char b)
{
    char *pixel = NULL;

    asprintf(&pixel, "\033[48;2;%u;%u;%um%c%s", r, g, b, CHAR, RESET);
    if (pixel == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    write(1, pixel, strlen(pixel));
    free(pixel);
    return;
}

void display_image(Image *image, Screen *screen)
{
    int index = 0;

    screen->resized = resize_image(image, screen);
    for (int i = 0; i < screen->rows; i++) {
        for (int j = 0; j < screen->cols; j++) {
            index = (i * screen->cols + j) * 3;
            print_pixel(screen->resized[index + 0], screen->resized[index + 1], screen->resized[index + 2]);
        }
        write(1, "\n", 1);
    }
    free(screen->resized);
    return;
}
