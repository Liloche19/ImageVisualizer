#include "../include/visualizer.h"
#include <unistd.h>

void apply_color_at_coord_on_buffer(Screen *screen, int x, int y, rgb_t color)
{
    char pixel[] = "\033[48;2;000;000;000m ";
    int start_index = sizeof(pixel) * x + (sizeof(RESET) + 1 + sizeof(pixel) * screen->cols) * y;

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
        screen->print_buffer[start_index + i] = pixel[i];
    if (x == screen->cols - 1) {
        for (int i = 0; i < (int) sizeof(RESET); i++)
            screen->print_buffer[start_index + sizeof(pixel) + i] = RESET[i];
        screen->print_buffer[start_index + sizeof(pixel) + sizeof(RESET)] = '\n';
    }
    return;
}

void display_image(Image *image, Screen *screen)
{
    char pixel[] = "\033[48;2;000;000;000m ";
    int size = 0;

    screen->print_buffer = resize_image(image, screen);
    size = (sizeof(pixel) * screen->cols * screen->rows + (sizeof(RESET) + 1) * screen->rows);
    write(1, screen->print_buffer, size);
    free(screen->print_buffer);
    return;
}
