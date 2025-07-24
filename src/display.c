#include "../include/visualizer.h"
#include <unistd.h>

long long get_time_ms(void)
{
    long long time = 0;
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    time = (ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
    return time;
}

void apply_color_at_coord_on_buffer(Screen *screen, int x, int y, rgb_t color)
{
    char pixel[] = PIXEL_TEMPLATE;
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
    int size = 0;
    long long last_frame_time = get_time_ms();
    long long actual_frame_time = 0;
    long diff_time = 0;

    while (image->actual_frame < image->nb_frames) {
        if (image->type == GIF)
            get_pixels_from_frame_gif(image, image->actual_frame);
        if (image->type == WEBP)
            get_pixels_from_next_frame_webp(image);
        get_screen_informations(screen);
        screen->print_buffer = resize_image(image, screen);
        size = (sizeof(PIXEL_TEMPLATE) * screen->cols * screen->rows + (sizeof(RESET) + 1) * screen->rows);
        actual_frame_time = get_time_ms();
        if (actual_frame_time - last_frame_time + image->ms_to_wait > 0)
            usleep((actual_frame_time - last_frame_time + image->ms_to_wait) * 1000);
        write(1, screen->print_buffer, size);
        last_frame_time = get_time_ms();
        image->actual_frame++;
        free(screen->print_buffer);
    }
    return;
}
