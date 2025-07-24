#ifdef USE_CUDA
    #include "../include/visualizer_cuda.cuh"
#else
    #include "../include/visualizer.h"
#endif // USE_CUDA

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
    rgb.rgb[0] = nb_pixels == 0 ? 0 : r / nb_pixels;
    rgb.rgb[1] = nb_pixels == 0 ? 0 : g / nb_pixels;
    rgb.rgb[2] = nb_pixels == 0 ? 0 : b / nb_pixels;
    return rgb;
}

void *resize_image_part(void *data)
{
    ThreadData *settings = data;
    int x = 0;
    int y = 0;

    for (int i = settings->start_index; i < settings->end_index; i++) {
        y = i / settings->screen->cols;
        x = i - (y * settings->screen->cols);
        apply_color_at_coord_on_buffer(settings->screen, x, y, avg_rgb(settings->image->pixels, settings->ratio_x, settings->ratio_y, x, y, settings->image->width, settings->image->height, settings->image->channels));
    }
    return NULL;
}

char *resize_image_with_threads(Screen *screen, Image *image, float ratio_x, float ratio_y)
{
    long nb_thread = get_nb_threads();
    pthread_t *threads_id = NULL;
    ThreadData *threads_data = NULL;
    float thread_ratio = 0.0;

    if (nb_thread >= 2)
        nb_thread -= 1;
    thread_ratio = (float) (screen->cols * screen->rows) / nb_thread;
    threads_id = malloc(sizeof(pthread_t) * nb_thread);
    threads_data = malloc(sizeof(ThreadData) * nb_thread);
    if (threads_id == NULL || threads_data == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    for (int i = 0; i < nb_thread; i++) {
        threads_data[i].start_index = i * thread_ratio;
        threads_data[i].end_index = (i + 1) * thread_ratio;
        threads_data[i].image = image;
        threads_data[i].screen = screen;
        threads_data[i].ratio_x = ratio_x;
        threads_data[i].ratio_y = ratio_y;
        pthread_create(&threads_id[i], NULL, resize_image_part, &threads_data[i]);
    }
    for (int i = 0; i < nb_thread; i++) {
        pthread_join(threads_id[i], NULL);
    }
    free(threads_id);
    free(threads_data);
    return screen->print_buffer;
}

char *resize_image(Image *image, Screen *screen)
{
    float img_ratio = 0;
    float ratio_x = 0.0;
    float ratio_y = 0.0;

    img_ratio = (float) image->width / image->height;
    if (img_ratio > ((float) screen->cols / screen->rows) / screen->char_ratio)
        screen->rows = screen->cols / (img_ratio * screen->char_ratio);
    else
        screen->cols = screen->rows * img_ratio * screen->char_ratio;
    screen->print_buffer = malloc(sizeof(char) * (sizeof(PIXEL_TEMPLATE) * screen->cols * screen->rows + (sizeof(RESET) + 1) * screen->rows));
    if (screen->print_buffer == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    ratio_x = (float) image->width / screen->cols;
    ratio_y = (float) image->height / screen->rows;
    #ifdef USE_CUDA
        if (resize_cuda(screen, image, ratio_x, ratio_y) != 0)
            return resize_image_with_threads(screen, image, ratio_x, ratio_y);
        return screen->print_buffer;
    #else
        return resize_image_with_threads(screen, image, ratio_x, ratio_y);
    #endif
    return screen->print_buffer;
}
