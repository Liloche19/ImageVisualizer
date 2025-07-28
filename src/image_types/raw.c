#include "../../include/visualizer.h"
#include <libraw/libraw.h>

void open_raw(char *filename, Image *settings)
{
    int ret = 0;
    libraw_processed_image_t *img = NULL;
    libraw_data_t *processor = libraw_init(0);

    ret = libraw_open_file(processor, filename);
    if (ret)
        exit(1);
    ret = libraw_unpack(processor);
    if (ret)
        exit(1);
    ret = libraw_dcraw_process(processor);
    if (ret)
        exit(1);
    img = libraw_dcraw_make_mem_image(processor, &ret);
    if (!img)
        exit(1);
    settings->height = img->height;
    settings->width = img->width;
    settings->channels = img->colors;
    settings->pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * settings->channels);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }
    memcpy(settings->pixels, img->data, sizeof(unsigned char) * settings->height * settings->width * settings->channels);
    libraw_dcraw_clear_mem(img);
    libraw_close(processor);
    settings->actual_frame = 0;
    settings->nb_frames = 1;
    settings->type = RAW;
    return;
}
