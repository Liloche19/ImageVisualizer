#include "../../include/visualizer.h"
#include <gif_lib.h>

void get_pixels_from_frame_gif(Image *settings, int frame_to_load)
{
    int error = 0;
    SavedImage *frame = NULL;
    ColorMapObject *colorMap = NULL;
    GifColorType c;
    int idx = 0;

    if (settings->pixels != NULL)
        free(settings->pixels);
    frame = &settings->gif->SavedImages[frame_to_load];
    settings->width = settings->gif->SWidth;
    settings->height = settings->gif->SHeight;
    colorMap = frame->ImageDesc.ColorMap ? frame->ImageDesc.ColorMap : settings->gif->SColorMap;
    if (colorMap == NULL) {
        fprintf(stderr, "Unable to get colors from gif!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    settings->channels = 3;
    settings->pixels = malloc(sizeof(unsigned char) * settings->width * settings->height * 3);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    for (int i = 0; i < settings->height * settings->width; i++) {
        idx = frame->RasterBits[i];
        if (idx >= colorMap->ColorCount)
            idx = 0;
        c = colorMap->Colors[idx];
        settings->pixels[3 * i] = c.Red;
        settings->pixels[3 * i + 1] = c.Green;
        settings->pixels[3 * i + 2] = c.Blue;
    }
    return;
}

void open_gif(char *filename, Image *settings)
{
    int error = 0;

    settings->gif = DGifOpenFileName(filename, &error);
    if (settings->gif == NULL) {
        fprintf(stderr, "Error opening image file!\n");
        exit(1);
    }
    if (DGifSlurp(settings->gif) == GIF_ERROR) {
        fprintf(stderr, "Error reading image file!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    settings->nb_frames = settings->gif->ImageCount;
    settings->actual_frame = 0;
    settings->ms_to_wait = 50;
    get_pixels_from_frame_gif(settings, 0);
    return;
}
