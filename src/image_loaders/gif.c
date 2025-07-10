#include "../../include/visualizer.h"
#include <gif_lib.h>

void open_gif(char *filename, Image *settings)
{
    int error = 0;
    GifFileType *gif = DGifOpenFileName(filename, &error);
    SavedImage *frame = NULL;
    ColorMapObject *colorMap = NULL;
    GifColorType c;
    int idx = 0;

    if (gif == NULL) {
        fprintf(stderr, "Error opening image file!\n");
        exit(1);
    }
    if (DGifSlurp(gif) == GIF_ERROR) {
        fprintf(stderr, "Error reading image file!\n");
        DGifCloseFile(gif, &error);
        exit(1);
    }
    frame = &gif->SavedImages[0];
    settings->width = gif->SWidth;
    settings->height = gif->SHeight;
    colorMap = frame->ImageDesc.ColorMap ? frame->ImageDesc.ColorMap : gif->SColorMap;
    if (colorMap == NULL) {
        fprintf(stderr, "Unable to get colors from gif!\n");
        DGifCloseFile(gif, &error);
        exit(1);
    }
    settings->channels = 3;
    settings->pixels = malloc(sizeof(unsigned char) * settings->width * settings->height * 3);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        DGifCloseFile(gif, &error);
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
    DGifCloseFile(gif, &error);
    return;
}
