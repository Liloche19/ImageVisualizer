#include "../../include/visualizer.h"
#include <gif_lib.h>

void get_pixels_from_frame_gif(Image *settings, int frame_to_load)
{
    int idx = 0;
    int error = 0;
    int transparent_index = 0;
    int has_transparency = 0;
    int disposal_minus_one = 0;
    int local_idx = 0;
    SavedImage *frame = NULL;
    ColorMapObject *colorMap = NULL;
    GraphicsControlBlock gcb;
    GifColorType bg_color = settings->gif->SColorMap->Colors[settings->gif->SBackGroundColor];
    GifImageDesc desc;
    GifColorType c;
    GifZone zone;
    GifZone zone_previous;

    frame = &settings->gif->SavedImages[frame_to_load];
    if (frame_to_load != 0) {
        DGifSavedExtensionToGCB(settings->gif, frame_to_load - 1, &gcb);
        disposal_minus_one = gcb.DisposalMode;
        desc = settings->gif->SavedImages[frame_to_load].ImageDesc;
        zone_previous.offset_left = desc.Left;
        zone_previous.offset_top = desc.Top;
        zone_previous.width = desc.Width;
        zone_previous.height = desc.Height;
    }
    if (disposal_minus_one == DISPOSE_BACKGROUND) {
        printf("retablishing background!\n");
        for (int y = 0; y < zone_previous.height; y++) {
            for (int x = 0; x < zone_previous.width; x++) {
                idx = ((zone_previous.offset_top + y) * settings->width + (zone_previous.offset_left + x)) * 3;
                settings->pixels[idx] = bg_color.Red;
                settings->pixels[idx + 1] = bg_color.Green;
                settings->pixels[idx + 2] = bg_color.Blue;
            }
        }
    }
    if (disposal_minus_one == DISPOSE_PREVIOUS) {
        printf("Retablishing previous!\n");
        for (int y = 0; y < zone_previous.height; y++) {
            for (int x = 0; x < zone_previous.width; x++) {
                idx = ((zone_previous.offset_top + y) * settings->width + (zone_previous.offset_left + x)) * 3;
                settings->pixels[idx] = settings->previous_pixels[idx * 3];
                settings->pixels[idx + 1] = settings->previous_pixels[idx * 3 + 1];
                settings->pixels[idx + 2] = settings->previous_pixels[idx * 3 + 2];
            }
        }
    }
    DGifSavedExtensionToGCB(settings->gif, frame_to_load, &gcb);
    if (gcb.DisposalMode == DISPOSAL_UNSPECIFIED)
        gcb.DisposalMode = DISPOSE_DO_NOT;
    colorMap = frame->ImageDesc.ColorMap ? frame->ImageDesc.ColorMap : settings->gif->SColorMap;
    if (colorMap == NULL) {
        fprintf(stderr, "Unable to get colors from gif!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    settings->channels = 3;
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    desc = frame->ImageDesc;
    zone.offset_left = desc.Left;
    zone.offset_top = desc.Top;
    zone.width = desc.Width;
    zone.height = desc.Height;
    if (gcb.DisposalMode == DISPOSE_PREVIOUS)
        memcpy(settings->previous_pixels, settings->pixels, sizeof(unsigned char) * settings->height * settings->width * 3);
    for (int y = 0; y < zone.height; y++) {
        for (int x = 0; x < zone.width; x++) {
            idx = (zone.offset_top + y) * settings->width + (zone.offset_left + x);
            local_idx = y * zone.width + x;
            if (gcb.TransparentColor != NO_TRANSPARENT_COLOR && frame->RasterBits[local_idx] == gcb.TransparentColor)
                continue;
            c = colorMap->Colors[frame->RasterBits[local_idx]];
            settings->pixels[idx * 3] = c.Red;
            settings->pixels[idx * 3 + 1] = c.Green;
            settings->pixels[idx * 3 + 2] = c.Blue;
        }
    }
    if (frame_to_load > 0 && DGifSavedExtensionToGCB(settings->gif, frame_to_load - 1, &gcb) == GIF_OK)
        settings->ms_to_wait = gcb.DelayTime * 10;
    else if (frame_to_load == 0)
        settings->ms_to_wait = 0;
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
    settings->height = settings->gif->SHeight;
    settings->width = settings->gif->SWidth;
    settings->pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * 3);
    settings->previous_pixels = malloc(sizeof(unsigned char) * settings->height * settings->width * 3);
    if (settings->pixels == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        DGifCloseFile(settings->gif, &error);
        exit(1);
    }
    get_pixels_from_frame_gif(settings, 0);
    return;
}
