#include "../../include/visualizer.h"
#include <webp/decode.h>
#include <webp/demux.h>

void get_pixels_from_next_frame_webp(Image *settings)
{
    FrameZone zone;
    unsigned char *frame = NULL;
    int idx = 0;
    int width = 0;
    int height = 0;
    int global_idx = 0;
    int local_idx = 0;
    float alpha = 0;

    WebPDemuxNextFrame(&settings->webp);
    zone.width = settings->webp.width;
    zone.height = settings->webp.height;
    zone.offset_left = settings->webp.x_offset;
    zone.offset_top = settings->webp.y_offset;
    if (settings->actual_frame == 0) {
        settings->height = zone.height;
        settings->width = zone.width;
        settings->pixels = calloc(settings->height * settings->width * settings->channels, 1);
        if (settings->pixels == NULL) {
            fprintf(stderr, "Malloc failed!\n");
            exit(1);
        }
    }
    if (settings->webp.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
        for (int y = 0; y < zone.height; y++) {
            for (int x = 0; x < zone.width; x++) {
                idx = ((y + zone.offset_top) * settings->width + (x + zone.offset_left)) * 4;
                settings->pixels[idx] = 0;
                settings->pixels[idx + 1] = 0;
                settings->pixels[idx + 2] = 0;
                settings->pixels[idx + 3] = 0;
            }
        }
    }
    frame = WebPDecodeRGBA(settings->webp.fragment.bytes, settings->webp.fragment.size, &width, &height);
    if (frame == NULL) {
        fprintf(stderr, "Error while decoding webp frame!\n");
        exit(1);
    }
    settings->ms_to_wait = settings->webp.duration;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            local_idx = (y * width + x) * 4;
            global_idx = ((y + zone.offset_top) * settings->width + (x + zone.offset_left)) * 4;
            if (settings->webp.blend_method == WEBP_MUX_NO_BLEND)
                memcpy(&settings->pixels[global_idx], &frame[local_idx], 4);
            else {
                alpha = frame[local_idx + 3] / 255.0f;
                for (int i = 0; i < 4; ++i)
                    settings->pixels[global_idx + i] =
                        (uint8_t)(frame[local_idx + i] * alpha +
                                  settings->pixels[global_idx + i] * (1.0f - alpha));
            }
        }
    }
    return;
}

void open_webp(char *filename, Image *settings)
{
    FILE *file = fopen(filename, "rb");
    size_t file_size = 0;
    uint8_t *webp_data = NULL;
    WebPDemuxer* demux = NULL;
    WebPData webp_raw_data;

    settings->use_webp = true;
    if (file == NULL) {
        fprintf(stderr, "Error opening image file!\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    rewind(file);
    webp_data = malloc(file_size);
    if (!webp_data) {
        fprintf(stderr, "Malloc failed!\n");
        fclose(file);
        exit(1);
    }
    if (fread(webp_data, 1, file_size, file) != file_size) {
        fprintf(stderr, "Error while reading image file!\n");
        free(webp_data);
        fclose(file);
        exit(1);
    }
    fclose(file);
    settings->channels = 4;
    webp_raw_data = (WebPData) {webp_data, file_size};
    demux = WebPDemux(&webp_raw_data);
    if (!demux) {
        fprintf(stderr, "Error while accessing webp frames!\n");
        free(webp_data);
        exit(1);
    }
    settings->nb_frames = WebPDemuxGetI(demux, WEBP_FF_FRAME_COUNT);
    if (!WebPDemuxGetFrame(demux, 1, &settings->webp)) {
        fprintf(stderr, "Error while accessing webp frames!\n");
        exit(1);
    }
    //WebPDemuxDelete(demux);
    //free(webp_data);
    settings->actual_frame = 0;
    return;
}
