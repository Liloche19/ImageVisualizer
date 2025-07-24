#include "../../include/visualizer.h"
#include <jpeglib.h>

void open_jpeg(char *filename, Image *settings)
{
    FILE *infile = fopen(filename, "rb");
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    int row_stride = 0;
    unsigned char *buffer = NULL;
    JSAMPARRAY line_buffer;

    if (infile == NULL) {
        fprintf(stderr, "Error while opening image file!\n");
        exit(1);
    }
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    settings->width = cinfo.output_width;
    settings->height = cinfo.output_height;
    settings->channels = cinfo.output_components;
    row_stride = (settings->width) * (settings->channels);
    buffer = malloc((settings->width) * (settings->height) * (settings->channels));
    if (buffer == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return;
    }
    line_buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, line_buffer, 1);
        memcpy(&buffer[(cinfo.output_scanline - 1) * row_stride], line_buffer[0], row_stride);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    settings->pixels = buffer;
    settings->actual_frame = 0;
    settings->nb_frames = 1;
    settings->type = JPEG;
    return;
}
