#include "../include/visualizer.h"

void destroy_image(Image *image)
{
    switch (image->type) {
        case GIF:
            destroy_gif(image);
            break;
        case WEBP:
            destroy_webp(image);
            break;
        default:
            break;
    }
    free(image->pixels);
    return;
}
