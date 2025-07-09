#ifndef _VISUALIZER_CUDA_H_
    #define _VISUALIZER_CUDA_H_

    #include "visualizer.h"

    #define CUDA_BLOCK_SIZE 256

// Kernel CUDA

#ifdef __cplusplus
extern "C" {
#endif
    int resize_cuda(Screen *screen, Image *image, float ratio_x, float ratio_y);
#ifdef __cplusplus
}
#endif

#endif // _VISUALIZER_CUDA_H
