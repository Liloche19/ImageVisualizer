#ifndef _BMP_H_
    #define _BMP_H_

typedef struct {
    unsigned short bfType;
    unsigned bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned bfOffBits;
} BmpFileHeader;

typedef struct {
    unsigned biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned biCompression;
    unsigned biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned biClrUsed;
    unsigned biClrImportant;
} BmpInfoHeader;

#endif // _BMP_H_
