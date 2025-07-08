#include "../include/visualizer.h"

long get_nb_threads(void)
{
    long nb_threads = 0;

    nb_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (nb_threads < 2)
        return 1;
    return nb_threads - 1;
}
