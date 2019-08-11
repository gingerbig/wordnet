#ifndef _PRINT_UTILS
#define _PRINT_UTILS

#include "model.h"

/*
 Print a page/slice of a 3D matrix
 * @param A, row, column: a ? x row x column matrix
 * @param page: assign page to print, start with 0
*/
void print3dPage(const int *A, int row, int column, int page);

/*
 Print words from indices
 * @param indices: word indices in a dictionary/vocabulary
 * @param length: length of the index string
 * @param vocabulary: the dictionary mapping between words and indices
*/
void printWordFromIndex(const int *indices, int length,
                        const char (*vocabulary)[20]);

void printHelp(const char *appName);

void printModelInfo(const WordNet *model);

#endif
