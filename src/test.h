#ifndef __TEST_H_
#define __TEST_H_

#include "model.h"

double test(WordNet *model, const int *batchInput4Test,
            const int *batchTarget4Test, int batchNum4Test);

#endif // __TEST_H_
