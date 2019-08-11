#ifndef __TRAIN_H_
#define __TRAIN_H_

#include "model.h"

void train(WordNet *model, const int *batchInput4Training,
           const int *batchTarget4Training, int batchNum4Training,
           const int *batchInput4Validation, const int *batchTarget4Validation,
           int batchNum4Validation);

#endif // __TRAIN_H_
