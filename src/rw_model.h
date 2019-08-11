#ifndef __RW_MODEL_H_
#define __RW_MODEL_H_

#include "model.h"

/*
 * Read and write model files
 * Contents:
 * hp: HyperParameter
 * D: int
 * N: int
 * vocabSize: int
 * h1: int
 * h2: int
 * W1: double, h1 x vocabSize
 * W2: double, h2 x (h1 x D)
 * bias2: double, h2 x 1
 * W3: double, vocabSize x h2
 * bias3: double, vocabSize x 1
 */

void writeModel(const WordNet *model, const char *modelFileName);
void readAndCreateModel(const char *modelFileName, WordNet *model);

#endif // __RW_MODEL_H_
