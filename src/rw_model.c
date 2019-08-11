#include "rw_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void writeModel(const WordNet *model, const char *modelFileName) {
  FILE *fp = fopen(modelFileName, "wb");
  if (fp == NULL) {
    printf("## FATAL ERROR: cannot write %s\n", modelFileName);
    exit(-1);
  }

  fwrite(&model->hp, sizeof(HyperParameter), 1, fp);

  fwrite(model->W1, sizeof(double), model->h1 * model->vocabSize, fp);
  fwrite(model->W2, sizeof(double), model->h2 * (model->h1 * model->D), fp);
  fwrite(model->bias2, sizeof(double), model->h2, fp);
  fwrite(model->W3, sizeof(double), model->vocabSize * model->h2, fp);
  fwrite(model->bias3, sizeof(double), model->vocabSize, fp);

  fclose(fp);
}

void readAndCreateModel(const char *modelFileName, WordNet *model) {
  FILE *fp = fopen(modelFileName, "rb");
  if (fp == NULL) {
    printf("## FATAL error: cannot read %s\n", modelFileName);
    exit(-1);
  }

  fread(&model->hp, sizeof(HyperParameter), 1, fp);
  createWordNet(model);

  fread(model->W1, sizeof(double), model->h1 * model->vocabSize, fp);
  fread(model->W2, sizeof(double), model->h2 * (model->h1 * model->D), fp);
  fread(model->bias2, sizeof(double), model->h2, fp);
  fread(model->W3, sizeof(double), model->vocabSize * model->h2, fp);
  fread(model->bias3, sizeof(double), model->vocabSize, fp);

  fclose(fp);
}
