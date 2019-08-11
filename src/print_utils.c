#include "print_utils.h"
#include <stdio.h>

/*
 Print a page/slice of a 3D matrix
 * @param A, row, column: a ? x row x column matrix
 * @param page: assign page to print, start with 0
*/
void print3dPage(const int *A, int row, int column, int page) {
  int i, j;
  for (i = 0; i < row; ++i) {
    for (j = 0; j < column; ++j) {
      printf("%8d", A[page * row * column + i * column + j]);
    }
    printf("\n");
  }
}

/*
 Print words from indices
 * @param indices: word indices in a dictionary/vocabulary
 * @param length: length of the index string
 * @param vocabulary: the dictionary mapping between words and indices
*/
void printWordFromIndex(const int *indices, int length,
                        const char (*vocabulary)[20]) {
  int i;
  for (i = 0; i < length; ++i) {
    printf("%s ", vocabulary[indices[i]]);
  }
}

void printHelp(const char *appName) {
  printf(
      "Usage:\n"
      "  %s info model.bin               | Show info of pretrained model.\n"
      "  %s train model.bin              | Train from scratch and save model.\n"
      "  %s train pretrain.bin model.bin | Read pretrained data, finetune it, "
      "& "
      "save model.\n"
      "  %s forward pretrain.bin         | Read pretrained data and do "
      "inferences.\n"
      "Or\n"
      "  make info load=model.bin\n"
      "  make train save=model.bin\n"
      "  make train load=pretrain.bin save=model.bin\n"
      "  make forward load=pretrain.bin\n",
      appName, appName, appName, appName);
}

void printModelInfo(const WordNet *model) {
  printf("## Model Info\n");
  printf("   Mini-batch size          = %8d\n", model->hp.miniBatchSize);
  printf("   Layer 1 Neurons          = %8d\n", model->hp.layer1Neurons);
  printf("   Layer 2 Neurons          = %8d\n", model->hp.layer2Neurons);
  printf("   Training epochs          = %8d\n", model->hp.epoch);
  printf("   Early stop @ iteration   = %8d\n", model->hp.earlyStopIteration);
  printf("   Momentum                 = %lf\n", model->hp.momentum);
  printf("   Learning rate            = %lf\n", model->hp.learningRate);
  printf("   Verify per iteration     = %8d\n", model->hp.verifyPerIterBatch);
  printf("   Raw training data rows   = %8d\n", model->hp.rawDataRow4Training);
  printf("   Raw validation data rows = %8d\n", model->hp.rawDataRow4Validation);
  printf("   Raw test data rows       = %8d\n", model->hp.rawDataRow4Test);
  printf("   Raw data columns         = %8d\n", model->hp.rawDataColumn);
  printf("   Input dimension          = %8d\n", model->hp.inputDimension);
  printf("   Vocabulary size          = %8d\n", model->hp.vocabSize);
}
