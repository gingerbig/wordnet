#include "load_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 Global variables: definition
*/
int g_train_data[372550][4];
int g_validation_data[46568][4];
int g_test_data[46568][4];
char g_vocabulary[251][20];

/*
 Load all data to global variables
 * data/train_data.csv => g_train_data
 * data/valid_data.csv => g_validation_data
 * data/test_data.csv  => g_test_data
 * data/vocab.txt      => g_vocabulary
*/
void loadAllData(void) {
  FILE *fpTrainData, *fpValidData, *fpTestData, *fpVocab;
  int count;

  fpTrainData = fopen("data/train_data.csv", "r");
  fpValidData = fopen("data/valid_data.csv", "r");
  fpTestData = fopen("data/test_data.csv", "r");
  fpVocab = fopen("data/vocab.txt", "r");

  count = 0;
  while (4 == fscanf(fpTrainData, "%d,%d,%d,%d\n", &g_train_data[count][0],
                     &g_train_data[count][1], &g_train_data[count][2],
                     &g_train_data[count][3])) {
    ++count;
  }

  count = 0;
  while (4 == fscanf(fpValidData, "%d,%d,%d,%d\n", &g_validation_data[count][0],
                     &g_validation_data[count][1], &g_validation_data[count][2],
                     &g_validation_data[count][3])) {
    ++count;
  }

  count = 0;
  while (4 == fscanf(fpTestData, "%d,%d,%d,%d\n", &g_test_data[count][0],
                     &g_test_data[count][1], &g_test_data[count][2],
                     &g_test_data[count][3])) {
    ++count;
  }

  count = 1;
  while (fgets(g_vocabulary[count], 20, fpVocab)) {
    g_vocabulary[count][strlen(g_vocabulary[count]) - 1] = '\0';
    ++count;
  }

  fclose(fpTrainData);
  fclose(fpValidData);
  fclose(fpTestData);
  fclose(fpVocab);
}

/*
 Create mini-batches for training
 * @param rawData, row, column, inputDimension: rawData is a row x column data
 matrix for training. Each row stands for one training datum: the first
 inputDimension columns are fed to the network, and the last (column -
 inputDimension) columns are the ground truth output.
 * @param miniBatchSize: how many items of training data should be contained in
 a mini-batch.
 * @return batchInput: a pointer to the input of mini-batches, batchNum x
 miniBatchSize x inputDimension
 * @return batchOutput: a pointer to the target output (i.e. ground truth) of
 mini-batches, batchNum x miniBatchSize x (column-inputDimension)
 * @return batchNum: after dividing the rawData with miniBatchSize, batchNum
 shows how many batches there are
*/
void createMiniBatch(const int *rawData, int row, int column,
                     int inputDimension, int miniBatchSize, int **batchInput,
                     int **batchOutput, int *batchNum) {
  int M = row / miniBatchSize;
  int N = miniBatchSize;
  int D = inputDimension;
  int i, j, k, count;
  int *pInputMatrix, *pOutputMatrx;

  *batchNum = M;

  // Imagine a book with M pages, and each page has N rows and D columns
  pInputMatrix = malloc(M * N * D * sizeof(int));
  *batchInput = pInputMatrix;

  // Imagine a book with M pages, and each page has N rows and (column -
  // inputDimension) columns
  pOutputMatrx = malloc(M * N * (column - D) * sizeof(int));
  *batchOutput = pOutputMatrx;

  count = 0;
  for (k = 0; k < M; ++k) {
    for (i = 0; i < N; ++i, ++count) {
      for (j = 0; j < D; ++j) {
        pInputMatrix[k * N * D + i * D + j] = rawData[count * column + j];
      }
    }
  }

  count = 0;
  for (k = 0; k < M; ++k) {
    for (i = 0; i < N; ++i, ++count) {
      for (j = 0; j < column-D; ++j) {
        pOutputMatrx[k * N * (column-D) + i * (column-D) + j] = rawData[count * column + j + D];
      }
    }
  }
}

/*
 Destroy the resourses used by mini-batches
 * @param batchInput
 * @param batchOutput
*/
void destroyMiniBatch(int *batchInput, int *batchOutput) {
  free(batchInput);
  free(batchOutput);
}