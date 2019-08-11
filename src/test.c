#include "test.h"
#include "math_utils.h"

double test(WordNet *model, const int *batchInput4Test,
            const int *batchTarget4Test, int batchNum4Test) {
  int iterTestBatch;
  const int *currentTestBatch, *currentTestTargetBatch;
  double temp;
  double crossEntropy4Test = 0.0;

  for(iterTestBatch = 0; iterTestBatch < batchNum4Test; ++iterTestBatch) {
    currentTestBatch =
        &batchInput4Test[iterTestBatch * model->hp.miniBatchSize * model->hp.inputDimension];
    currentTestTargetBatch =
        &batchTarget4Test[iterTestBatch * model->hp.miniBatchSize *
                          (model->hp.rawDataColumn - model->hp.inputDimension)];
    forwardPropagate(model, currentTestBatch, model->N);
    loadTarget2TargetVectorBatch(model, currentTestTargetBatch);
    temp = averageCrossEntropy(model->targetVectorBatch, model->outputStateBatch,
                               model->hp.vocabSize, model->hp.miniBatchSize);
    crossEntropy4Test += (temp - crossEntropy4Test) / (iterTestBatch + 1);
  }

  return crossEntropy4Test;
}
