#include "train.h"
#include "math_utils.h"
#include <stdio.h>

void train(WordNet *model, const int *batchInput4Training,
           const int *batchTarget4Training, int batchNum4Training,
           const int *batchInput4Validation, const int *batchTarget4Validation,
           int batchNum4Validation) {
  int epoch = model->hp.epoch;
  int miniBatchSize = model->hp.miniBatchSize;
  int inputDimension = model->hp.inputDimension;
  int rawDataColumn = model->hp.rawDataColumn;
  int vocabSize = model->hp.vocabSize;
  int earlyStopIteration = model->hp.earlyStopIteration;
  double momentum = model->hp.momentum;
  double learningRate = model->hp.learningRate;
  int verifyPerIterBatch = model->hp.verifyPerIterBatch;

  int iterEpoch, iterBatch, iterValidationBatch;
  const int *currentInputBatch, *currentTargetBatch;
  const int *currentValidationBatch, *currentValidationTargetBatch;

  double temp;
  double crossEntropy4Training;
  double crossEntropy4Validation;

  int ceCounter; // Increasingly mark iteration steps

  ceCounter = 0;
  for (iterEpoch = 0; iterEpoch < epoch; ++iterEpoch) {
    printf("# Begin iteration Epoch = %d\n", iterEpoch);

    // Loop over mini-batches
    crossEntropy4Training = 0.0;
    for (iterBatch = 0; iterBatch < batchNum4Training;
         ++iterBatch, ++ceCounter) {
      // miniBatchSize x inputDimension, on page iterBatch
      currentInputBatch =
          &batchInput4Training[iterBatch * miniBatchSize * inputDimension];

      // miniBatchSize x 1, on page iterBatch
      currentTargetBatch =
          &batchTarget4Training[iterBatch * miniBatchSize *
                                (rawDataColumn - inputDimension)];

      // Do forward propagation. Now we get
      // model.outputStateBatch: vocabSize x N
      forwardPropagate(model, currentInputBatch, model->N);

      // Construct one-hot target batch to compare with model.outputStateBatch
      // Stored in model->targetVectorBatch, vocabSize x N
      loadTarget2TargetVectorBatch(model, currentTargetBatch); // yt

      // Measure average loss between target & output batch
      temp =
          averageCrossEntropy(model->targetVectorBatch, model->outputStateBatch,
                              vocabSize, miniBatchSize);
      crossEntropy4Training += (temp - crossEntropy4Training) / (iterBatch + 1);
      printf("  Training CE (@%d, minibatch %d of %d, epoch %d) = %.8lf\n",
             ceCounter + 1, iterBatch + 1, batchNum4Training, iterEpoch + 1,
             crossEntropy4Training);

      // If hits early stop condition, should exit loop
      if (iterEpoch == epoch - 1 && iterBatch == earlyStopIteration) {
        return;
      }

      // Do back propagation
      //* Calculate: dL/dW3, dL/db3, dL/dW2, dL/db2, dL/dW1
      backPropagate(model);

      // Update network parameters: weights and biases
      //* W3, bias3, W2, bias2, W1
      updateNetworkParameters(model, momentum, learningRate);

      // Validation
      if (iterBatch && iterBatch % verifyPerIterBatch == 0) {
        printf("##Start validation ...\n");
        crossEntropy4Validation = 0.0;
        for (iterValidationBatch = 0; iterValidationBatch < batchNum4Validation;
             ++iterValidationBatch) {
          currentValidationBatch =
              &batchInput4Validation[iterValidationBatch * miniBatchSize *
                                     inputDimension];
          currentValidationTargetBatch =
              &batchTarget4Validation[iterValidationBatch * miniBatchSize *
                                      (rawDataColumn - inputDimension)];
          forwardPropagate(model, currentValidationBatch, model->N);
          loadTarget2TargetVectorBatch(model, currentValidationTargetBatch);
          temp = averageCrossEntropy(model->targetVectorBatch,
                                     model->outputStateBatch, vocabSize,
                                     miniBatchSize);
          crossEntropy4Validation +=
              (temp - crossEntropy4Validation) / (iterValidationBatch + 1);
        }
        printf("##Validation CE (@%d) = %.8lf\n", ceCounter + 1,
               crossEntropy4Validation);
      }
    }
  }
}
