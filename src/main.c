#include "load_data.h"
#include "math_utils.h"
#include "model.h"
#include "print_utils.h"
#include "rw_model.h"
#include "test.h"
#include "train.h"
#include "forward_ui.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

enum RunType { INFO, TRAIN, FORWARD, UNKNOWN };

int main(int argc, char **argv) {
  // Script tunable variables
  WordNet model;
  // Hyperparameters: tunable
  model.hp.miniBatchSize = 100; // N
  model.hp.layer1Neurons = 50;  // h1
  model.hp.layer2Neurons = 200; // h2
  model.hp.epoch = 9;
  // -- Stop when iterBatch == earlyStopIteration
  //           && iterEpoch == epoch - 1
  model.hp.earlyStopIteration = 3000;
  model.hp.momentum = 0.9;
  model.hp.learningRate = 0.1;
  model.hp.verifyPerIterBatch = INT32_MAX;
  // Hyperparameters: fixed in this example
  model.hp.rawDataRow4Training = 372550;
  model.hp.rawDataRow4Validation = 46568;
  model.hp.rawDataRow4Test = 46568;
  model.hp.rawDataColumn = 4;
  model.hp.inputDimension = 3;
  model.hp.vocabSize = 250;

  // Local variables
  const char *writeModelFileName = NULL;
  const char *readModelFileName = NULL;
  int *batchInput4Training, *batchTarget4Training; // For training
  int batchNum4Training;
  int *batchInput4Validation, *batchTarget4Validation; // For validation
  int batchNum4Validation;
  int *batchInput4Test, *batchTarget4Test; // For test
  int batchNum4Test;

  enum RunType runType = UNKNOWN;

  if (argc == 3) {
    if (!strcmp(argv[1], "info")) {
      // %s info model.bin
      // Show info of pretrained model
      runType = INFO;
      readModelFileName = argv[2];
    } else if (!strcmp(argv[1], "train")) {
      // %s train model.bin
      // Train from scratch and save model
      runType = TRAIN;
      writeModelFileName = argv[2];
    } else if (!strcmp(argv[1], "forward")) {
      // %s forward pretrained.bin
      // Read pretrained model and do inferences
      runType = FORWARD;
      readModelFileName = argv[2];
    } else {
      printf("## Argument error!\n");
      printHelp(argv[0]);
      return -1;
    }
  } else if (argc == 4) {
    // %s train pretrain.bin model.bin
    // Read pretrained data, finetune it, & save model
    if (!strcmp(argv[1], "train")) {
      runType = TRAIN;
      readModelFileName = argv[2];
      writeModelFileName = argv[3];
    } else {
      printf("## Argument error!\n");
      printHelp(argv[0]);
      return -1;
    }
  } else {
    printf("## Argument error!\n");
    printHelp(argv[0]);
    return -1;
  }

  switch (runType) {
  case INFO:
    readAndCreateModel(readModelFileName, &model);
    printModelInfo(&model);
    break;
  case TRAIN:
    printf("# Load all data\n");
    loadAllData();
    printf("# Create [Training] mini-batches\n");
    createMiniBatch(&g_train_data[0][0], model.hp.rawDataRow4Training,
                    model.hp.rawDataColumn, model.hp.inputDimension,
                    model.hp.miniBatchSize, &batchInput4Training,
                    &batchTarget4Training, &batchNum4Training);
    printf("  Batch number    = %8d\n", batchNum4Training);

    printf("# Create [Validation] mini-batches\n");
    createMiniBatch(&g_validation_data[0][0], model.hp.rawDataRow4Validation,
                    model.hp.rawDataColumn, model.hp.inputDimension,
                    model.hp.miniBatchSize, &batchInput4Validation,
                    &batchTarget4Validation, &batchNum4Validation);
    printf("  Batch number    = %8d\n", batchNum4Validation);

    printf("# Create [Test] mini-batches\n");
    createMiniBatch(&g_test_data[0][0], model.hp.rawDataRow4Test,
                    model.hp.rawDataColumn, model.hp.inputDimension,
                    model.hp.miniBatchSize, &batchInput4Test, &batchTarget4Test,
                    &batchNum4Test);
    printf("  Batch number    = %8d\n", batchNum4Test);

    if (argc == 3) {
      // Train from scratch
      printf("# Create WordNet:\n");
      createWordNet(&model);
    } else {
      // Read pretrained data
      printf("# Read model: %s\n", readModelFileName);
      readAndCreateModel(readModelFileName, &model);
    }

    printModelInfo(&model);
    printf("# Start training ...\n");
    train(&model, batchInput4Training, batchTarget4Training, batchNum4Training,
          batchInput4Validation, batchTarget4Validation, batchNum4Validation);
    printf("# Finish training ...\n");

    printf("##Start testing ...\n");
    printf("##Test CE = %.8lf\n",
           test(&model, batchInput4Test, batchTarget4Test, batchNum4Test));

    printf("# Saving model to %s\n", writeModelFileName);
    writeModel(&model, writeModelFileName);

    destroyMiniBatch(batchInput4Training, batchTarget4Training);
    destroyMiniBatch(batchInput4Validation, batchTarget4Validation);
    destroyMiniBatch(batchInput4Test, batchTarget4Test);

    break;
  case FORWARD:
    printf("# Load all data\n");
    loadAllData();
    printf("# Load model: %s\n", readModelFileName);
    readAndCreateModel(readModelFileName, &model);
    printModelInfo(&model);
    forward_ui(&model);
    break;
  default:
    printf("## Unknown error!\n");
    return -1;
  }

  destroyWordNet(&model);
  return 0;
}
