#include "model.h"
#include "math_utils.h"
#include <stdio.h> // for debug
#include <stdlib.h>
#include <string.h>

/*
 Create and initialize a WordNet
 */
void createWordNet(WordNet *model) {
  int i;

  // Assign shortcuts
  int D = model->hp.inputDimension;
  int N = model->hp.miniBatchSize;
  int h1 = model->hp.layer1Neurons;
  int h2 = model->hp.layer2Neurons;
  int vocabSize = model->hp.vocabSize;
  model->D = D;
  model->N = N;
  model->vocabSize = vocabSize;
  model->h1 = h1;
  model->h2 = h2;

  // Allocate memory
  model->W1 = malloc(h1 * vocabSize * sizeof(double));
  model->dW1 = calloc(h1 * vocabSize, sizeof(double));
  model->inputWordVectorBatch = malloc(vocabSize * (D * N) * sizeof(double));
  model->xit = malloc(N * vocabSize * sizeof(double));
  model->bufferW1xInputWordVectorBatch = malloc(h1 * (D * N) * sizeof(double));
  model->layer1StateBatch = malloc((h1 * D) * N * sizeof(double));
  model->y1t = malloc(N * (h1 * D) * sizeof(double));

  model->W2 = malloc(h2 * (h1 * D) * sizeof(double));
  model->dW2 = calloc(h2 * (h1 * D), sizeof(double));
  model->W2t = malloc((h1 * D) * h2 * sizeof(double));
  model->layer2StateBatch = malloc(h2 * N * sizeof(double));
  model->y2t = malloc(N * h2 * sizeof(double));
  model->bias2 = calloc(h2 * 1, sizeof(double)); // Set to Zeros
  model->db2 = calloc(h2 * 1, sizeof(double));   // Set to Zeros

  model->W3 = malloc(vocabSize * h2 * sizeof(double));
  model->dW3 = calloc(vocabSize * h2, sizeof(double));
  model->W3t = malloc(h2 * vocabSize * sizeof(double));
  model->layer3StateBatch = malloc(vocabSize * N * sizeof(double));
  model->bias3 = calloc(vocabSize * 1, sizeof(double)); // Set to Zeros
  model->db3 = calloc(vocabSize * 1, sizeof(double));   // Set to Zeros

  model->outputStateBatch = model->layer3StateBatch;

  /*
   * Fill Gaussian random numbers to
   * W1, W2, W3
   * mean = 0, std = 0.01
   */
  for (i = 0; i < h1 * vocabSize; ++i) {
    model->W1[i] = gaussRand(0.0, 0.01);
  }
  for (i = 0; i < h2 * (h1 * D); ++i) {
    model->W2[i] = gaussRand(0.0, 0.01);
  }
  for (i = 0; i < vocabSize * h2; ++i) {
    model->W3[i] = gaussRand(0.0, 0.01);
  }

  //! Following: for back-propagation
  model->targetVectorBatch = malloc(vocabSize * N * sizeof(double));
  model->yt = model->targetVectorBatch;

  model->dLdy3_ = malloc(vocabSize * N * sizeof(double));
  model->dLdy3_t = malloc(vocabSize * N * sizeof(double));
  model->dLdW3 = malloc(vocabSize * h2 * sizeof(double));
  model->dLdb3 = malloc(vocabSize * 1 * sizeof(double));
  model->dLdy2 = malloc(h2 * N * sizeof(double));
  model->dLdy2_ = malloc(h2 * N * sizeof(double));
  model->dLdb2 = malloc(h2 * 1 * sizeof(double));
  model->dLdW2 = malloc(h2 * (D * h1) * sizeof(double));
  model->dLdy1 = malloc((h1 * D) * N * sizeof(double));
  model->dLdy1i = malloc(h1 * N * sizeof(double));
  model->dLdW1 = malloc(h1 * vocabSize * sizeof(double));
}

/*
 Destroy a WordNet
 */
void destroyWordNet(WordNet *model) {
  free(model->W1);
  free(model->dW1);
  free(model->inputWordVectorBatch);
  free(model->xit);
  free(model->bufferW1xInputWordVectorBatch);
  free(model->layer1StateBatch);
  free(model->y1t);

  free(model->W2);
  free(model->dW2);
  free(model->W2t);
  free(model->layer2StateBatch);
  free(model->y2t);
  free(model->bias2);
  free(model->db2);

  free(model->W3);
  free(model->dW3);
  free(model->W3t);
  free(model->layer3StateBatch);
  free(model->bias3);
  free(model->db3);

  free(model->targetVectorBatch);
  free(model->dLdy3_);
  free(model->dLdy3_t);
  free(model->dLdW3);
  free(model->dLdb3);
  free(model->dLdy2);
  free(model->dLdy2_);
  free(model->dLdb2);
  free(model->dLdW2);
  free(model->dLdy1);
  free(model->dLdy1i);
  free(model->dLdW1);
}

/*
 Forward propagation for batch training
 * @param model
 * @param miniBatchInput: n x inputDimension, where
 !        n <= model->N
 *        miniBatchSize == model->N,
 *        inputDimension == model->D
 * @return model->outputStateBatch: vocabSize x N
 *         outputStateBatch
 *         = softmax(W3 * (sigmoid(W2*layer1StateBatch + bias2)) + bias3)
 */
void forwardPropagate(WordNet *model, const int *miniBatchInput, int n) {
  int i, j, k, l, index;

  int N = n;        //! n <= model->N
  int D = model->D; // Column

  // Construct inputWordVectorBatch: vocabSize x (D x N)
  memset(model->inputWordVectorBatch, 0,
         model->vocabSize * D * N * sizeof(double));
  k = 0;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < D; ++j) {
      index = miniBatchInput[i * D + j] - 1; // MATLAB index starts from 1
      model->inputWordVectorBatch[index * (D * N) + k] = 1.0;
      ++k;
    }
  }

  // Calculate
  // bufferW1xInputWordVectorBatch = W1 * inputWordVectorBatch
  // We get an h1 x (D x N) matrix
  multiplyMatrix(model->W1, model->inputWordVectorBatch, model->h1,
                 model->vocabSize, D * N, model->bufferW1xInputWordVectorBatch);

  /*
   * Now merge every D columns into a new column,
   * which is the word embedding layer,
   * and bufferW1xInputWordVectorBatch becomes layer1StateBatch,
   * a (h1 x D) x N matrix
   */
  k = 0, l = 0;
  for (j = 0; j < D * N; ++j) {
    for (i = 0; i < model->h1; ++i) {
      model->layer1StateBatch[k++ * N + l] =
          model->bufferW1xInputWordVectorBatch[i * (D * N) + j];
      if (k == model->h1 * D) {
        k = 0;
        ++l;
      }
    }
  }

  /*
   * Calculate W2 * layer1StateBatch
   * (h2 x (h1 x D)) x ((h1 x D) x N)
   * Using layer2StateBatch as a buffer
   * an h2 x N matrix
   */
  multiplyMatrix(model->W2, model->layer1StateBatch, model->h2, model->h1 * D,
                 N, model->layer2StateBatch);

  // Add bias2 to each column of layer2StateBatch
  for (j = 0; j < N; ++j) {
    for (i = 0; i < model->h2; ++i) {
      model->layer2StateBatch[i * N + j] += model->bias2[i];
    }
  }

  /*
   * Calculate final
   * layer2StateBatch = sigmoid(W2 * layer1StateBatch + bias2)
   */
  sigmoid(model->layer2StateBatch, model->h2, N);

  /*
   * Calculate W3 * layer2StateBatch
   * (vocabSize x h2) x (h2 x N)
   * Using layer3StateBatch as a buffer
   * a vocabSize x N matrix
   */
  multiplyMatrix(model->W3, model->layer2StateBatch, model->vocabSize,
                 model->h2, N, model->layer3StateBatch);

  // Add bias3 to each column of layer3StateBatch
  for (j = 0; j < N; ++j) {
    for (i = 0; i < model->vocabSize; ++i) {
      model->layer3StateBatch[i * N + j] += model->bias3[i];
    }
  }

  // Calculate softmax for each column of layer3StateBatch
  softmax(model->layer3StateBatch, model->vocabSize, N);
}

/*
 Load index target batch and convert to one-hot batch
 * @param targetBatch: N x 1
 * @return model->targetVectorBatch, vocabSize x N
 */
void loadTarget2TargetVectorBatch(WordNet *model, const int *targetBatch) {
  int i, index;
  memset(model->targetVectorBatch, 0, model->vocabSize * model->N * sizeof(double));
  for (i = 0; i < model->N; ++i) {
    index = targetBatch[i] - 1; // MATLAB index starts from 1
    model->targetVectorBatch[index * model->N + i] = 1.0;
  }
}

/*
 Back propagation for batch training
 * @param model
 * @return
 */
void backPropagate(WordNet *model) {
  /*
   * y1 = [ W1X1 ]
   *      [ W1X2 ]
   *      [ W1X3 ]
   * y2_ = W2y1 + b2
   * y2  = sigmoid(y2_)
   * y3_ = W3y2 + b3
   * y3  = softmax(y3_)
   ! output: y3
   ! target: yt
   ! Loss = - sum( yt .* log(y3) )
   */

  int i, j, k, l;

  // dL/dy3_ batch: vocabSize x N
  // dL/dy3_ = y3 - yt
  // dLdy3_t: transpose of dLdy3_
  subtractMatrix(model->outputStateBatch, model->yt, model->vocabSize, model->N,
                 model->dLdy3_);
  transposeMatrix(model->dLdy3_, model->vocabSize, model->N, model->dLdy3_t);

  // dL/dW3: vocabSize x h2
  // dL/dW3 = dL/dy3_ * (dy3_/dW3)'
  //        = dLdy3_ * y2'
  // (vocabSize x N) x (N x h2)
  transposeMatrix(model->layer2StateBatch, model->h2, model->N, model->y2t);
  multiplyMatrix(model->dLdy3_, model->y2t, model->vocabSize, model->N,
                 model->h2, model->dLdW3);

  /*
   * dL/db3 = dy3_/db3 * dL/dy3_
   *        = dL/dy3_
   * dL/db3 = sumColumn2Vector(dL/dy3_)
   * vocabSize x 1
   */
  sumColumn2Vector(model->dLdy3_, model->vocabSize, model->N, model->dLdb3);

  // dL/dy2: h2 x N
  // dL/dy2 = dy3_/dy2 * dL/dy3_
  //        = W3' * dLdy3_
  // (h2 x vocabSize) x (vocabSize x N)
  transposeMatrix(model->W3, model->vocabSize, model->h2, model->W3t);
  multiplyMatrix(model->W3t, model->dLdy3_, model->h2, model->vocabSize,
                 model->N, model->dLdy2);

  // dLdy2_: h2 x N
  // dL/dy2_ = dy2/dy2_ * dL/dy2
  //         = y2 .* (1 - y2) .* dLdy2
  // (h2 x N) .* (h2 x N) .* (h2 x N)
  j = model->h2 * model->N;
  for (i = 0; i < j; ++i) {
    model->dLdy2_[i] = model->dLdy2[i] * model->layer2StateBatch[i] *
                       (1.0 - model->layer2StateBatch[i]);
  }

  /*
   * dL/db2 = dy2_/db2 * dL/dy2_
   *        = dL/dy2_
   * dL/db2 = sumColumn2Vector(dL/dy2_)
   * h2 x 1
   */
  sumColumn2Vector(model->dLdy2_, model->h2, model->N, model->dLdb2);

  // dL/dW2: h2 x (D x h1)
  // dL/dW2 = dL/dy2_ * (dy2_/dW2)'
  //        = dLdy2_ * y1'
  // (h2 x N) x (N x (h1 x D))
  transposeMatrix(model->layer1StateBatch, model->h1 * model->D, model->N,
                  model->y1t);
  multiplyMatrix(model->dLdy2_, model->y1t, model->h2, model->N,
                 model->h1 * model->D, model->dLdW2);

  // dL/dy1: (h1 x D) x N
  // dL/dy1 = dy2_/dy1 * dL/dy2_
  //        = W2' * dLdy2_
  // ((h1 x D) x h2) x (h2 x N)
  transposeMatrix(model->W2, model->h2, model->h1 * model->D, model->W2t);
  multiplyMatrix(model->W2t, model->dLdy2_, model->h1 * model->D, model->h2,
                 model->N, model->dLdy1);

  // dL/dW1: h1 x vocabSize
  // dL/dW1 = sum_{i=1}^D dL/dy1[i] * Xi'
  // (h1 x N) x (N x vocabSize)
  memset(model->dLdW1, 0, model->h1 * model->vocabSize * sizeof(double));
  for (k = 0; k < model->D; ++k) {
    // Fill xit
    for (i = 0; i < model->vocabSize; ++i) {
      for (j = k, l = 0; j < model->D * model->N; j += model->D, ++l) {
        model->xit[l * model->vocabSize + i] =
            model->inputWordVectorBatch[i * model->D * model->N + j];
      }
    }
    // Fill dLdy1i
    for (i = k * model->h1, l = 0; l < model->h1; ++i, ++l) {
      for (j = 0; j < model->N; ++j) {
        model->dLdy1i[l * model->N + j] = model->dLdy1[i * model->N + j];
      }
    }
    // Calculate dLdW1 += dLdy1i * xit
    multiplyAddMatrix(model->dLdy1i, model->xit, model->h1, model->N,
                      model->vocabSize, model->dLdW1);
  }
}

/*
 Update network parameters: weights and biases
 * @param model, momentum, learningRate
 * @return W3, bias3, W2, bias2, W1
 */
void updateNetworkParameters(WordNet *model, double momentum,
                             double learningRate) {
  // All parameters have been initialized to zeros
  int rows, columns;

  // Upadte: W3 = W3 - learningRate * dW3, where
  // dW3 = momentum * dW3 + dLdW3 / N
  rows = model->vocabSize;
  columns = model->h2;
  scaleMatrix(model->dW3, rows, columns, momentum);
  scaleMatrix(model->dLdW3, rows, columns, 1.0 / model->N);
  addMatrix(model->dW3, model->dLdW3, rows, columns, model->dW3);
  subtractScaleMatrix(model->W3, model->dW3, learningRate, rows, columns, model->W3);

  // Upadte: bias3 = bias3 - learningRate * db3, where
  // db3 = momentum * db3 + dLdb3 / N
  rows = model->vocabSize;
  columns = 1;
  scaleMatrix(model->db3, rows, columns, momentum);
  scaleMatrix(model->dLdb3, rows, columns, 1.0 / model->N);
  addMatrix(model->db3, model->dLdb3, rows, columns, model->db3);
  subtractScaleMatrix(model->bias3, model->db3, learningRate, rows, columns, model->bias3);

  // Upadte: W2 = W2 - learningRate * dW2, where
  // dW2 = momentum * dW2 + dLdW2 / N
  rows = model->h2;
  columns = model->h1 * model->D;
  scaleMatrix(model->dW2, rows, columns, momentum);
  scaleMatrix(model->dLdW2, rows, columns, 1.0 / model->N);
  addMatrix(model->dW2, model->dLdW2, rows, columns, model->dW2);
  subtractScaleMatrix(model->W2, model->dW2, learningRate, rows, columns, model->W2);

  // Upadte: bias2 = bias2 - learningRate * db2, where
  // db2 = momentum * db2 + dLdb2 / N
  rows = model->h2;
  columns = 1;
  scaleMatrix(model->db2, rows, columns, momentum);
  scaleMatrix(model->dLdb2, rows, columns, 1.0 / model->N);
  addMatrix(model->db2, model->dLdb2, rows, columns, model->db2);
  subtractScaleMatrix(model->bias2, model->db2, learningRate, rows, columns, model->bias2);

  // Upadte: W1 = W1 - learningRate * dW1, where
  // dW1 = momentum * dW1 + dLdW1 / N
  rows = model->h1;
  columns = model->vocabSize;
  scaleMatrix(model->dW1, rows, columns, momentum);
  scaleMatrix(model->dLdW1, rows, columns, 1.0 / model->N);
  addMatrix(model->dW1, model->dLdW1, rows, columns, model->dW1);
  subtractScaleMatrix(model->W1, model->dW1, learningRate, rows, columns, model->W1);
}
