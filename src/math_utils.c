#include "math_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 Calculate matrix multiplication
 * @param A m x n
 * @param B n x p
 * @return result m x p
*/
void multiplyMatrix(const double *A, const double *B, int m, int n, int p,
                    double *result) {
  int i, j, k;
  double sum;

  for (i = 0; i < m; ++i) {
    for (k = 0; k < p; ++k) {
      sum = 0.0;
      for (j = 0; j < n; ++j) {
        sum += A[i * n + j] * B[j * p + k];
      }
      result[i * p + k] = sum;
    }
  }
}

/*
 Calculate matrix C = C + A * B
 * @param A m x n
 * @param B n x p
 * @return C += A * B, m x p
*/
void multiplyAddMatrix(const double *A, const double *B, int m, int n, int p,
                       double *C) {
  int i, j, k;
  double sum;

  for (i = 0; i < m; ++i) {
    for (k = 0; k < p; ++k) {
      sum = 0.0;
      for (j = 0; j < n; ++j) {
        sum += A[i * n + j] * B[j * p + k];
      }
      C[i * p + k] += sum;
    }
  }
}

/*
 Calculate matrix subtraction
 * @param A: m x n
 * @param B: m x n
 * @return result = A - B, m x n, can overwrite A or B
 */
void subtractMatrix(const double *A, const double *B, int m, int n, double *result) {
  int i, length = m * n;
  for (i = 0; i < length; ++i) {
    result[i] = A[i] - B[i];
  }
}

/*
 Calculate matrix scaled subtraction
 * @param A: m x n
 * @param B: m x n
 ! @param scale: scaling operation doesn't modify matrices!
 * @return result = A - scale * B, m x n, can overwrite A or B
 */
void subtractScaleMatrix(const double *A, const double *B, double scale, int m,
                         int n, double *result) {
  int i, length = m * n;
  for (i = 0; i < length; ++i) {
    result[i] = A[i] - scale * B[i];
  }
}

/*
 Calculate matrix sum
 * @param A: m x n
 * @param B: m x n
 * @return result = A + B, m x n, can overwrite A or B
 */
void addMatrix(const double *A, const double *B, int m, int n, double *result) {
  int i, length = m * n;
  for (i = 0; i < length; ++i) {
    result[i] = A[i] + B[i];
  }
}

/*
 Transpose a matrix
 * @param A: m x n
 * @return B: n x m
 */
void transposeMatrix(const double *A, int m, int n, double *B) {
  int i, j;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      B[j * m + i] = A[i * n + j];
    }
  }
}

/*
 Scale a matrix
 * @param A: m x n
 * @param c: a constant
 * @return A: cA
 */
void scaleMatrix(double *A, int m, int n, double c) {
  int i, length = m * n;
  for (i = 0; i < length; ++i) {
    A[i] *= c;
  }
}

/*
 Sum a matrix to a column vector
 * @param A: m x n
 * @return v: m x 1
 */
void sumColumn2Vector(const double *A, int m, int n, double *v) {
  int i, j;
  memset(v, 0, m * sizeof(double));
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      v[i] += A[i * n + j];
    }
  }
}

/*
 Gaussian random number generator
*/
double gaussRand(double mean, double std) {
  static double V1, V2, S;
  static int phase = 0;
  double X, U1, U2;

  if (phase == 0) {
    do {
      U1 = (double)rand() / RAND_MAX;
      U2 = (double)rand() / RAND_MAX;
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);
  } else {
    X = V2 * sqrt(-2 * log(S) / S);
  }

  phase = 1 - phase;
  return X * std + mean;
}

/*
 Entry-wise sigmoid function
 ! A will be overwritten
 */
void sigmoid(double *A, int m, int n) {
  int i, j;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      A[i * n + j] = 1.0 / (1.0 + exp(-A[i * n + j]));
    }
  }
}

/*
 Column-wise softmax function
 ! A will be overwritten
 */
void softmax(double *A, int m, int n) {
  int i, j;
  double max, sum;

  for (j = 0; j < n; ++j) {
    for (i = 1, max = A[0 * n + j]; i < m; ++i) {
      if (A[i * n + j] > max) {
        max = A[i * n + j];
      }
    }

    sum = 0.0;
    for (i = 0; i < m; ++i) {
      A[i * n + j] -= max;
      A[i * n + j] = exp(A[i * n + j]);
      sum += A[i * n + j];
    }

    for (i = 0; i < m; ++i) {
      A[i * n + j] /= sum;
    }
  }
}

/*
 Average cross entropy of two softmax batches
 * @param Each column of A & B is a softmax correspondingly, B in log
 * @return Average of all column cross entropies
 */
double averageCrossEntropy(const double *A, const double *B, int row,
                           int column) {
  const double tiny = exp(-30);
  double sum = 0.0;
  int length = row * column;
  int i;

  for (i = 0; i < length; ++i) {
    sum += A[i] * log(B[i] + tiny);
  }

  return -sum / column;
}