#ifndef _MATH_UTILS
#define _MATH_UTILS

/*
 Calculate matrix multiplication
 * @param A m x n
 * @param B n x p
 * @return result m x p
*/
void multiplyMatrix(const double *A, const double *B, int m, int n, int p,
                    double *result);

/*
 Calculate matrix C = C + A * B
 * @param A m x n
 * @param B n x p
 * @return C += A * B, m x p
*/
void multiplyAddMatrix(const double *A, const double *B, int m, int n, int p,
                       double *C);

/*
 Calculate matrix subtraction
 * @param A: m x n
 * @param B: m x n
 * @return result = A - B, m x n, can overwrite A or B
 */
void subtractMatrix(const double *A, const double *B, int m, int n,
                    double *result);

/*
 Calculate matrix scaled subtraction
 * @param A: m x n
 * @param B: m x n
 ! @param scale: scaling operation doesn't modify matrices!
 * @return result = A - scale * B, m x n, can overwrite A or B
 */
void subtractScaleMatrix(const double *A, const double *B, double scale, int m,
                         int n, double *result);

/*
 Calculate matrix sum
 * @param A: m x n
 * @param B: m x n
 * @return result = A + B, m x n, can overwrite A or B
 */
void addMatrix(const double *A, const double *B, int m, int n, double *result);

/*
 Transpose a matrix
 * @param A: m x n
 * @return B: n x m
 */
void transposeMatrix(const double *A, int m, int n, double *B);

/*
 Scale a matrix
 * @param A: m x n
 * @param c: a constant
 * @return A: cA
 */
void scaleMatrix(double *A, int m, int n, double c);

/*
 Sum a matrix to a column vector
 * @param A: m x n
 * @return v: m x 1
 */
void sumColumn2Vector(const double *A, int m, int n, double *v);

/*
 Gaussian random number generator
*/
double gaussRand(double mean, double std);

/*
 Entry-wise sigmoid function
 ! A will be overwritten
 */
void sigmoid(double *A, int m, int n);

/*
 Column-wise softmax function
 ! A will be overwritten
 */
void softmax(double *A, int m, int n);

/*
 Average cross entropy of two softmax batches
 * @param Each column of A & B is a softmax correspondingly
 * @return Average of all column cross entropies
 */
double averageCrossEntropy(const double *A, const double *B, int row,
                           int column);

#endif