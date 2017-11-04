# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import math
import cmath
import numpy as np

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        F = np.empty((len(matrix),len(matrix)), dtype=complex)

        for i in range(0,len(matrix-1)):
            u = 0
            u = u+1
            sums = 0.0
            for j in range (0,len(matrix-1)):
                v = j
                angle = 2 * cmath.pi * (float(i * u) + float(j * v)) * 1 / len(matrix)
                sums =  matrix[i][j]* (math.cos((angle)) - 1j * math.sin((angle)))
                F[u][v] = sums
                print(F[u][v], end=' ')
            print()


    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        
        Inv = np.empty((len(matrix), len(matrix)), dtype=complex)

        for u in range(0, len(matrix - 1)):
            i = 0
            i += 1
            sums = 0.0
            for v in range(0, len(matrix - 1)):
                j = v
                angle = 2 * cmath.pi * (float(i * u) + float(j * v)) * 1 / len(matrix)
                sums = sums + (matrix[i][j] * (cmath.cos(angle) + 1j * cmath.sin(angle)))

                Inv[i][j] = sums
                print(Inv[i][j], end=' ')
            print()




    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        F = np.empty((len(matrix), len(matrix)), dtype=float)
        for i in range(0, len(matrix - 1)):
            u = 0
            u += 1
            sums = 0.0
            for j in range(0, len(matrix - 1)):
                v = j
                angle = 2 * math.pi * ((i * u) + (j * v)) * 1 / len(matrix)
                sums = sums + (matrix[i][j] * (math.cos(angle)))

                print(F[u][v], end=' ')
            print()



    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        F = np.empty((len(matrix), len(matrix)), dtype=complex)

        for i in range(0, len(matrix - 1)):
            u = 0
            u += 1
            sums = 0.0
            for j in range(0, len(matrix - 1)):
                v = j
                angle = 2 * cmath.pi * (float(i * u) + float(j * v)) * 1 / len(matrix)
                sums = sums + (matrix[i][j] * (cmath.cos(angle) - 1j * cmath.sin(angle)))

                F[u][v] = sums
                print(abs(F[u][v]), end=' ')
                print()
