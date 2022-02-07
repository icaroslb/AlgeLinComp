import math
import numpy as np
import decomposicao as decomp
import regressao as regre

def Householder (M):
  A = M.copy()
  qtdColumns = np.shape(A)[1]

  RH = np.identity(qtdColumns)

  for j in range(qtdColumns - 2):
    H = contructHouseholder(A, j)
    A = H @ A @ H
    RH = RH @ H

    print("\n----------- Passo {} -----------\n".format(j + 1))
    print("Matriz de Householder:\n{}\n\nMatriz modificada:\n{}\n\nMatriz acumulada:\n{}\n".format(H, A, RH))
  
  return [A, RH]

def contructHouseholder(M, j):
  qtdLines = np.shape(M)[1]
  v = np.zeros((qtdLines, 1))
  w = np.zeros((qtdLines, 1))
  
  v[j+1:qtdLines, 0] = M[j+1:qtdLines, j]
  
  jSign = 1
  if (v[j+1, 0] > 0):
    jSign = -1
  
  lenghtV = np.linalg.norm(v)
  w[j+1, 0] = jSign * lenghtV

  N = v - w
  n = N / np.linalg.norm(N)
  print(n @ np.transpose(n))
  return np.eye(qtdLines) - (2.0 * (n @ n.transpose()))


def QR (M, P, eps):
  A = M.copy()
  [qtdLines, qtdColumns] = np.shape(A)

  [Q, R] = decomp.decomposicaoQR(A)
  A = R @ Q
  P = P @ Q
  previousAlfa = math.inf
  currentAlfa = math.sqrt(sum([(A[i, i] * A[i, i]) for i in range(qtdColumns)]))

  # Enquanto os valores da diagonal não convergirem, decompõe a matriz
  while (abs(currentAlfa - previousAlfa) > eps):
    [Q, R] = decomp.decomposicaoQR(A)
    A = R @ Q
    P = P @ Q
    previousAlfa = currentAlfa
    currentAlfa = math.sqrt(sum([(A[i, i] * A[i, i]) for i in range(qtdColumns)]))
  
  diagonalMatrix = True

  for i in range(qtdLines):
    for j in range(qtdColumns):
      if (abs(M[i, j] - M[j, i]) > eps):
        print(M[i, j] - M[j, i])
        diagonalMatrix = False
        break
  
  # Se a matriz é diagonal, os autovalores são os valores da diagonal
  # e os autovetores são as matrizes Qs acumuladas
  if (diagonalMatrix):
    return [[A[i, i] for i in range(qtdColumns)], P]
  else:
    eigenValues = []
    eigenVectorsMatrix = np.zeros((qtdLines, qtdColumns))

    id = 0
    while (id < qtdLines):
      if ((id != qtdLines - 1) and (abs(A[id+1, id]) > eps)):
        [[a, b], [c, d]] = A[id:id+2, id:id+2]
        roots = np.roots([1, -(a + d), (a * d) - (b * c)])
        eigenValues.append(roots[0])
        eigenValues.append(roots[1])
        id += 1
      else:
        eigenValues.append(A[id, id])
      
      id += 1
    
    for i in range(qtdColumns):
      autoMatrix = A - (eigenValues[i] * np.eye(qtdColumns))
      eigenVectorsMatrix[:, i] = regre.minQuadrado(autoMatrix, np.zeros((qtdLines, 1))).transpose()
    
    return [eigenValues, eigenVectorsMatrix]

def main ():
  C = np.array([[2.0, 2.0, 3.0],[2.0, 5.0, 4.0],[3.0, 4.0, 6.0]])
  
  #[A, H] = Householder(C)
  [eigenValues, eigenVectors] = QR(C, np.eye(3), 0.000001)

  #for i in range(3):
  #  eigenVectors[:, i] = eigenVectors[:, i] / eigenVectors[2, i]
  #print("Matriz:\n{}\n\nHouseholder\n{}".format(A, H))
  print("Autovalores:\n{}\n\nAutovetores:\n{}".format(eigenValues, eigenVectors))

if __name__ == "__main__":
  main()