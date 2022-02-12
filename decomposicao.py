import numpy as np
import math
import similaridade as simi
import metodosPotencia as metPot

## decomposição.py possui as funções de decomposição
## main() apenas para testes

def LU (M):
  L = np.identity(np.shape(M)[0])
  U = np.identity(np.shape(M)[0])

  for j in range(np.shape(M)[0]):
    for i in range(j + 1):
      value = 0
      for k in range(i):
        value = value + (L[i, k] * U[k, j])
      U[i, j] = M[i, j] - value

    for i in range(j + 1, np.shape(M)[0]):
      value = 0
      for k in range(j):
        value = value + (L[i, k] * U[k, j])
      if (U[j, j] == 0):
        raise ValueError("Matriz não se decompõe em LU!")
      L[i, j] = (M[i, j] - value) / U[j, j]
  
  return (L, U)

def resolutionLU (L, U, b):
  y = np.zeros(np.shape(b))
  x = np.zeros(np.shape(b))
  size = np.shape(b)[0]

  for i in range(size):
    y[i, 0] = b[i, 0]
    for j in range(i):
      y[i, 0] = y[i, 0] - (L[i, j] * y[j, 0])

  for i in range(size - 1, -1, -1):
    x[i, 0] = y[i, 0]
    for j in range(size - 1, i, -1):
      x[i, 0] = x[i, 0] - (U[i, j] * x[j, 0])
      
    x[i, 0] = x[i, 0] / U[i, i]
  
  return x

def cholesky(M):
  S = np.identity(np.shape(M)[0])

  for j in range(np.shape(M)[1]):
    s = 0
    for k in range(j):
      s = s + (S[j, k] * S[j, k])
    if (M[j, j] <= s):
      raise ValueError("Matriz não é positiva definida!")
    
    S[j, j] = math.sqrt(M[j, j] - s)

    for i in range(j + 1, np.shape(M)[0]):
      s = 0
      for k in range(j):
        s = s + (S[i, k] * S[j, k])
      S[i, j] = (M[i, j] - s) / S[j, j]

  return S

def RREF (M, eps):
  eps = abs(eps)
  columnPivots = []

  rowDimention = np.shape(M)[0]
  columnDimention = np.shape(M)[1]
  minDimention = min(rowDimention, columnDimention)
  A = M.copy()

  ## Encontrando os pivôs de cada linha
  indiceRow = 0
  indiceColumn = 0

  while (indiceRow < rowDimention and indiceColumn < columnDimention):
    while (indiceColumn < columnDimention and abs(A[indiceRow, indiceColumn]) <= eps):
      changeLine = indiceRow
      greaterValue = 0

      for i in range(indiceRow + 1, rowDimention):
        if (abs(A[i, indiceColumn]) > eps and abs(A[i, indiceColumn]) > greaterValue):
          changeLine = i
          greaterValue = abs(A[i, indiceColumn])
      
      if (changeLine == indiceRow):
        indiceColumn += 1
      else:
        auxLine = A[indiceRow, :]
        A[indiceRow, :] = A[changeLine, :]
        A[changeLine, :] = auxLine
    
    if (indiceColumn < columnDimention):
      columnPivots.append(indiceColumn)
      A[indiceRow, :] = A[indiceRow, :] / A[indiceRow, indiceColumn]
      
      for i in range(indiceRow + 1, rowDimention):
        aux = A[i, indiceColumn]
        for j in range(indiceColumn, columnDimention):
          A[i, j] = A[i, j] - (aux * A[indiceRow, j])
    
    indiceRow += 1
  
  ## Zerando os vlores a cima do pivô
  indiceRow = 0
  indiceColumn = 0

  while (indiceRow < rowDimention and indiceColumn < columnDimention):
    while (indiceColumn < columnDimention and abs(A[indiceRow, indiceColumn]) <= eps):
      indiceColumn = indiceColumn + 1
    
    if (indiceColumn < columnDimention):
      for i in range(indiceRow - 1, -1, -1):
        aux = A[i, indiceColumn]
        for j in range(indiceColumn, columnDimention):
          A[i, j] = A[i, j] - (aux * A[indiceRow, j])

    indiceRow = indiceRow + 1
  
  ## Procurando o rank e a dimenção nula da matriz
  indiceRow = 0
  indiceColumn = 0
  rank = 0
  dimentionNull = 0

  while (indiceRow < rowDimention and indiceColumn < columnDimention) :
    if (abs(A[indiceRow, indiceColumn]) > eps):
      rank = rank + 1
      indiceRow = indiceRow + 1
      indiceColumn = indiceColumn + 1
    else:
      dimentionNull = dimentionNull + 1
      indiceColumn = indiceColumn + 1

  dimentionNull = dimentionNull + (columnDimention - indiceColumn)

  return [A, rank, dimentionNull, columnPivots]

def decomposicaoQR(M):
  qtdColumns = np.shape(M)[1]
  Q = np.eye(qtdColumns)
  R = M.copy()

  for j in range(qtdColumns - 1):
    Qj = contructQR(R, j)
    R = Qj @ R
    Q = Q @ Qj
  
  return [Q, R]

def contructQR(M, j):
  qtdLines = np.shape(M)[0]
  v = np.zeros((qtdLines, 1))
  w = np.zeros((qtdLines, 1))
  
  v[j:qtdLines, 0] = M[j:qtdLines, j]
  
  jSign = 1
  if (v[j, 0] > 0):
    jSign = -1
  
  lenghtV = np.linalg.norm(v)
  w[j, 0] = jSign * lenghtV

  N = v - w
  n = N / np.linalg.norm(N)
  
  return np.eye(qtdLines) - (2.0 * (n @ n.transpose()))

def decomposicaoSVD(M, eps):
  MtM = M.transpose() @ M
  MMt = M @ M.transpose()
  [MtMLines, MtMColumns] = np.shape(M)
  [MtMLines, MtMColumns] = np.shape(MtM)
  [MMtLines, MMtColumns] = np.shape(MMt)
  
  [MtMeigenValues, V] = simi.QR(MtM, np.eye(min(MtMLines, MtMColumns)), eps)
  [MMteigenValues, U] = simi.QR(MMt, np.eye(min(MMtLines, MMtColumns)), eps)

  S = np.zeros(np.shape(M))
  qtdEigenValues = min(np.shape(M))

  for i in range(qtdEigenValues):
    S[i, i] = np.sqrt(MtMeigenValues[i])

  return [U, S, V]

def main ():
  #M = np.array([[1, 2, 3], [4, 5, 6], [7, 7, 8]])
  #b = np.array([[1], [5], [7]])
  #(L, U) = LU(M)
  #x = resolutionLU(L, U, b)
  #print("\n\nL:\n{}\n\nU:\n{}\n\nL * U:\n{}\n\nx:\n{}\n\n LU * x:\n{}".format(L, U, L @ U, x, L @ U @ x))
  C = np.array([[2.0, 2.0, 3.0],[2.0, 3.0, 4.0],[3.0, 4.0, 6.0]])
  D = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
  
  #try:
  #  S = cholesky(C)
  #  print("Matriz:\n{}\n\nS:\n{}\n\nSt:\n{}\n\nS * St:\n{}".format(C, S, np.transpose(S), S @ np.transpose(S)))
  #except ValueError as error:
  #  print(error)
  
  #A = RREF(C, 0.0001)
  [U, S, V] = decomposicaoSVD(D, 0.0001)

  #print("Matriz:\n{}\nRank:\n{}\nEspaço nulo:\n{}\n".format(A[0], A[1], A[2]))
  print("U:\n{}\n\nS:\n{}\n\nV:\n{}\n\nU * S * Vt:\n{}\n".format(U, S, V, U @ S @ V.transpose()))


if __name__ == "__main__":
  main()