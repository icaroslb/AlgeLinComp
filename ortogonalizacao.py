import numpy as np
import decomposicao as decomp

def gramSchmidt(M, eps):
  A = M.copy()
  qtdLines = np.shape(A)[0]
  qtdColumns = np.shape(A)[1]
  identity = np.identity(qtdLines)

  AExtended = np.concatenate((A, identity), axis=1)
  [ARREF, rank, dimentionNull, columnPivot] = decomp.RREF(AExtended, eps)

  A = np.zeros((qtdLines, qtdLines))
  for i in range(qtdLines):
    A[:, i] = AExtended[:, columnPivot[i]]
  
  A[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
  
  for i in range(1, qtdLines):
    A[:, i] = ortogonalizacaoVetor(A[:, i], A[:, 0:i])
    A[:, i] = A[:, i] / np.linalg.norm(A[:, i])
  
  return A  

def ortogonalizacaoVetor(v, conjV):
  u = v.copy()
  qtdVectors = np.shape(conjV)[1]
  for i in range(qtdVectors):
    u = u - (np.dot(v, conjV[:, i]) * conjV[:, i])
  
  return u

def main():
  M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
  
  A = gramSchmidt(M)

  print(A)

if __name__ == "__main__":
  main()