import numpy as np
import gauss

def SOR(M, omega, b, eps, maxIterations = np.Infinity):
  MShape = np.shape(M)
  L = np.zeros(MShape)
  D = np.zeros(MShape)
  U = np.zeros(MShape)

  for i in range(MShape[0]):
    for j in range(MShape[1]):
      if (i > j):
        L[i, j] = M[i, j]
      elif (i == j):
        D[i, j] = M[i, j]
      else:
        U[i, j] = M[i, j]
  
  DL = D + (omega * L)
  DLInv = np.linalg.inv(DL)
  
  MSOR = DLInv @ (((1.0 - omega) * D) - (omega * U))
  ySOR = DLInv @ (omega * b)

  x = np.ones((MShape[0], 1))
  b1 = M @ x

  iteration = 0

  while (np.linalg.norm(b1 - b) > eps and iteration < maxIterations):
    x = (MSOR @ x) + ySOR
    b1 = M @ x

    iteration += 1
  
  return x

def gradienteConjugado(M, b, eps):
  S = M.transpose() @ M
  bs = M.transpose() @ b

  x = np.ones((np.shape(S)[0], 1))
  fGradiente = (S @ x) - bs
  d = -fGradiente

  while(np.linalg.norm(fGradiente) > eps):
    t = -(fGradiente.transpose() @ d) / (d.transpose() @ S @ d)
    
    x = x + (t * d)
    fGradiente = (S @ x) - bs

    beta = (d.transpose() @ S @ fGradiente) / (d.transpose() @ S @ d)
    d = -fGradiente + (beta * d)
  
  return x

def main ():
  C = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]])
  #C = np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
  #b = np.array([[0.0],[1.0],[0.0]])
  #b = np.array([[1.0],[2.0],[3.0]])

  A = np.array([[4.0, -1.0, -6.0, 0.0], [-5.0, -4.0, 10.0, 8.0], [0.0, 9.0, 4.0, -2.0], [1.0, 0.0, -7.0, 5.0]])
  b = np.array([[2.0], [21.0], [-12.0], [-6.0]])
  #A = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0],[7.0, 8.0, 9.0]])
  #A = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
  #b = np.array([[1.0], [2.0], [3.0]])
  

  #x = SOR(C, 1.1, b, 0.00001)
  #x = SOR(A, 0.5, b, 0.00001, 100)

  x = gradienteConjugado(A, b, 0.00001)

  print("b:\n{}\n\nx:\n{}\n\nM * x\n{}\n".format(b, x, A @ x))

if __name__ == "__main__":
  main()