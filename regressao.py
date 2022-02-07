import numpy as np
import gauss

def minQuadrado(M, b):
  qtdLines = np.shape(M)[0]
  qtdColumns = np.shape(M)[1]
  A = np.concatenate((M, np.ones((qtdLines, 1))), axis=1)
  
  At = np.transpose(A)
  Aq = At @ A
  v = At @ b

  AqI = gauss.gaussJordan(Aq, 0.00001)
  x = AqI @ v
  xReturn = x[0:qtdColumns]
  
  return xReturn

def main():
  M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 5.0, 4.0], [1.0, 5.0, 1.0], [2.0, 7.0, 8.0]])
  b = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
  
  x = minQuadrado(M, b)

  print("x:\n{}\n\nM * x:\n{}".format(x, M @ x))

if __name__ == "__main__":
  main()