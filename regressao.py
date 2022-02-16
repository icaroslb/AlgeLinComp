import numpy as np
import gauss

def minQuadrado(M, b):
  x = np.linalg.solve(M.transpose() @ M, M.transpose() @ b)
  
  return x

def main():
  M = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 5.0, 4.0], [1.0, 5.0, 1.0], [2.0, 7.0, 8.0]])
  b = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
  
  x = minQuadrado(M, b)

  print("x:\n{}\n\nM * x:\n{}".format(x, M @ x))

if __name__ == "__main__":
  main()