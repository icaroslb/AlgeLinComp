import numpy as np

import similaridade as simi

def main ():
  eps = abs(float(input("Insira o erro: ")))
  sizeMatrix = int(input("Insira o tamanho da matriz: "))

  M = np.zeros([sizeMatrix, sizeMatrix])
  
  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]
  print("\nMatriz inicial:\n{}\n".format(M))

  [A, H] = simi.Householder(M)
  [eigenValues, eigenVectors] = simi.QR(M, H, eps)

  print("\nMatriz de entrada:\n{}\n\n\nAuto valores: {}\n\nAuto vetores:\n{}\n".format(M, eigenValues, eigenVectors))

if __name__ == "__main__":
  main()