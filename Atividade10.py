import numpy as np
import decomposicao as decomp

def main ():
  sizeMatrix = int(input("Insira o tamanho da matriz: "))

  M = np.zeros([sizeMatrix, sizeMatrix])

  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]
  print("\nMatriz inicial:\n{}\n".format(M))

  [Q, R] = decomp.decomposicaoQR(M)

  print("\nMatriz de entrada:\n{}\n\n\nMatriz Q:\n{}\n\nMatriz R:\n{}\n\nMatriz Q * R:\n{}\n".format(M, Q, R, Q @ R))

if __name__ == "__main__":
  main()