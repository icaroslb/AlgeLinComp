import numpy as np
import similaridade as simi

def main ():
  sizeMatrix = int(input("Insira o tamanho da matriz: "))

  M = np.zeros([sizeMatrix, sizeMatrix])

  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]
  print("\nMatriz inicial:\n{}\n".format(M))

  [D, H] = simi.Householder(M)

  print("\n----------- Matrizes finais -----------\n")
  print("Matriz final:\n{}\n\nMatriz Householder final:\n{}\n".format(D, H))

if __name__ == "__main__":
  main()