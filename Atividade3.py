import decomposicao
import numpy as np

## Arquivo resolvendo a atividade 3

def main ():
  sizeMatrix = int(input("Insira o tamanho da matriz: "))
  print("\nInsira a matriz por linha: ")

  M = np.zeros([sizeMatrix, sizeMatrix])
  b = np.zeros([sizeMatrix, 1])

  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]

  print("\nInsira os valores do vetor b: ")
  for i in range(sizeMatrix):
    b[i, 0] = float(input("Insira o valor {}: ".format(i)))
  
  print("M:\n{}\n\nb:\n{}\n".format(M, b))
  
  try:
    (L, U) = decomposicao.LU(M)
    x = decomposicao.resolutionLU(L, U, b)

    print("\nL: {}\n\nU:\n{}\n\nx:\n{}".format(L, U, x))
  except ValueError as error:
    print(error)

if __name__ == "__main__":
  main()