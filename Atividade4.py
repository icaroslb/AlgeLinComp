import decomposicao
import numpy as np

## Arquivo resolvendo a atividade 3

def main ():
  sizeMatrix = int(input("Insira o tamanho da matriz simétrica: "))
  print("\nInsira a matriz simétrica por linha: ")

  M = np.zeros([sizeMatrix, sizeMatrix])

  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]

  print("M:\n{}\n".format(M))
  
  try:
    S = decomposicao.cholesky(M)

    print("\nS: {}\n\nSt:\n{}".format(S, np.transpose(S)))
  except ValueError as error:
    print(error)

if __name__ == "__main__":
  main()