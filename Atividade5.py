import numpy as np
import decomposicao

def main():
  [lines, columns] = [int(value) for value in input("Insira a quantidade de linhas e colunas separadas por espaço: ").split(" ")]
  eps = float(input("Insira o valor do erro: "))

  M = np.zeros([lines, columns])

  for i in range(lines):
    M[i, :] = [float(value) for value in input("Insira os valores das linhas {} separados por espaço: ".format(i + 1)).split(" ")]
  
  [N, rank, nulo, columnPivots] = decomposicao.RREF(M, eps)

  print("\nMatriz entrada:\n{}\n\nMatriz RREF:\n{}\n\nRank: {}\n\nDimenção nula: {}".format(M, N, rank, nulo))

if __name__ == "__main__":
  main()