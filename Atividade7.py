import numpy as np
import regressao as regre

def main():
  [lines, columns] = [int(value) for value in input("Insira a quantidade de linhas e colunas separadas por espaço: ").split(" ")]
  
  M = np.zeros([lines, columns])
  b = np.zeros([lines, 1])

  for i in range(lines):
    M[i, :] = [float(value) for value in input("Insira os valores das linhas {} separados por espaço: ".format(i + 1)).split(" ")]
  
  b[:, 0] = [float(value) for value in input("Insira os valores do vetor b separados por espaço: ").split(" ")]
  
  x = regre.minQuadrado(M, b)

  print("\nO x com valor mais próximo é:\n{}".format(x))

if __name__ == "__main__":
  main()