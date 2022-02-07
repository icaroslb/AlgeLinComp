import numpy as np
import ortogonalizacao as orto

def main():
  qtdVectors = int(input("Insira a quantidade de vetores: "))
  sizeVectors = int(input("Insira o tamanho dos vetores: "))
  
  eps = float(input("Insira o valor do erro: "))

  M = np.zeros([sizeVectors, qtdVectors])

  for i in range(qtdVectors):
    M[:, i] = [float(value) for value in input("Insira os valores do {}º vetor separados por espaço: ".format(i + 1)).split(" ")]
  
  A = orto.gramSchmidt(M, eps)

  print("\nBase do R{} ortonormalizada:".format(sizeVectors))
  for i in range(sizeVectors):
    print("{}º - {}".format(i + 1, A[:, i]))

if __name__ == "__main__":
  main()