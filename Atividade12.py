import numpy as np
import decomposicao as decom

def main ():
  eps = abs(float(input("Insira o erro: ")))
  [qtdLines, qtdColumns] = [int(value) for value in input("Insira a largura e a altura da matriz: ").split(" ")]

  M = np.zeros([qtdLines, qtdColumns])
  
  for i in range(qtdLines):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]

  [U, S, V] = decom.decomposicaoSVD(M, eps)

  print("\nU:\n{}\n\nS:\n{}\n\nV:\n{}\n\nU*S*Vt:\n{}\n".format(U, S, V, U @ S @ V.transpose()))

if __name__ == "__main__":
  main()