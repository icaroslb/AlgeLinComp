import metodosPotencia as mp
import numpy as np

## Arquivo resolvendo a atividade 8

def main ():
  sizeMatrix = int(input("Insira o tamanho da matriz: "))
  print("\nInsira a matriz por linha: ")

  M = np.zeros([sizeMatrix, sizeMatrix])
  x = np.random.rand(sizeMatrix, 1)

  for i in range(sizeMatrix):
    M[i, :] = [float(n) for n in input("Insira a linha {}: ".format(i)).split(" ")]
  print("{}\n\n{}\n".format(M, x))

  eps = abs(float(input("Insira o erro: ")))

  print("\nOpções:\n1 - Método da potência\n2 - Método da potência inversa\n3 - Método da potência com deslocamento\n")
  option = int(input("Insira a opção: "))

  if (option == 1):
    (autoValue, autoVector) = mp.metPotencia(M, x, eps)
  elif (option == 2):
    (autoValue, autoVector) = mp.metPotenciaInv(M, x, eps)
  elif(option == 3):
    desloc = float(input("Insira o deslocamento: "))
    (autoValue, autoVector) = mp.metPotenciaDesloc(M, x, desloc, eps)
  
  print("\nAutovalor: {}\nAutovetor:\n{}".format(autoValue, autoVector))

if __name__ == "__main__":
  main()