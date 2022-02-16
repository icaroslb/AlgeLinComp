from cmath import inf
import numpy as np
import gauss
import decomposicao

## Arquivo com as funções de métodos de potência, potência inversa e com deslocamento
## main() apenas para testes

def metPotencia (M, x, eps):
  error = float('inf')

  autoVector = x /np.linalg.norm(x)
  nextY = autoVector
  autoValue = 1
  nextAutoValue = inf

  while (error > eps):
    nextY = M @ autoVector
    nextAutoValue = np.linalg.norm(nextY)

    error = abs((nextAutoValue - autoValue) / autoValue)

    autoValue = nextAutoValue
    autoVector = nextY / np.linalg.norm(nextY)

  return (autoValue, autoVector)

def metPotenciaInv (M, x, eps):
  try:
    (L, U) = decomposicao.LU(M)
    error = float('inf')

    autoVector = x /np.linalg.norm(x)
    nextY = autoVector
    autoValue = 1
    nextAutoValue = 0

    while (error > eps):
      nextY = decomposicao.resolutionLU(L, U, autoVector)
      nextAutoValue = np.linalg.norm(nextY)

      error = abs((nextAutoValue - autoValue) / autoValue)

      autoValue = nextAutoValue
      autoVector = nextY / np.linalg.norm(nextY)
  except ValueError as error:
    MInv = gauss.gaussJordan(M, eps)
    (autoValue, autoVector) = metPotencia(MInv, x, eps)

  return (1 / autoValue, autoVector)

def metPotenciaDesloc (M, x, desloc, eps):
  MDesloc = M - (np.identity(np.shape(M)[0]) * desloc)
  (autoValue, autoVector) = metPotenciaInv(MDesloc, x, eps)
  return (autoValue + desloc, autoVector)

def main() -> None:
  M = np.array([[1, 2, 3], [4, 5, 6], [7, 7, 8]])
  x = np.array([[1], [2], [3]])

  (maxAutoValue, maxAutoVector) = metPotencia(M, x, 0.00001)
  (minAutoValue, minAutoVector) = metPotenciaInv(M, x, 0.00001)
  (deslocAutoValue, deslocAutoVector) = metPotenciaDesloc(M, x, -2, 0.0000001)

  print("Potência:\n{}\n\n{}".format(maxAutoValue, maxAutoVector))
  print("\n\n")
  print("Potência inversa:\n{}\n\n{}".format(minAutoValue, minAutoVector))
  print("\n\n")
  print("Potência deslocada:\n{}\n\n{}".format(deslocAutoValue, deslocAutoVector))

if __name__ == "__main__":
  main()