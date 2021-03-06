import numpy as np

## Arquivo com a função de Gauss Jordan
## main() apenas para testes

def gauss(M, eps):
  copy = M.copy()
  size = np.shape(copy)
  minSize = min(size[0], size[1])
  MInv = np.zeros(size)

  for i in range(minSize):
    MInv[i, i] = 1
  
  for i in range(minSize):
    if (abs(copy[i, i]) <= eps):
      maxLine = min(i + 1, size[0] - 1)

      for k in range(i + 1, size[0]):
        if (abs(copy[k, i]) > eps and abs(copy[k, i]) > abs(M[maxLine, i])):
          maxLine = k

      aux = copy[i, :]
      copy[i, :] = copy[maxLine, :]
      copy[maxLine, :] = aux

      aux = MInv[i, :]
      MInv[i, :] = MInv[maxLine, :]
      MInv[maxLine, :] = aux
    
    MInv[i, :] = MInv[i, :] / copy[i, i]
    copy[i, :] = copy[i, :] / copy[i, i]

    for k in range(i + 1, size[0]):
      MInv[k, :] = MInv[k, :] - (MInv[i, :] * copy[k, i])
      copy[k, :] = copy[k, :] - (copy[i, :] * copy[k, i])
  
  return MInv

def resolutionGauss (M, b):
  x = np.zeros(np.shape(b))
  size = np.shape(b)[0]

  for i in range(size - 1, -1, -1):
    x[i, 0] = b[i, 0]
    for j in range(size - 1, i, -1):
      x[i, 0] = x[i, 0] - (M[i, j] * x[j, 0])
      
    x[i, 0] = x[i, 0] / M[i, i]
  
  return x

def gaussJordan(M, eps):
  copy = M.copy()
  size = np.shape(copy)
  minSize = min(size[0], size[1])
  MInv = np.zeros(size)

  for i in range(minSize):
    MInv[i, i] = 1
  
  for i in range(minSize):
    if (abs(copy[i, i]) <= eps):
      maxLine = min(i + 1, size[0] - 1)

      for k in range(i + 1, size[0]):
        if (abs(copy[k, i]) > eps and abs(copy[k, i]) > abs(M[maxLine, i])):
          maxLine = k

      aux = copy[i, :]
      copy[i, :] = copy[maxLine, :]
      copy[maxLine, :] = aux

      aux = MInv[i, :]
      MInv[i, :] = MInv[maxLine, :]
      MInv[maxLine, :] = aux
    
    if (abs(copy[i, i]) > eps):
      MInv[i, :] = MInv[i, :] / copy[i, i]
      copy[i, :] = copy[i, :] / copy[i, i]

    for k in range(i + 1, size[0]):
      MInv[k, :] = MInv[k, :] - (MInv[i, :] * copy[k, i])
      copy[k, :] = copy[k, :] - (copy[i, :] * copy[k, i])
  
  for i in range(minSize - 1, -1, -1):
    for k in range(i - 1, -1, -1):
      MInv[k, :] = MInv[k, :] - (MInv[i, :] * copy[k, i])
      copy[k, :] = copy[k, :] - (copy[i, :] * copy[k, i])
  
  print("teste")
  return MInv

def main ():
  M = np.array([[1, 2, 3], [4, 5, 6], [7, 7, 8]])
  MInv = gaussJordan(M, 0.00001)
  print("Inersa:\n{}\n\nM * M-1:\n{}".format(MInv, M @ MInv))

if __name__ == '__main__':
  main()