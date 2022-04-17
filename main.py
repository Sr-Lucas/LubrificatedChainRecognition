# pybrain
from pybrain.datasets import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# images
from PIL import Image
import glob

# graph 
import matplotlib.pyplot as plt

outputFile = open("output.txt", "w")

def getImageData(path):
  image = Image.open(path)
  
  imageWidth = image.size[0]
  imageHeight = image.size[1]

  pixels = image.load()

  data = []

  for i in range(imageWidth):
    for j in range(imageHeight):
      pixel = []
      pixel = pixels[i, j]

      data.append(pixel[0])
      data.append(pixel[1])
      data.append(pixel[2])

  exif_data = image._getexif()
  exif_data

  return data


def loadImageDataFromDir(dir):
  return glob.glob(f"{dir}/*.jpg")


def test(paths, network, target):
  rightAswns = 0

  for path in paths:
    print(path)
    result = network.activate(getImageData(path))
    print(result)

    for r in result:
      outputFile.write(str(f"{r}, "))
    outputFile.write(f" - {path}\n")

    count = 0
    for res in result:
      if(res >= 0.5):
        count += 1
    
    if(count > 2):
      print("LUBRIFICADA \n\n")
      if(target): rightAswns += 1

    else:
      print("NAO LUBRIFICADA \n\n")
      if(not target): rightAswns += 1
  
  return rightAswns


# network
inputSize = 150 * 150 * 3 # 67.500
outputSize = 4

network = buildNetwork(inputSize, 150, 150, outputSize)
dataSet = SupervisedDataSet(inputSize, outputSize)


# populate dataset
# lubrificadas
lubrificatedImages = loadImageDataFromDir("lubrificadas")

for lub in lubrificatedImages:
  dataSet.addSample(getImageData(lub), (1, 1, 1, 1))

# nlubrificadas
notLubrificatedImages = loadImageDataFromDir("nlubrificadas")

for nlub in notLubrificatedImages:
  dataSet.addSample(getImageData(nlub), (0, 0, 0, 0))

trainer = BackpropTrainer(network, dataSet)


# training network
countIteration = 0
error = 1

outputs = []

while error > 0.001:
  error = trainer.train()
  outputs.append(error)
  countIteration += 1
  if(countIteration % 10 == 0):
    print(countIteration, error)

plt.ioff()
plt.plot(outputs)
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()


# testing
lubrificatedImgs = loadImageDataFromDir("testing\\lubrificadas")
notLubrificatedImgs = loadImageDataFromDir("testing\\nlubrificadas")

total = lubrificatedImgs.__len__() + notLubrificatedImgs.__len__()
rightAswns = 0

rightAswns += test(lubrificatedImgs, network, 1)
rightAswns += test(notLubrificatedImgs, network, 0)

rightPercentage = (rightAswns/total)*100
print("PERCENTUAL DE ACERTO: ", str(rightPercentage))

outputFile.write(f"\n PERCENTUAL DE ACERTO: {rightPercentage}%")