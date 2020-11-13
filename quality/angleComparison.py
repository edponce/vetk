import w2v_helper as w2vh

import sys
args = sys.argv

if len(args) < 5:
  print("Usage: python",args[0],"<embedding1.vec> <embedding2.vec> <angles.txt> <results.log>")
  exit()


def loadAngles(fname):
  angles = list()
  angleFile = open(fname,"r")
  # check file return!
  for line in angleFile:
    words = line.strip().split()
    if len(words) != 3:
      print("Skipping unusual line:",line)
    else:
      angles.append([ words[0], words[1], words[2] ])
  angleFile.close()
  return angles
      
embedding1 = w2vh.readEmbeddingFile(args[1])
embedding2 = w2vh.readEmbeddingFile(args[2])

vocab = set(embedding1.keys()) and set( embedding2.keys())

testAngles = loadAngles(args[3])

sumDiff = 0
numAngles = len(testAngles)
outfile = open(args[4],"w")
for angle in testAngles:
  if angle[0] not in vocab or angle[1] not in vocab or angle[2] not in vocab:
    print("Angle not in vocabulary:",angle[0],angle[1],angle[2])
    numAngles = numAngles - 1
    continue
  points1 = [ embedding1[angle[0]], embedding1[angle[1]], embedding1[angle[2]] ]
  angle1 = w2vh.threePointAngle(points1[0], points1[1], points1[2])
  points2 = [ embedding2[angle[0]], embedding2[angle[1]], embedding2[angle[2]] ]
  angle2 = w2vh.threePointAngle(points2[0], points2[1], points2[2])
  diff = (angle1 - angle2) * (angle1 - angle2)
  outfile.write(angle[0] + " " + angle[1] + " " + angle[2] + " " + str(diff) + "\n")
  sumDiff += diff

meanSquaredError = sumDiff / float(numAngles)
outfile.write("\n")
outfile.write("Mean Squared Error: " + str(meanSquaredError) + "\n")
print("Mean squared error:", meanSquaredError)
outfile.close()
