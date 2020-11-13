from w2v_helper import *
import sys

args = sys.argv

# Usage: python <this> embedding.vec analogies.txt output.txt
if len(args) < 4:
  print("Usage: python",args[0],"<word_embedding.vec> <analogies.txt> <output.txt>")
  exit()

analogies = loadAnalogiesFromFile(args[2])
embedding = readEmbeddingFile(args[1])
outfile = open(args[3],"w")
numAnalogies = len(analogies) # We'll modify this if we run into any unusable.
sumScore = 0.0
for analogy in analogies:
  val = analogyIsClosest(embedding,analogy)
  if val == -1: # there was a missing word, skip it
    numAnalogies -= 1
    outfile.write("SKIPPING ANALOGY: " + ' '.join(analogy) + "\n")
  else:
    sumScore += val
    outfile.write(' '.join(analogy) + " : " + str(val) + "\n")
finalScore = float(sumScore)/float(numAnalogies)
print("Final score:",finalScore)
outfile.write("Final score: " + str(finalScore) + "\n")
outfile.close()
