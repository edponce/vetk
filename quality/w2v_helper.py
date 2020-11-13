# This script contains a bunch of useful functions for a variety of tasks
#   related to processing word vectors.

import numpy as np

# Read a word embedding or document embedding file and return a dictionary.
# Note that if you run this on a doc embedding, you'll get <ID> <vec> whereas
#  on a word embedding file you'll get <word> <vec>. Same code works either way.
# Currently this function has no error checking.
#  - Should add checking on file I/O.
#  - Should add checking that line contains at least two elements when split.
#  - Don't really need to check if the key makes sense, if not it'll just never get used.
def readEmbeddingFile(filename):
  embedFile = open(filename, "r")
  embeddings = dict()
  # read embeddings into dictionary:
  for line in embedFile:
    # Remove trailing newline, break line into exactly 2 pieces on whitespace.
    splitline = line.strip().split(None,1)
    key = splitline[0]
    value = splitline[1]
    # Convert vector string into a numpy vector, then save it in the dictionary
    embeddings[key] = np.fromstring(value,sep=' ')
  embedFile.close()
  return embeddings

# Take in a file, such as a vocab list, and create a list from it.
def readListFromFile(filename):
  listfile = open(filename,"r")
  listresult = list()
  for line in listfile:
    splitline = line.strip().split()
    for chunk in splitline:
      listresult.append(chunk)
  return listresult

# Load a set of documents from a file, one per line.  
def loadDocsFromFile(docfilename):
  docfile = open(docfilename,"r")
  docs = list()
  for line in docfile:
    docs.append(line.strip())
  return docs

# Load analogies from a file.
def loadAnalogiesFromFile(filename):
  analogyFile = open(filename,"r")
  analogyList = list()
  for line in analogyFile:
    analogy = line.strip().split()
    if len(analogy) != 4:
      print("Unexpected analogy length, skipping.",line)
    else:
      analogyList.append(analogy)
  return analogyList

# Take two numpy vectors of the same length and return their cosine similarity.
def cosineSim(vec1,vec2):
  if len(vec1) != len(vec2):
    print("Cannot compute cosine similarity between vectors of mismatched length.")
    return -2
  else:
    dotprod = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
      print("Cosine similarity: Cannot compare vectors with 0 length.")
      return 0
    cossim = dotprod / (norm1 * norm2)
    return cossim

# Determine angle defined by three points.
def threePointAngle(vec1,vec2,vec3):
  # TODO : Check compatability.
  return np.arccos(cosineSim(vec1 - vec2, vec3 - vec2))

# Create document vector from text string, given a word embedding (dict)
def documentVectorFromString(embedding, doc, dimensionality = -1):
  if dimensionality == -1:
    # There has to be a better way to initialize the document vector.
    someItem = list(embedding.values())[0] # grab some value from the dict
    dimensionality = len(someItem) # grab the length of the value
  docvec = np.zeros(dimensionality) # perhaps should use zeros_like instead?
  # iterate over the document:
  words = doc.strip().split()
  numwords = 0
  skippedWords = list()
  for word in words:
    if word in embedding:
      docvec = docvec + embedding[word]
      numwords = numwords + 1
    else:
      # word not in embedding; make user aware, then move on
      skippedWords.append(word)
  docvec = docvec / float(numwords)
  if len(skippedWords) > 0:
    print("Skipped words:",skippedWords)
  return docvec

# Get a list of all words closer than the target word to the target vector.
# Will always have at least one entry, the word itself.
def getCloserWords(embedding, targetVector, targetWord):
  wordVector = embedding[targetWord]
  searchRadius = cosineSim(wordVector, targetVector)
  pointCloud = list()
  for key, value in embedding.items():
    csim = cosineSim(value, targetVector)
    if csim >= searchRadius:
      pointCloud.append(key)
  return pointCloud

# Get a list of the closest x words, defaulting to 1.
# embedding should be a dictionary, and targetVector a numpy vector
def getClosestWords(embedding, targetVector, num=1):
  # TODO: Add some real input error checking

  # Creating a proper data structure that maintains sort is probably better if we do a larger operation, but if n is ~10 then n*log(n) insert/delete is totally fine.
  candidates = list()
  minSim = 0
  for key, value in embedding.items():
    csim = cosineSim(value, targetVector)
    if len(candidates) < num or csim > minSim:
      candidates.append([key, csim])
    if len(candidates) > num:
      candidates = sorted(candidates, key=lambda pair : (pair[1],pair[0]))
      candidates.pop(0) # remove element with lowest similarity score
      minSim = candidates[0][1] # grab similarity score from lowest entry
  # on the off chance the sort was never called, do it one last time
  candidates = sorted(candidates, key=lambda pair : (pair[1],pair[0]))
  # Return the words, not the scores
  rval = list()
  for element in candidates:
    rval.append(element[0])
  return rval

# Alternate handle that returns just the word rather than a list containing it.
def getClosestWord(embedding, targetVector):
  return getClosestWords(embedding, targetVector, 1)[0]

def getCosineSimToClosestToTarget(embedding, targetVector, targetWord):
  closestWord = getClosestWord(embedding, targetVector)
  return cosineSim(embedding[closestWord], embedding[targetWord])

# Traditional analogy test; is the target word the closest to the target vector?
def analogyIsClosest(embedding, analogy):
  # First, double check if the words are in our vocabulary.
  for word in analogy:
    if word not in embedding:
      print("Skipping analogy, word missing from vocabulary.",word)
      return -1
  # Next, create target vector.
  # A is to B as C is to ___
  # D = C - A + B
  vec_A = embedding[analogy[0]]
  vec_B = embedding[analogy[1]]
  vec_C = embedding[analogy[2]]
  target_vector = vec_C - vec_A + vec_B
  closestWord = getClosestWord(embedding, target_vector)
  word = analogy[3]
  if word == closestWord:
    return 1
  else:
    return 0

# Modified analogy test, top X
# is the target word within the closest X to the target vector?
def analogyIsClosestX(embedding, analogy, X):
  # First, double check if the words are in our vocabulary.
  for word in analogy:
    if word not in embedding:
      print("Skipping analogy, word missing from vocabulary.",word)
      return -1
  # Next, create target vector.
  # A is to B as C is to ___
  # D = C - A + B
  vec_A = embedding[analogy[0]]
  vec_B = embedding[analogy[1]]
  vec_C = embedding[analogy[2]]
  target_vector = vec_C - vec_A + vec_B
  closestWords = getClosestWords(embedding, target_vector, X)
  word = analogy[3]
  if word in closestWords:
    return 1
  else:
    return 0

# New test; analogy cosine test
# What's the cosine similarity between the target word and the closest to the
#   target vector?
def analogyCosineSim(embedding, analogy):
  # First, double check if the words are in our vocabulary.
  for word in analogy:
    if word not in embedding:
      print("Skipping analogy, word missing from vocabulary.",word)
      return -1
  # Next, create target vector.
  # A is to B as C is to ___
  # D = C - A + B
  vec_A = embedding[analogy[0]]
  vec_B = embedding[analogy[1]]
  vec_C = embedding[analogy[2]]
  target_vector = vec_C - vec_A + vec_B
  closestWord = getClosestWord(embedding, target_vector)
  word = analogy[3]
  if word == closestWord:
    return 1 # Don't bother taking cosine similarity between a word and itself.
  else:
    return cosineSim(embedding[word], embedding[closestWord])

# New test: cosine distance * points in radius
def analogyPointCloud(embedding, analogy):
  # First, double check if the words are in our vocabulary.
  for word in analogy:
    if word not in embedding:
      print("Skipping analogy, word missing from vocabulary.",word)
      return -1
  # Next, create target vector.
  # A is to B as C is to ___
  # D = C - A + B
  vec_A = embedding[analogy[0]]
  vec_B = embedding[analogy[1]]
  vec_C = embedding[analogy[2]]
  target_vector = vec_C - vec_A + vec_B
  word = analogy[3]
  closerWords = getCloserWords(embedding, target_vector, word)
  csim = cosimeSim(embedding[word], target_vector)
  cdist = 1 - csim
  return 1 - cdist * len(closerWords)
