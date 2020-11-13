from w2v_helper import *
import numpy as np
import sys
args = sys.argv

# Input two embeddings and a set of test documents
# Compute difference in their document similarity matrices

def computeDocumentSimilarity(embed1, embed2, doclist):
  numdocs = len(doclist)
  # I'm choosing to do this the more compute efficient and less mem efficient way, because we have plenty of memory (especially for reasonable test sets).
  docembed1 = list()
  docembed2 = list()
  # This will be faster if you precompute the dimensionality and pass it as an optional argument. ...I should probably separate that functionality.
  for doc in doclist:
    docembed1.append(documentVectorFromString(embed1,doc))
    docembed2.append(documentVectorFromString(embed2,doc))
  # Now that we've got all our document embeddings, go through all unique pairs
  frobeniusNorm = 0
  for i in range(numdocs):
    for j in range(i):
      embed1sim = cosineSim(docembed1[i], docembed1[j])
      embed2sim = cosineSim(docembed2[i], docembed2[j])
      frobeniusNorm += (embed1sim - embed2sim)*(embed1sim - embed2sim)
  frobeniusNorm = np.sqrt(frobeniusNorm)
  # I need to do some normalization, but this gives us a number
  return frobeniusNorm


docembed1 = readEmbeddingFile(args[1])
docembed2 = readEmbeddingFile(args[2])
doclist = loadDocsFromFile(args[3])
score = computeDocumentSimilarity(docembed1, docembed2, doclist)
print(score)
