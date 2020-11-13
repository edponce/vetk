#!/opt/anaconda3/bin/python3

import vetkit as vk

filt = 100

v1 = vk.WordEmbedding(vectors='checkpoint_runs/text8-1threads.vec', vocabulary='checkpoint_runs/text8-1threads.voc', label='1threads', filter=filt)
v1.process_attributes(2)

v12 = vk.WordEmbedding(vectors='checkpoint_runs/text8-12threads.vec', vocabulary='checkpoint_runs/text8-12threads.voc', label='12threads', filter=filt)
v12.process_attributes(2)

v12r = vk.WordEmbedding(vectors='checkpoint_runs/text8-12threads-r.vec', vocabulary='checkpoint_runs/text8-12threads-r.voc', label='12rthreads', filter=filt)
v12r.process_attributes(2)


a = vk.WordEmbeddingAnalysis([v1, v12, v12r], prefix='w2v-checkpoint')

a.histogram(['angle pairs', 'distances', 'point distances'])
a.reduction_clustering(reduce='tsne', cluster='spectral', num_clusters=4)
a.reduction_clustering(reduce='pca', cluster='kmeans', dim=2, num_clusters=8)
