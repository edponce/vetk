"""WordEmbeddingAnalysis

Classes:
    :class:`WordEmbeddingAnalysis`
"""
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://docs.scipy.org/doc/scipy/reference/stats.html
# https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
# https://www.datacamp.com/community/tutorials/apache-spark-python
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# http://scikit-learn.org/stable/modules/model_evaluation.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html


# Notes:
#There is no "k-means algorithm". There is MacQueens algorithm for k-means, the
#Lloyd/Forgy algorithm for k-means, the Hartigan-Wong method, ...
#
#There also isn't "the" EM-algorithm. It is a general scheme of repeatedly
#expecting the likelihoods and then maximizing the model. The most popular
#variant of EM is also known as "Gaussian Mixture Modeling" (GMM), where the
#model are multivariate Gaussian distributions.


import os
import sys
import math
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import sklearn
import sklearn.cluster
import sklearn.mixture
import scipy
from .clustering.mst import MSTClustering


__all__ = ['WordEmbeddingAnalysis']


class WordEmbeddingAnalysis:
    """
    Plot, analyze, and compare vector embeddings

    Notes:
        * embedding can be either single/list of WordEmbedding objects
        * embedding passed as parameters will be assigned as internal embedding

    TODO:
        * Markersize should be list of list instead of list of numpy arrays
    """
    FP_PRECISION = 6
    COL_WIDTH = 10


    def __init__(self, embedding=[], prefix='', plot_enable=True):
        self._file_prefix = ''
        self._embedding = []
        self.num_procs = 4
        self.plot_enable = plot_enable

        ###########
        # Setters #
        ###########

        self.file_prefix = prefix
        self.embedding = embedding

    def clear(self):
        self.file_prefix = ''
        self.embedding = []

    @property
    def file_prefix(self):
        return self._file_prefix

    @file_prefix.setter
    def file_prefix(self, prefix):
        '''
        Always end prefix with a dash/underscore because it is prepended to filenames
        '''
        if len(prefix) > 0 and prefix[-1] not in ['-', '_']:
            self._file_prefix = prefix + '-'
        else:
            self._file_prefix = prefix

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, embed):
        if embed:
            self._embedding = self._make_iterable(embed)

    def _calculate_subplots(self, n_plots):
        '''
        Calculate dimensions of subplots required for given number of plots
        '''
        n_rows = int(numpy.sqrt(n_plots))
        n_cols = math.ceil(numpy.sqrt(n_plots))
        if n_plots > (n_rows * n_cols):
            n_cols += 1
        return n_rows, n_cols

    def _extend_label(self, label, n_labels):
        '''
        Check if number of labels matches given size
        or
        extend single string to array of given size
        or
        set labels to empty

        Notes:
            * Replicate label if it is an iterable with length as a factor of size
        '''
        if isinstance(label, str):
            ext_label = n_labels * [label]
        elif isinstance(label, (list, tuple)):
            # Check if label length is multiple of repetitions
            if len(label) > 0 and (n_labels % len(label)) == 0:
                ext_label = (n_labels // len(label)) * label
            else:
                ext_label = n_labels * ['']
        else:
            ext_label = n_labels * ['']
        return ext_label

    def _make_iterable(self, data, levels=1):
        '''
        Detect single/iterable data with 1-2 levels of iterables

        Notes:
            * Strings are special case because they are always iterable in all levels,
              only supports 1-level for strings
        '''
        try:
            if isinstance(data, str):
                raise
            if levels == 1:
                dummy = len(data)
            elif levels == 2:
                dummy = len(data[0])
            iter_data = data
        except:
            iter_data = [data]
        return iter_data

    def _sphere_coordinates(self, center, radius):
        '''
        Calculate surface coordinates for a sphere given a circle's
        center and radius

        Notes:
            * center is a 3-D array

        Example:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z,  rstride=5, cstride=5, color='b', linewidth=0,
        alpha=0.5)
        '''
        u = numpy.linspace(0, 2 * numpy.pi, 100)
        v = numpy.linspace(0, numpy.pi, 100)
        x = center[0] + radius * numpy.outer(numpy.cos(u), numpy.sin(v))
        y = center[1] + radius * numpy.outer(numpy.sin(u), numpy.sin(v))
        z = center[2] + radius * numpy.outer(numpy.ones(u.size), numpy.cos(v))

        return x, y, z

    def _plot_wrapper(self, plot_type, data, vocab=[], xlabel=[], ylabel=[],
    zlabel=[], title=[], label=[], markersize=[], center=[], file=''):
        '''
        Manages subplots

        TODO:
            * In scatter plot add text, when mouse over?
        '''
        if not self.plot_enable:
            return

        data = self._make_iterable(data, levels=2)
        vocab = self._make_iterable(vocab, levels=2)

        # Cluster centroids
        center = self._make_iterable(center, levels=2)

        # Cluster/color labels
        label = self._make_iterable(label, levels=2)

        # Extend labels and titles
        xlabel = self._extend_label(xlabel, n_labels=len(data))
        ylabel = self._extend_label(ylabel, n_labels=len(data))
        zlabel = self._extend_label(zlabel, n_labels=len(data))
        title = self._extend_label(title, n_labels=len(data))
        plot_type = self._extend_label(plot_type, n_labels=len(data))

        # Create subplots
        n_rows, n_cols = self._calculate_subplots(len(data))
        fig = matplotlib.pyplot.figure()

        idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if idx < len(data):
                    if plot_type[idx] == 'histogram':
                        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                        bins = int(numpy.sqrt(len(data[idx]) / 2))  # bins = sqrt(n / 2)
                        ax.hist(data[idx], bins=bins)
                        ax.set_xlabel(xlabel[idx])
                        ax.set_ylabel(ylabel[idx])
                        ax.set_title(title[idx])
                    elif plot_type[idx] == 'boxplot':
                        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                        ax.boxplot(data[idx], sym='.')
                        ax.set_xlabel(xlabel[idx])
                        ax.set_ylabel(ylabel[idx])
                        ax.set_title(title[idx])
                        ax.set_xticklabels([])
                    elif plot_type[idx] == 'scatter':
                        # Marker size
                        if len(markersize) > idx:
                            markersz = markersize[idx]
                        else:
                            markersz = 10

                        # Color using labels
                        if len(label) > idx:
                            lcolor = label[idx]
                        else:
                            lcolor = 'b'

                        if data[idx].shape[1] == 2:
                            xdata = data[idx][:,0]
                            ydata = data[idx][:,1]

                            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                            ax.scatter(xdata, ydata, s=markersz, c=lcolor)

                            # Word labels
                            if len(vocab) > idx:
                                for word, xc, yc in zip(vocab[idx], xdata, ydata):
                                    matplotlib.pyplot.annotate(word, xy=(xc,yc),
                                    xytext=(0,0), textcoords='offset points',
                                    zorder=2)

                            # Cluster centers
                            if len(center) > idx and len(label) > idx:
                                radii = [scipy.spatial.distance.cdist(data[idx][label[idx] == i], [ctr]).max() for i, ctr in enumerate(center[idx])]
                                for ctr, rad in zip(center[idx], radii):
                                    ax.add_patch(matplotlib.pyplot.Circle(ctr,
                                    rad, facecolor='#CCCCCC', edgecolor='k',
                                    linewidth=1, alpha=0.3, zorder=0))

                        elif data[idx].shape[1] == 3:
                            xdata = data[idx][:,0]
                            ydata = data[idx][:,1]
                            zdata = data[idx][:,2]

                            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
                            ax.scatter(xdata, ydata, zdata, s=markersz, c=lcolor)

                            zlim_offs = (abs(zdata.min()) + abs(zdata.max())) / 10
                            ax.set_ylim(zdata.min() - zlim_offs, zdata.max() + zlim_offs)

                            # Word labels
                            if len(vocab) > idx:
                                for word, xc, yc, zc in zip(vocab[idx], xdata, ydata, zdata):
                                    ax.text(xc, yc, zc, '{}'.format(word), zorder=2)

                            ax.set_zlabel(zlabel[idx])

                            # Cluster centers
                            if len(center) > idx and len(label) > idx:
                                radii = [scipy.spatial.distance.cdist(data[idx][label[idx] == i], [ctr]).max() for i, ctr in enumerate(center[idx])]
                                for ctr, rad in zip(center[idx], radii):
                                    x3, y3, z3 = self._sphere_coordinates(ctr, rad)
                                    ax.plot_surface(x3, y3, z3, rstride=10,
                                    cstride=10, color='#CCCCCC',
                                    linewidth=0, alpha=0.1, zorder=0)

                        ax.set_xlabel(xlabel[idx])
                        ax.set_ylabel(ylabel[idx])
                        ax.set_title(title[idx])
                        xlim_offs = (abs(xdata.min()) + abs(xdata.max())) / 10
                        ylim_offs = (abs(ydata.min()) + abs(ydata.max())) / 10
                        ax.set_xlim(xdata.min() - xlim_offs, xdata.max() + xlim_offs)
                        ax.set_ylim(ydata.min() - ylim_offs, ydata.max() + ylim_offs)
                    elif plot_type[idx] == 'matrix':
                        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                        # scale data to [-1, 1]
                        cax = ax.matshow(data[idx]/data[idx].max(), interpolation='nearest')
                        ax.xaxis.tick_bottom()
                        ax.set_xlabel(xlabel[idx])
                        ax.set_ylabel(ylabel[idx])
                        ax.set_title(title[idx])
                        fig.colorbar(cax, fraction=0.045, ticks=[x * 0.1 for x in range(-10, 11, 2)])
                idx += 1

        # Show/print figure
        matplotlib.pyplot.tight_layout()
        if file:
            matplotlib.pyplot.savefig(self.file_prefix + file,
            bbox_inches='tight', dpi=300)
        else:
            matplotlib.pyplot.show()

    def _unique(self, data):
        '''
        Generate a unique list/tuple in same order as they appear in paramter
        '''
        if isinstance(data, list):
            uniq = []
            for x in data:
                if x not in uniq:
                    uniq.append(x)
        elif isinstance(data, tuple):
            uniq = ()
            for x in data:
                if x not in uniq:
                    uniq += (x,)
        else:
            raise ValueError('Invalid data type \'{}\', valid types are list/tuple'.format(type(data)))

        return uniq

    def _get_embedding_data(self, key='', dim=1):
        '''
        Extract data and titles from vector embedding object
        Data can be extract as 1-D/2-D formats

        Notes:
            * 2-D data populates only the lower triangular of the matrix

        TODO:
            * Fix arrangement of label and keys, this might require changing how
            _extend_label works
            * Arrangement is key-per-row, embedding-per-column
            * For arrangement, should we call _extend_label here
        '''
        data = []
        name = [e.label for e in self.embedding]
        keys = []
        if key:
            tkey = self._unique(self._make_iterable(key))
            name = len(tkey) * name

            # Duplicate/extend keys
            keys = [k for k in tkey for e in self.embedding]

            if dim == 1:
                data = [e.get(k) for k in tkey for e in self.embedding]
            elif dim == 2:
                for k in tkey:
                    for e in self.embedding:
                        # TODO: Detect if key refers to a pair/triple attribute (hack)
                        if len(e.get(k)) == e._flat_size(2, e._pairProcessingCount):
                            n = e._pairProcessingCount
                        elif len(e.get(k)) == e._flat_size(3, e._tripleProcessingCount):
                            n = e._tripleProcessingCount
                        matrix = numpy.zeros(shape=(n, n), dtype=numpy.float32)
                        matrix[numpy.tril_indices(n, -1)] = e.get(k)
                        data.append(matrix)

        return data, name, keys

    def histogram(self, key, embedding=[], file=''):
        '''Create histogram plot
        '''
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key)
        self._plot_wrapper(plot_type='histogram', data=data, xlabel=keys, ylabel='Frequency', title=name, file=file)

    def boxplot(self, key, embedding=[], file=''):
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key)
        self._plot_wrapper(plot_type='boxplot', data=data, ylabel=keys, title=name, file=file)

    def lineplot(self, key, embedding=[], file=''):
        '''
        TODO: not complete, what is its purpose?
        '''
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key)
        self._plot_wrapper(plot_type='lineplot', data=data, ylabel=keys, title=name, file=file)

    def matrixplot(self, key, embedding=[], file=''):
        '''Create a similarity matrix plot
        A square matrix with size as vocabulary is initialized to zeros, then
        its lower triangular part is populated with similarity values.
        '''
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key, dim=2)
        self._plot_wrapper(plot_type='matrix', data=data, xlabel=keys, title=name, file=file)

    def pca(self, dim=2, embedding=[], show_vocab=False, file=''):
        '''PCA (Principal Component Analysis)
        Linear dimensionality reduction using SVD
        PCA finds the k best linear combination in terms of variance
        '''
        self.embedding = embedding
        tdata, name, _ = self._get_embedding_data('vectors')

        model = []
        data = []
        for d in  tdata:
            m = sklearn.decomposition.PCA(n_components=dim)
            data.append(m.fit_transform(d))
            model.append(m)

        if self.plot_enable:
            # Plot up to 3-components
            k = dim if dim <= 3 else 3

            # Trim data based on components
            if data[0].shape[1] != k:
                tdata = [d[:,:k] for d in data]
            else:
                tdata = data

            # Vocabulary text, show 50 at most
            vocab = []
            if show_vocab:
                vocab = [list(e.vocabulary)[:min(50, len(e.vocabulary))] for e in self.embedding]

            # Marker size based on vocabulary occurrences
            # markersize = [(144/x.max())*x for e in self.embedding for x in
            # numpy.array(list(e.vocabulary.values()))]]
            markersize = []

            self._plot_wrapper(plot_type='scatter', data=tdata, vocab=vocab, xlabel='PC1', ylabel='PC2', zlabel='PC3', title=name, markersize=markersize, file=file)

        return data, model, name

    def tsne(self, dim=2, embedding=[], show_vocab=False, file=''):
        '''t-SNE (t-Distributed Stochastic Neighbor Embedding)
        Technique for dimensionality reduction, useful for visualizing highly-dimensional data
        t-SNE always produces a 2D separation, in contrast to PCA which can produce many different components
        t-SNE is a non-parametric learning algorithm. The embedding is learned by directly moving the data across the low dimensional space. That means one does not get an eigenvector to use in new data. In contrast, using PCA the eigenvectors offer a new basis for projecting new data.
        t-SNE is non-deterministic, does not guarantees to get exactly the same output each time it is run (though the results are likely to be similar)
        '''
        self.embedding = embedding
        tdata, name, _ = self._get_embedding_data('vectors')

        model = []
        data = []
        for d in tdata:
            m = sklearn.manifold.TSNE(n_components=dim, perplexity=50)
            data.append(m.fit_transform(d))
            model.append(m)

        if self.plot_enable:
            # Plot up to 3-components
            k = dim if dim <= 3 else 3

            # Trim data based on components
            if data[0].shape[1] != k:
                tdata = [d[:,:k] for d in data]
            else:
                tdata = data

            # Vocabulary text, show 50 at most
            vocab = []
            if show_vocab:
                vocab = [list(e.vocabulary)[:min(50, len(e.vocabulary))] for e in self.embedding]

            # Marker size based on vocabulary occurrences
            # markersize = [(144/x.max())*x for e in self.embedding for x in
            # numpy.array(list(e.vocabulary.values()))]]
            markersize = []

            self._plot_wrapper(plot_type='scatter', data=tdata, vocab=vocab, xlabel='Dim1', ylabel='Dim2', zlabel='Dim3', title=name, markersize=markersize, file=file)

        return data, model, name

    def anova(self, key, embedding=[], test='oneway'):
        '''ANOVA
        Perform 1-way ANOVA (mean) or Kruskal-Wallis H-test (median)
        '''
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key)

        nfmt = '{}: ' + ','.join(name)
        gfmt = '{:<' + str(self.COL_WIDTH) + '}'
        fpfmt = '{:<' + str(self.COL_WIDTH) + '.' + str(self.FP_PRECISION) + 'f}'
        sfmt = ' '.join([gfmt] + [fpfmt])

        stat = []
        for i, j in enumerate(range(0, len(data), len(self.embedding))):
            if test == 'oneway':
                s = scipy.stats.f_oneway(*data[j:j+len(self.embedding)])
            elif test == 'kruskal':
                s = scipy.stats.kruskal(*data[j:j+len(self.embedding)])
            stat.append(s)

            print(nfmt.format(keys[i]))
            print(sfmt.format('F-stat', s.statistic))
            print(sfmt.format('p-value', s.pvalue))

        return stat, keys, name

    def correlation(self, key, y, embedding=[]):
        '''
        Calculate correlations for mean and variance
        * y - iterable same size as embedding (number of threads/data size)

        Calculates a Pearson correlation coefficient and the p-value for testing
        non-correlation.

        The Pearson correlation coefficient measures the linear relationship
        between two datasets. Strictly speaking, Pearson's correlation requires
        that each dataset be normally distributed. Like other correlation
        coefficients, this one varies between -1 and +1 with 0 implying no
        correlation. Correlations of -1 or +1 imply an exact linear
        relationship. Positive correlations imply that as x increases, so does
        y. Negative correlations imply that as x increases, y decreases.

        The p-value roughly indicates the probability of an uncorrelated system
        producing datasets that have a Pearson correlation at least as extreme
        as the one computed from these datasets. The p-values are not entirely
        reliable but are probably reasonable for datasets larger than 500 or so.
        '''
        self.embedding = embedding
        data, name, keys = self._get_embedding_data(key)

        headers = ['mean', 'variance', 'skewness', 'kurtosis']

        # Output formatting
        nfmt = '{}: ' + ','.join(name)
        gfmt = '{:<' + str(self.COL_WIDTH) + '}'
        fpfmt = '{:<' + str(self.COL_WIDTH) + '.' + str(self.FP_PRECISION) + 'f}'
        hfmt = ' '.join([gfmt] + len(headers)*[gfmt])
        sfmt = ' '.join([gfmt] + len(headers)*[fpfmt])

        for i, j in enumerate(range(0, len(data), len(self.embedding))):

            # rows = embeddings, columns = statistics
            table = numpy.empty(shape=(len(self.embedding), len(headers)),
                                dtype=numpy.float32)
            for k in range(len(self.embedding)):
                stat = scipy.stats.describe(data[j+k])
                table[k,:] = [stat.mean, stat.variance, stat.skewness, stat.kurtosis]
            corr = [scipy.stats.pearsonr(table[:,k], y) for k in range(len(headers))]

            print(nfmt.format(keys[i]))
            print(hfmt.format('', *headers))  # blank space for aligning table
            print(sfmt.format('Coeff', *[c for c, _ in corr]))
            print(sfmt.format('p-value', *[c for _, c in corr]))
            print()

    def kmeans(self, data=[], num_clusters=3, embedding=[]):
        '''K-means clustering
        The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.KMeans(n_clusters=num_clusters, n_jobs=self.num_procs)
            labels.append(m.fit_predict(d))
            model.append(m)

        return labels, model, name

    def spectral(self, data=[], num_clusters=3, embedding=[]):
        '''Spectral clustering
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.SpectralClustering(n_clusters=num_clusters,
            eigen_solver='arpack', affinity='nearest_neighbors', n_jobs=self.num_procs)
            labels.append(m.fit_predict(d))
            model.append(m)

        return labels, model, name

    def birch(self, data=[], num_clusters=3, embedding=[]):
        '''Birch clustering
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.Birch(n_clusters=num_clusters)
            labels.append(m.fit_predict(d))
            model.append(m)

        return labels, model, name

    def mini_batch_kmeans(self, data=[], num_clusters=3, embedding=[]):
        '''Mini-batch K-means clustering
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters)
            labels.append(m.fit_predict(d))
            model.append(m)

        return labels, model, name

    def mean_shift(self, data=[], bin_seeding=True, cluster_all=True, embedding=[]):
        '''Mean-shift clustering
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.MeanShift(bin_seeding=bin_seeding,
            cluster_all=cluster_all, n_jobs=self.num_procs)
            labels.append(m.fit_predict(d))
            model.append(m)

        return labels, model, name

    def gmm(self, data=[], num_clusters=3, covar_type='diag', embedding=[]):
        '''Gaussian mixture models clustering (expectation maximization)
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.mixture.GaussianMixture(n_components=num_clusters, covariance_type=covar_type)
            m.fit(d)
            labels.append(m.predict(d))
            model.append(m)

        return labels, model, name

    def bayesian_gmm(self, data=[], num_clusters=3, covar_type='diag', embedding=[]):
        '''Bayesian Gaussian mixture models clustering (expectation maximization)
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.mixture.BayesianGaussianMixture(n_components=num_clusters, covariance_type=covar_type)
            m.fit(d)
            labels.append(m.predict(d))
            model.append(m)

        return labels, model, name

    def reduction_clustering(self, reduce, cluster, dim=2, num_clusters=3,
    cutoff=0.3, covar_type='diag', embedding=[], file=''):
        '''Dimensionality reduction and clustering
        Perform dimensionality reduction on embedding vectors followed by a
        clustering technique.

        Dimensionality reduction - 'reduce':
        * None - 'none'
        * PCA - 'pca'
        * t-SNE - 'tsne'

        Clustering - 'cluster'
        * None - 'none'
        * Spectral - 'spectral'
        - 'num_clusters'
        * K-means - 'kmeans'
        - 'num_clusters'
        * Mini-batch K-means - 'batchkmeans'
        - 'num_clusters'
        * Birch - 'birch'
        - 'num_clusters'
        * Mean shift - 'meanshift'
        - 'bin_seeding'
        - 'cluster_all'
        * Agglomerative - 'agglomerative'
        - 'num_clusters'
        * Gaussian mixture model (expectation maximization) - 'gmm'
        - 'num_clusters'
        - 'covar_type' - 'full', 'tied', 'diag', 'spherical'
        * Bayesian Gaussian mixture model (expectation maximization) - 'bgmm'
        - 'num_clusters'
        - 'covar_type' - 'full', 'tied', 'diag', 'spherical'
        * Minimum spanning tree - 'mst'
        - 'cutoff'
        * Affinity propagation - 'affinity'
        '''
        self.embedding = embedding

        # Disable plotting option for submethods
        plot_enable = self.plot_enable
        self.plot_enable = False

        # Dimensionality reduction
        if reduce == 'pca':
            data, rmodel, name = self.pca(dim=dim)
            xyzlabel = ['PC' + str(i) for i in range(1,4)]
        elif reduce == 'tsne':
            data, rmodel, name = self.tsne(dim=dim)
            xyzlabel = ['Dim' + str(i) for i in range(1,4)]
        elif reduce == 'none':
            data, name, _ = self._get_embedding_data('vectors')
            rmodel = len(self.embedding)*[None]
            xyzlabel = [str(i) for i in range(1,4)]

        # Clustering
        centers = []
        if cluster == 'kmeans':
            labels, cmodel, _ = self.kmeans(data=data, num_clusters=num_clusters)
            centers = [cm.cluster_centers_ for cm in cmodel]
        elif cluster == 'batchkmeans':
            labels, cmodel, _ = self.mini_batch_kmeans(data=data, num_clusters=num_clusters)
            centers = [cm.cluster_centers_ for cm in cmodel]
        elif cluster == 'birch':
            labels, cmodel, _ = self.birch(data=data, num_clusters=num_clusters)
            centers = len(cmodel)*[[]]
            #centers = [cm.subcluster_centers_ for cm in cmodel]
        elif cluster == 'meanshift':
            labels, cmodel, _ = self.mean_shift(data=data)
            centers = [cm.cluster_centers_ for cm in cmodel]
        elif cluster == 'gmm':
            labels, cmodel, _ = self.gmm(data=data, num_clusters=num_clusters,
            covar_type=covar_type)
            centers = [cm.means_ for cm in cmodel]
        elif cluster == 'bgmm':
            labels, cmodel, _ = self.bayesian_gmm(data=data, num_clusters=num_clusters,
            covar_type=covar_type)
            centers = [cm.means_ for cm in cmodel]
        elif cluster == 'mst':
            labels, cmodel, _ = self.min_spanning_tree(data=data, cutoff=cutoff)
            centers = len(cmodel)*[[]]
        elif cluster == 'spectral':
            labels, cmodel, _ = self.spectral(data=data,
            num_clusters=num_clusters)
            centers = len(cmodel)*[[]]
        elif cluster == 'agglomerative':
            labels, cmodel, _ = self.agglomerative(data=data, num_clusters=num_clusters)
            centers = len(cmodel)*[[]]
        elif cluster == 'affinity':
            labels, cmodel, _ = self.affinity_propagation(data=data)
            centers = [cm.cluster_centers_ for cm in cmodel]
        elif cluster == 'none':
            labels = len(self.embedding)*[numpy.empty(shape=0, dtype=self.embedding[0]._dtype)]
            cmodel = len(self.embedding)*[None]
            centers = len(self.embedding)*[[]]

        # Restore plotting option
        self.plot_enable = plot_enable

        # NOTE: do not plot circles because we can't deal with more than 3 dims
        if not reduce:
            centers = []

        if (reduce or cluster) and self.plot_enable:
            # Plot up to 3-components
            k = dim if dim <= 3 else 3

            # Trim data based on components
            # NOTE: Assumes all data has more columns than dimensions
            if data[0].shape[1] != k:
                tdata = [d[:,:k] for d in data]
            else:
                tdata = data

            # Vocabulary text, show 50 at most
            vocab = [list(e.vocabulary)[:min(50, len(e.vocabulary))] for e in self.embedding]

            # Marker size based on vocabulary occurrences
            # markersize = [(144*x.max())/x for e in self.embedding for x in
            # numpy.array(list(e.vocabulary.values()))]]
            markersize = []

            self._plot_wrapper(plot_type='scatter', data=tdata, vocab=vocab,
            xlabel=xyzlabel[0], ylabel=xyzlabel[1], zlabel=xyzlabel[2],
            title=name, markersize=markersize, label=labels, center=centers, file=file)

        return data, labels, name, rmodel, cmodel

    def affinity_propagation(self, data=[], embedding=[]):
        '''Affinity propagation clustering
        Clustering algorithm based on 'message passing'
        * Does not require the number of clusters to be determined.
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.AffinityPropagation()
            model.append(m.fit(d))
            labels.append(m.labels_)

        return labels, model, name

    def agglomerative(self, data=[], num_clusters=3, embedding=[]):
        '''Agglomerative clustering
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters)
            model.append(m.fit(d))
            labels.append(m.labels_)

        return labels, model, name

    def min_spanning_tree(self, data=[], cutoff=0.3, embedding=[]):
        '''Minimum spanning tree clustering
        Simple scikit-learn style estimator for clustering with a MST
        Based on a trimmed Euclidean MST

        Requires 'cutoff' parameter
        * Number/fraction of edges to cut
        * [0-1] - fraction
        * [2-:] - value
        'cutoff_scale' parameter
        * Minimum size of edges
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        model = []
        labels = []
        for d in data:
            m = MSTClustering(cutoff=cutoff)
            model.append(m.fit(d))
            labels.append(m.labels_)

        return labels, model, name

    def silhouette(self, labels, data=[], embedding=[]):
        '''Silhouette analysis
        Silhouette coefficients near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
        * The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters.
        * For each cluster number, calculate the average silhouette of observations (avg.sil)
        * Plot the curve of avg.sil according to the number of clusters
        * The location of the maximum is considered as the appropriate number of clusters

        Notes:
            * labels refers to clustering labels (e.g., K-means)
            * calculates either average or samples silhouette scores
        '''
        self.embedding = embedding
        if len(data) == 0:
            data, name, _ = self._get_embedding_data('vectors')
        else:
            data = self._make_iterable(data)
            name = []

        sample = []
        score = []
        for d, l in zip(data, labels):
            # Samples scores
            s = sklearn.metrics.silhouette_samples(d, l)
            sample.append(s)

            # Average scores
            #score.append(sklearn.metrics.silhouette_score(d, l))
            score.append(numpy.mean(s))

        return score, sample, name
