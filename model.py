import numpy as np

class Model:
    @staticmethod
    def euclideanDistance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # @staticmethod
    # def seedGen(data, k):
    #     seed = np.random.choice(range(len(data)), k, replace=False)
    #     return seed

    @staticmethod
    def initializeCentroids(data, k, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        seed = np.random.choice(range(len(data)), k, replace=False)
        centroids = data[seed]
        return centroids

    @staticmethod
    def assignClusters(data, centroids):
        clusters = []
        for x in data:
            distances = [Model.euclideanDistance(x, centroid) for centroid in centroids]
            clusterLabel = np.argmin(distances)
            clusters.append(clusterLabel)
        return np.array(clusters)

    @staticmethod
    def updateCentroids(data, clusters, k):
        centroids = np.zeros((k, data.shape[1]))
        for clusterLabel in range(k):
            clusterMean = np.mean(data[clusters == clusterLabel], axis=0)
            centroids[clusterLabel] = clusterMean
        return centroids

    @staticmethod
    def computeDBI(data, centroids, clusters):
        cluster_distances = []
        for i in range(len(centroids)):
            distances = []
            for j in range(len(centroids)):
                if i != j:
                    cluster_i = data[clusters == i]
                    cluster_j = data[clusters == j]
                    centroid_i = centroids[i]
                    centroid_j = centroids[j]
                    SSWi = np.mean([Model.euclideanDistance(point, centroid_i) for point in cluster_i])
                    SSWj = np.mean([Model.euclideanDistance(point, centroid_j) for point in cluster_j])
                    SSB = Model.euclideanDistance(centroid_i, centroid_j)
                    distances.append((SSWi + SSWj) / SSB)
            cluster_distances.append(max(distances))
        return np.array(cluster_distances)

    @staticmethod
    def kmeans(data, k_range):
        dbiVal = []
        for k in k_range:
            centroids = Model.initializeCentroids(data, k)
            for _ in range(300):  # jumlah iterasi
                clusters = Model.assignClusters(data, centroids)
                new_centroids = Model.updateCentroids(data, clusters, k)
                if np.all(centroids == new_centroids):
                    break
                centroids = new_centroids
            dbi = np.mean(Model.computeDBI(data, centroids, clusters))
            dbiVal.append(dbi)
        return dbiVal
