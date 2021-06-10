import numpy as np

class S_Dbw():
    def __init__(self,data,data_cluster,cluster_centroids_):
        """
        data --> raw data
        data_cluster --> The category that represents each piece of data(the number of category should begin 0)
        cluster_centroids_ --> the center_id of each cluster's center
        """
        self.data = data
        self.data_cluster = data_cluster
        self.cluster_centroids_ = cluster_centroids_

        # cluster_centroids_ жЇдёЂдёЄ array, з»™е‡єз±»ж•°k
        self.k = cluster_centroids_.shape[0]
        self.stdev = 0                # stdev зљ„е€ќе§‹еЊ–
        # еЇ№жЇЏдёЄз±»е€«ж ‡и®°иї›иЎЊеѕЄзЋЇпјЊе¦‚пјљ0пјЊ1пјЊ2пјЊ...
        # иЇҐеѕЄзЋЇи®Ўз®—зљ„жЇдё‹йќўе…¬ејЏй‡Њж №еЏ·й‡Њзљ„е†…е®№пјљ
        for i in range(self.k):
            # и®Ўз®—жџђз±»е€«дё‹ж‰Ђжњ‰ж ·жњ¬еђ„и‡Єзљ„е…ЁйѓЁз‰№еѕЃеЂјзљ„ж–№е·®пјљ
            #пј€vectorпјЊshapeдёєж ·жњ¬зљ„дёЄж•°пјЊз›ёеЅ“дєЋдё‹йќўе…¬ејЏй‡Њзљ„ signmaпј‰
            std_matrix_i = np.std(data[self.data_cluster == i],axis=0)
            # ж±‚е’Њ
            self.stdev += np.sqrt(np.dot(std_matrix_i.T,std_matrix_i))
        self.stdev = np.sqrt(self.stdev)/self.k # еЏ–е№іеќ‡


    def density(self,density_list=[]):
        """
        compute the density of one or two cluster(depend on density_list)
        еЏй‡Џ density_list е°†дЅњдёєж­¤е‡Ѕж•°зљ„е†…йѓЁе€—иЎЁпјЊе…¶еЏ–еЂјиЊѓе›ґжЇ0,1,2,... пјЊе…ѓзґ дёЄж•°жЇиЃљз±»з±»е€«ж•°з›®
        """
        density = 0
        if len(density_list) == 2:    # еЅ“иЂѓи™‘дё¤дёЄиЃљз±»з±»е€«ж—¶еЂ™пјЊз»™е‡єдё­еїѓз‚№дЅЌзЅ®
            center_v = (self.cluster_centroids_[density_list[0]] +self.cluster_centroids_[density_list[1]])/2
        else:                         # еЅ“еЏЄиЂѓи™‘жџђдёЂдёЄиЃљз±»з±»е€«зљ„ж—¶еЂ™пјЊз»™е‡єдё­еїѓз‚№дЅЌзЅ®
            center_v = self.cluster_centroids_[density_list[0]]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:    # np.linalg.norm жЇж±‚иЊѓж•°(order=2)
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density


    def Dens_bw(self):
        density_list = []
        result = 0
        # дё‹йќўзљ„еЏй‡Џ density_list е€—иЎЁе°†дјљз®—е‡єжЇЏдёЄеЇ№еє”еЌ•з±»зљ„еЇ†еє¦еЂјгЂ‚
        for i in range(self.k):
            density_list.append(self.density(density_list=[i])) # i жЇеѕЄзЋЇз±»е€«ж ‡з­ѕ
        # ејЂе§‹еѕЄзЋЇжЋ’е€—
        for i in range(self.k):
            for j in range(self.k):
                if i==j:
                    continue
                result += self.density([i,j])/max(density_list[i],density_list[j])
        return result/(self.k*(self.k-1))

    def Scat(self):
        # е€†жЇЌйѓЁе€†пјљ
        sigma_s = np.std(self.data,axis=0)
        sigma_s_2norm = np.sqrt(np.dot(sigma_s.T,sigma_s))

        # е€†е­ђйѓЁе€†пјљ
        sum_sigma_2norm = 0
        for i in range(self.k):
            matrix_data_i = self.data[self.data_cluster == i]
            sigma_i = np.std(matrix_data_i,axis=0)
            sum_sigma_2norm += np.sqrt(np.dot(sigma_i.T,sigma_i))
        return sum_sigma_2norm/(sigma_s_2norm*self.k)


    def S_Dbw_result(self):
        """
        compute the final result
        """
        return self.Dens_bw()+self.Scat()

#just for tests
#data = np.array([[1,2,1],[0,1,4],[3,3,3],[2,2,2]])
#data_cluster = np.array([1,0,1,2]) # The category represents each piece of data belongs
#centers_id = np.array([1,0,3]) # the cluster's num is 3

#a = S_Dbw(data,data_cluster,centers_id)
#print(a.S_Dbw_result())


# дѕ‹е­ђ
import S_Dbw as sdbw
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin

#import matplotlib.pyplot as plt

np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(X)

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

KS = sdbw.S_Dbw(X, k_means_labels, k_means_cluster_centers)
print(KS.S_Dbw_result())
