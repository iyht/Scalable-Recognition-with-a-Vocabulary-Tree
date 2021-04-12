import os
import cv2
import imghdr
import time
from feature import *
import pickle
from sklearn.cluster import KMeans

class Node:
    def __init__(self):
        self.value = None
        self.children = []
        self.img_count = {}

class Database:
    def __init__(self):
        self.data_path = ''
        # self.img_to_des_and_kpts = {}
        self.num_imgs = 0
        self.word_to_img = {}
        self.word_count = []
        self.img_to_histgram = {}
        self.all_des = [] # store all the descriptors for all the image in the database
        self.all_kpts = [] # store all the keypointes  ...
        self.all_image = [] # store all the image paths ...
        self.num_feature_per_image = [] # store number of features for each images, we use it extra corresponding kpts/des
        self.feature_start_idx = [] # feature_start_idx[i] store the start index of img i's descriptor in all_des
        self.kmeans = None
        
    def loadImgs(self, data_path, des_method='SIFT'):
        self.data_path = data_path
        fd = FeatureDetector()
        for subdir, dirs, files in os.walk(self.data_path):
            for f in files:
                img_path = os.path.join(subdir, f)
                img_type = imghdr.what(img_path)
                if (imghdr.what(img_path) != None and img_type in 'png/jpg/jpeg/'):
                    img = cv2.imread(img_path)
                    # get all the kpts and des for each images.
                    # kpts, des = SIFT_match_points_single(img)
                    kpts, des = fd.detect(img, des_method)
                    self.all_des += des.tolist()
                    # self.all_kpts += kpts
                    self.all_image.append(img_path)
                    idx = 0 if len(self.num_feature_per_image) == 0 else self.num_feature_per_image[-1] + self.feature_start_idx[-1]
                    self.num_feature_per_image.append(des.shape[0])
                    self.feature_start_idx.append(idx)
        # turn the list into a np.array
        self.num_imgs = len(self.all_image)
        self.all_des = np.asarray(self.all_des, dtype=np.float32)
        # import pdb; pdb.set_trace()
    
    def run_KMeans(self, num_clusters):
        # self.word_count = np.zeros(num_clusters)
        # self.kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.all_des)
        root = self.hierarchical_KMeans(3, 5, self.all_des)
        import pdb;pdb.set_trace()


    

    def hierarchical_KMeans(self, k, L, all_vec):
        # root find the center
        root = Node()
        root.value = KMeans(n_clusters=k, random_state=0).fit(all_vec)
        if L == 0:
            return root
        for i in range(k):
            cluster_i = all_vec[root.value.labels_ == i]
            node_i = self.hierarchical_KMeans(k, L-1, cluster_i)
            root.children.append(node_i)
        return root
        





    def build_inverted_file_index(self):
        for i in range(len(self.num_feature_per_image)):
            n_feature_img_i = self.num_feature_per_image[i]
            img_path_i = self.all_image[i]
            start_idx = self.feature_start_idx[i]
            end_idx = start_idx + n_feature_img_i 
            for j in range(start_idx, end_idx):
                d_ij = self.all_des[j]
                # calculate the norm to all the centers to check which W does the d_ij belongs to
                norm = np.linalg.norm(self.kmeans.cluster_centers_ - d_ij, axis=1)
                word_idx = np.argmin(norm)
                self.word_count[word_idx] += 1

                # update the word_to_img dictionary
                if word_idx in self.word_to_img:
                    if img_path_i in self.word_to_img[word_idx]:
                        # import pdb; pdb.set_trace()
                        self.word_to_img[word_idx][img_path_i] += 1
                    else:
                        self.word_to_img[word_idx][img_path_i] = 1
                else:
                    self.word_to_img[word_idx] = {}
                    self.word_to_img[word_idx][img_path_i] = 1

                # acculate the bad-of-word description in img_to_histgram dictionary
                n_clusters = norm.shape[0]
                if img_path_i in self.img_to_histgram:
                    self.img_to_histgram[img_path_i][word_idx] += 1
                else:
                    self.img_to_histgram[img_path_i] = np.zeros(n_clusters)

    def query(self, input_img, top_K, method):
        # compute the features
        fd = FeatureDetector()
        kpts, des = fd.detect(input_img, method=method)

        # compute bag-of-word description
        # import pdb;pdb.set_trace()
        n_clusters = self.kmeans.cluster_centers_.shape[0]
        BoW_q = np.zeros(n_clusters)

        # loop all the des to get all the visual words of the input_img
        for d in des:
            norm = np.linalg.norm(self.kmeans.cluster_centers_ - d, axis=1)
            word_idx = np.argmin(norm)
            BoW_q[word_idx] += 1

        # get a list of img from database that have the same visual words
        target_img_lst = []
        for w in range(len(BoW_q)):
            if BoW_q[w] != 0:
                images_dict = self.word_to_img[w]
                for i in images_dict:
                    if i not in target_img_lst:
                        target_img_lst.append(i)
        # compute similarity between query BoW and the all targets 
        similarity_lst = np.zeros(len(target_img_lst))

        q = np.zeros(n_clusters)
        for j in range(len(target_img_lst)):
            img = target_img_lst[j]
            t = np.zeros(n_clusters)
            t =self.img_to_histgram[img] 
            for w in range(n_clusters):
                n_wj = self.img_to_histgram[img][w]
                n_j = np.sum(self.img_to_histgram[img])
                # n_w = self.word_count[w]
                # N = len(self.all_des)
                n_w = len(self.word_to_img[w])
                N = self.num_imgs
                t[w] = (n_wj/n_j) * np.log(N/n_w)
                if j == 0:
                    n_wq = BoW_q[w]
                    n_q = np.sum(BoW_q)
                    q[w] = (n_wq / n_q) * np.log(N/n_w)

            if j == 0:
                q = q/np.linalg.norm(q)
                # import pdb;pdb.set_trace()
            t = t/np.linalg.norm(t)
            # normalize dot product
            similarity_lst[j] = np.linalg.norm(t-q)

        # sort the similarity and take the top_K most similar image
        best_K_match_imgs_idx = np.argsort(similarity_lst)[-top_K:][::-1]
        best_K_match_imgs = [target_img_lst[i] for i in best_K_match_imgs_idx]
        import pdb;pdb.set_trace()

        # TODO: spatial verification
        return best_K_match_imgs




    def save(self, db_name):
        file = open(db_name,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, db_name):
        file = open(db_name,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        


db = Database()
## save test
test_path = '../data/test'
cover_path = '../data/DVDcovers'

db.loadImgs(cover_path, des_method='ORB')
db.run_KMeans(30)
db.build_inverted_file_index()
# db.save('data_.txt')

## load test
# db.load('data_.txt')
# cover = cover_path + '/matrix.jpg'
test = test_path + '/image_07.jpeg'
# test = 'test.png'
test = cv2.imread(test)
print(db.query(test, 50, method='ORB'))
import pdb;pdb.set_trace()