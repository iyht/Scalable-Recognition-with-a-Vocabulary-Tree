import os
import cv2
import imghdr
import time
from feature import *
from homography import *
import pickle
from sklearn.cluster import KMeans

class Node:
    def __init__(self):
        self.value = None
        self.kmeans = None
        self.children = []
        # self.imgs_count = {}
        self.occurrences_in_img = {}
        self.index = None
        # self.img_count = {}

class Database:
    def __init__(self):
        self.data_path = ''
        # self.img_to_des_and_kpts = {}
        self.num_imgs = 0
        self.word_to_img = {}
        self.BoW = {}
        self.word_count = []
        self.img_to_histgram = {}
        self.all_des = [] # store all the descriptors for all the image in the database
        self.all_kpts = [] # store all the keypointes  ...
        self.all_image = [] # store all the image paths ...
        self.num_feature_per_image = [] # store number of features for each images, we use it extra corresponding kpts/des
        self.feature_start_idx = [] # feature_start_idx[i] store the start index of img i's descriptor in all_des
        self.kmeans = None
        self.total_words_in_img = {}
        self.word_idx_count = 0

        self.vocabulary_tree = None
        
    # def loadImgs(self, data_path, des_method='SIFT'):
    #     self.data_path = data_path
    #     fd = FeatureDetector()
    #     for subdir, dirs, files in os.walk(self.data_path):
    #         for f in files:
    #             img_path = os.path.join(subdir, f)
    #             img_type = imghdr.what(img_path)
    #             if (imghdr.what(img_path) != None and img_type in 'png/jpg/jpeg/'):
    #                 img = cv2.imread(img_path)
    #                 # get all the kpts and des for each images.
    #                 # kpts, des = SIFT_match_points_single(img)
    #                 kpts, des = fd.detect(img, des_method)
    #                 self.all_des += des.tolist()
    #                 # self.all_kpts += kpts
    #                 self.all_image.append(img_path)
    #                 idx = 0 if len(self.num_feature_per_image) == 0 else self.num_feature_per_image[-1] + self.feature_start_idx[-1]
    #                 self.num_feature_per_image.append(des.shape[0])
    #                 self.feature_start_idx.append(idx)
    #     # turn the list into a np.array
    #     self.num_imgs = len(self.all_image)
    #     self.all_des = np.asarray(self.all_des, dtype=np.float32)
    #     # import pdb; pdb.set_trace()



    def loadImgs(self, data_path, method='SIFT'):
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
                    kpts, des = fd.detect(img, method)
                    self.all_des += [[d, img_path] for d in des.tolist()]
                    self.all_image.append(img_path)
                    idx = 0 if len(self.num_feature_per_image) == 0 else self.num_feature_per_image[-1] + self.feature_start_idx[-1]
                    self.num_feature_per_image.append(des.shape[0])
                    self.feature_start_idx.append(idx)
        # turn the list into a np.array
        self.num_imgs = len(self.all_image)
        self.all_des = np.array(self.all_des, dtype=object)
        # import pdb; pdb.set_trace()
    
    def run_KMeans(self, k, L):
        # self.kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.all_des)
        total_nodes = (k*(k**L)-1)/(k-1)
        n_leafs = k**L
        self.word_count = np.zeros(n_leafs)
        self.vocabulary_tree = self.hierarchical_KMeans(k,L, self.all_des)
        self.print_tree(self.vocabulary_tree)
        # import pdb;pdb.set_trace()


    def print_tree(self, node):
        children = node.children
        if len(children) == 0:
            print(node.index)
        else:
            for c in children:
                self.print_tree(c)

    

    def hierarchical_KMeans(self, k, L, des_and_path):
        # devide the given des vector in to k cluster
        des = [ pair[0] for pair in des_and_path]
        root = Node()
        root.kmeans = KMeans(n_clusters=k, random_state=0).fit(des)
        root.value = des_and_path

        # we reach the leaf node
        if L == 0:
            # assign the index to the leaf nodes.
            root.index = self.word_idx_count
            self.word_idx_count += 1

            # count the number of occurrences of a word in a image used in tf-idf
            for pair in root.value:
                img_path = pair[1]
                if img_path not in root.occurrences_in_img:
                    root.occurrences_in_img[img_path] = 1
                else:
                    root.occurrences_in_img[img_path] += 1
            
            self.word_count[root.index] = len(root.occurrences_in_img)
            # # accumulate the total numbre of words of a image used in tf-idf
            # for img_path, count in root.occurrences_in_img.items():
            #     if img_path not in self.total_words_in_img:
            #         self.total_words_in_img[img_path] = count
            #     else:
            #         self.total_words_in_img[img_path] += count

            # calculate the number of occurrences of word in the whole database
            return root

        # if we are not on the leaf level, then for each cluster, 
        # we recursively run KMean
        for i in range(k):
            cluster_i = des_and_path[root.kmeans.labels_ == i]
            node_i = self.hierarchical_KMeans(k, L-1, cluster_i)
            root.children.append(node_i)
        return root
        
    
    def build_histgram(self, node):
        '''
        build the histgram for the leaf nodes
        '''
        
        children = node.children
        if len(children) == 0:
            for img, count in node.occurrences_in_img.items():
                # print(img)
                if img not in self.img_to_histgram:
                    self.img_to_histgram[img] = np.zeros(self.word_idx_count)
                    self.img_to_histgram[img][node.index] += count
                else:
                    self.img_to_histgram[img][node.index] += count
        else:
            for c in children:
                self.build_histgram(c)
        
    def build_BoW(self):
        for j in range(len(self.all_image)):
            img = self.all_image[j]
            t = np.zeros(self.word_idx_count)
            t =self.img_to_histgram[img] 
            for w in range(self.word_idx_count):
                n_wj = self.img_to_histgram[img][w]
                n_j = np.sum(self.img_to_histgram[img])
                n_w = self.word_count[w]
                # # N = len(self.all_des)
                # n_w = len(self.word_to_img[w])
                N = self.num_imgs
                t[w] = (n_wj/n_j) * np.log(N/n_w)
            self.BoW[img] = t

    def spatial_verification(self, query, img_path_lst, method):
        fd = FeatureDetector()
        max_inliers = np.NINF
        max_img_path = None
        for img_path in img_path_lst:
            img = cv2.imread(img_path)
            correspondences = fd.detect_and_match(img, query, method=method)
            print('Running RANSAC with {}'.format(img_path))
            inliers, optimal_H = RANSAC_find_optimal_Homography(correspondences, num_rounds=2000)
            if max_inliers < inliers:
                max_inliers = inliers
                max_img_path = img_path
        return max_img_path







    # def build_inverted_file_index(self):
    #     for i in range(len(self.num_feature_per_image)):
    #         n_feature_img_i = self.num_feature_per_image[i]
    #         img_path_i = self.all_image[i]
    #         start_idx = self.feature_start_idx[i]
    #         end_idx = start_idx + n_feature_img_i 
    #         for j in range(start_idx, end_idx):
    #             d_ij = self.all_des[j]
    #             # calculate the norm to all the centers to check which W does the d_ij belongs to
    #             norm = np.linalg.norm(self.kmeans.cluster_centers_ - d_ij, axis=1)
    #             word_idx = np.argmin(norm)
    #             self.word_count[word_idx] += 1

    #             # update the word_to_img dictionary
    #             if word_idx in self.word_to_img:
    #                 if img_path_i in self.word_to_img[word_idx]:
    #                     # import pdb; pdb.set_trace()
    #                     self.word_to_img[word_idx][img_path_i] += 1
    #                 else:
    #                     self.word_to_img[word_idx][img_path_i] = 1
    #             else:
    #                 self.word_to_img[word_idx] = {}
    #                 self.word_to_img[word_idx][img_path_i] = 1

    #             # acculate the bad-of-word description in img_to_histgram dictionary
    #             n_clusters = norm.shape[0]
    #             if img_path_i in self.img_to_histgram:
    #                 self.img_to_histgram[img_path_i][word_idx] += 1
    #             else:
    #                 self.img_to_histgram[img_path_i] = np.zeros(n_clusters)

    # def query(self, input_img, top_K, method):
    #     # compute the features
    #     fd = FeatureDetector()
    #     kpts, des = fd.detect(input_img, method=method)

    #     # compute bag-of-word description
    #     # import pdb;pdb.set_trace()
    #     n_clusters = self.kmeans.cluster_centers_.shape[0]
    #     BoW_q = np.zeros(n_clusters)

    #     # loop all the des to get all the visual words of the input_img
    #     for d in des:
    #         norm = np.linalg.norm(self.kmeans.cluster_centers_ - d, axis=1)
    #         word_idx = np.argmin(norm)
    #         BoW_q[word_idx] += 1

    #     # get a list of img from database that have the same visual words
    #     target_img_lst = []
    #     for w in range(len(BoW_q)):
    #         if BoW_q[w] != 0:
    #             images_dict = self.word_to_img[w]
    #             for i in images_dict:
    #                 if i not in target_img_lst:
    #                     target_img_lst.append(i)
    #     # compute similarity between query BoW and the all targets 
    #     score_lst = np.zeros(len(target_img_lst))

    #     q = np.zeros(n_clusters)
    #     for j in range(len(target_img_lst)):
    #         img = target_img_lst[j]
    #         t = np.zeros(n_clusters)
    #         t =self.img_to_histgram[img] 
    #         for w in range(n_clusters):
    #             n_wj = self.img_to_histgram[img][w]
    #             n_j = np.sum(self.img_to_histgram[img])
    #             # n_w = self.word_count[w]
    #             # N = len(self.all_des)
    #             n_w = len(self.word_to_img[w])
    #             N = self.num_imgs
    #             t[w] = (n_wj/n_j) * np.log(N/n_w)
    #             if j == 0:
    #                 n_wq = BoW_q[w]
    #                 n_q = np.sum(BoW_q)
    #                 q[w] = (n_wq / n_q) * np.log(N/n_w)

    #         if j == 0:
    #             q = q/np.linalg.norm(q)
    #             # import pdb;pdb.set_trace()
    #         t = t/np.linalg.norm(t)
    #         # normalize dot product
    #         score_lst[j] = np.linalg.norm(t-q)

    #     # sort the similarity and take the top_K most similar image
    #     best_K_match_imgs_idx = np.argsort(score_lst)[-top_K:][::-1]
    #     best_K_match_imgs = [target_img_lst[i] for i in best_K_match_imgs_idx]
    #     import pdb;pdb.set_trace()

    #     # TODO: spatial verification
    #     return best_K_match_imgs

    def get_leaf_nodes(self, root, des):
        children = root.children
        if len(children) == 0:
            # import pdb;pdb.set_trace()
            return root
        
        norm = np.linalg.norm(root.kmeans.cluster_centers_ - des, axis=1)
        child_idx = np.argmin(norm)
        return self.get_leaf_nodes(children[child_idx], des)

    def query(self, input_img, top_K, method):
        # compute the features
        fd = FeatureDetector()
        kpts, des = fd.detect(input_img, method=method)
        # import pdb;pdb.set_trace()

        q = np.zeros(self.word_idx_count)
        node_lst = []
        for d in des:
            node = self.get_leaf_nodes(self.vocabulary_tree, d)
            node_lst.append(node)
            q[node.index] += 1

        for w in range(self.word_idx_count):
            n_w = self.word_count[w]
            N = self.num_imgs
            n_wq = q[w]
            n_q = np.sum(q)
            q[w] = (n_wq / n_q) * np.log(N/n_w)

        # get a list of img from database that have the same visual words
        target_img_lst = []
        for n in node_lst:
            for img, count in n.occurrences_in_img.items():
                if img not in target_img_lst:
                    target_img_lst.append(img)

        # compute similarity between query BoW and the all targets 
        score_lst = np.zeros(len(target_img_lst))


        # import pdb;pdb.set_trace()
        for j in range(len(target_img_lst)):
            img = target_img_lst[j]
            # t = np.zeros(self.word_idx_count)
            # t =self.img_to_histgram[img] 
            # for w in range(self.word_idx_count):
            #     n_wj = self.img_to_histgram[img][w]
            #     n_j = np.sum(self.img_to_histgram[img])
            #     n_w = self.word_count[w]
            #     # # N = len(self.all_des)
            #     # n_w = len(self.word_to_img[w])
            #     N = self.num_imgs
            #     t[w] = (n_wj/n_j) * np.log(N/n_w)
            #     if j == 0:
            #         n_wq = q[w]
            #         n_q = np.sum(q)
            #         q[w] = (n_wq / n_q) * np.log(N/n_w)
            t = self.BoW[img]

            # if j == 0:
                # q = q /np.linalg.norm(q)
                # import pdb;pdb.set_trace()
            # t = t /np.linalg.norm(t)
            # normalize dot product
            # score_lst[j] = np.linalg.norm(t-q, ord=1)
            score_lst[j] = 2 + np.sum(np.abs(q - t) - np.abs(q) - np.abs(t)) 



        # sort the similarity and take the top_K most similar image
        # best_K_match_imgs_idx = np.argsort(score_lst)[-top_K:][::-1]
        best_K_match_imgs_idx = np.argsort(score_lst)[:top_K]
        best_K_match_imgs = [target_img_lst[i] for i in best_K_match_imgs_idx]
        print(best_K_match_imgs)
        print(score_lst)
        
        best_img = self.spatial_verification(input_img, best_K_match_imgs, method)
        return best_img


    def save(self, db_name):
        file = open(db_name,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, db_name):
        file = open(db_name,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
    
def build_database(load_path, k, L, method, save_path):
    print('Initial the Database')
    db = Database()

    print('Loading the images from {}, use {} for features'.format(load_path, method))
    db.loadImgs(load_path, method=method)

    print('Building Vocabulary Tree, with {} clusters, {} levels'.format(k, L))
    db.run_KMeans(k=k, L=L)

    print('Building Histgram for each images')
    db.build_histgram(db.vocabulary_tree)

    print('Building BoW for each images')
    db.build_BoW()

    print('Saving the database to {}'.format(save_path))
    db.save(save_path)
        



db = Database()
## save test
test_path = '../data/test'
cover_path = '../data/DVDcovers'


# build_database(cover_path, k=5, L=5, method='ORB', save_path='data_orb.txt')
## load test
db.load('data_orb.txt')
# cover = cover_path + '/matrix.jpg'
test = test_path + '/image_07.jpeg'
# test = 'test.png'
test = cv2.imread(test)
print(db.query(test, 20, method='ORB'))
import pdb;pdb.set_trace()