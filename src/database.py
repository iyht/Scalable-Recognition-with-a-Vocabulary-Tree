import os
import cv2
import imghdr
import time
from feature import *
import pickle


from sklearn.cluster import KMeans

class Database:
    def __init__(self):
        self.data_path = ''
        # self.img_to_des_and_kpts = {}
        self.word_to_img = {}
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
        self.all_des = np.asarray(self.all_des, dtype=np.float32)
        # import pdb; pdb.set_trace()
    
    def run_KMeans(self, num_clusters):
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.all_des)


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

                # update the img_to_histgram dictionary
                if img_path_i in self.img_to_histgram:
                    self.img_to_histgram[img_path_i][word_idx] += 1
                else:
                    self.img_to_histgram[img_path_i] = np.zeros(norm.shape[0])






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

cover_path = '../data/DVDcovers'
db.loadImgs(cover_path)
db.run_KMeans(20)
# db.load('data_.txt')
db.build_inverted_file_index()
import pdb;pdb.set_trace()
db.save('data_.txt')