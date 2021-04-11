import os
import cv2
import imghdr
import time
from sift import *
import pickle


from sklearn.cluster import KMeans

class Database:
    def __init__(self):
        self.data_path = ''
        # self.img_to_des_and_kpts = {}
        # self.word_to_img = {}
        self.all_des = [] # store all the descriptors for all the image in the database
        self.all_kpts = [] # store all the keypointes  ...
        self.all_image = [] # store all the image paths ...
        self.num_feature_per_image = [] # store number of features for each images, we use it extra corresponding kpts/des
        self.kmeans = None
        
    def LoadImgs(self, data_path, des_method='SIFT'):
        self.data_path = data_path
        # start = time.time()
        for subdir, dirs, files in os.walk(self.data_path):
            for f in files:
                img_path = os.path.join(subdir, f)
                img_type = imghdr.what(img_path)
                # print(img_type)
                # print(img_path)
                if (imghdr.what(img_path) != None and img_type in 'png/jpg/jpeg/'):
                    img = cv2.imread(img_path)
                    kpts, des = SIFT_match_points_single(img)
                    self.all_des += des.tolist()
                    # self.all_kpts += kpts
                    self.all_image.append(img_path)
                    self.num_feature_per_image.append(des.shape[0])
        # turn the list into a np.array
        self.all_des = np.asarray(self.all_des, dtype=np.float32)
        # import pdb;pdb.set_trace()
    
    def KMeans(self, num_clusters):
        # import pdb;pdb.set_trace()
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.all_des)


    def save(self, db_name):
        file = open(name,'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self, db_name):
        file = open(db_name,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        
        # end = time.time()
        # print(end - start)



db = Database()
db.load()
import pdb;pdb.set_trace()

# cover_path = '../data/DVDcovers'
# db.LoadImgs(cover_path)
# db.KMeans(5)
# db.save()