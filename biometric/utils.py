import os, errno
from cv2 import cv2
import facenet
from biometric import detect_face
import numpy as np
from django.conf import settings
from .apps import BiometricConfig
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

def aling_image(path_temp_image,image_name,username,save_directory):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    pnet = BiometricConfig.pnet
    rnet = BiometricConfig.rnet
    onet = BiometricConfig.onet
    try:
        img = cv2.imread(path_temp_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (img.shape[0]/img.shape[1]) > 1:
            img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_LINEAR)
        else :
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(path_temp_image, e)
    else:
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)

                """To prevent the resized image from been skewed, the bounding box needs to be
                adjusted to become a square. Also, the size of the margin needs to be mapped
                from the the target scaled image space to the size in the original image space."""
                dx = det[2] - det[0]
                dy = det[3] - det[1]
                margin = 0
                if(dy >= dx):
                    margin =  dy * 44 / (182 - 44)
                    bb[0] = np.maximum(det[0] - (dy-dx)/2 - margin/2, 0)
                    bb[2] = np.minimum(det[2] + (dy-dx)/2 + margin/2, img_size[1])
                    bb[1] = np.maximum(det[1] - margin/2, 0)
                    bb[3] = np.minimum(det[3] + margin/2, img_size[0])
                else:
                    margin =  dx * 44 / (182 - 44)
                    bb[0] = np.maximum(det[0] - margin/2, 0)
                    bb[2] = np.minimum(det[2] + margin/2, img_size[1])
                    bb[1] = np.maximum(det[1]-(dx-dy)/2 - margin/2, 0)
                    bb[3] = np.minimum(det[3] + (dx-dy)/2 + margin/2, img_size[0])

                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = cv2.resize(cropped, (182, 182), interpolation=cv2.INTER_LINEAR)
                user_dir_path = os.path.join(settings.FCPATH, save_directory+username)
                try:
                    os.makedirs(user_dir_path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        #directory already exists
                        pass
                path_image_saved = os.path.join(user_dir_path,image_name)
                cv2.imwrite(path_image_saved, cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))
                return path_image_saved

def encrypt_image(pathImage, encryKey):
    with open(pathImage, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        key = hashlib.sha256(encryKey.encode()).digest()
        iv = Random.new().read(AES.block_size)
        aes = AES.new(key, AES.MODE_CBC, iv)
        data = encoded_string + (AES.block_size - len(encoded_string) % AES.block_size) * chr(AES.block_size - len(encoded_string) % AES.block_size)
        imgData = base64.b64encode(iv + aes.encrypt(data.encode()))

    if os.path.exists(pathImage):
        os.remove(pathImage)

    with open(pathImage, 'wb') as imageEncrypt:
        imageEncrypt.write(imgData)

    return pathImage

def dencrypt_image(pathImage, encryKey):
    with open(pathImage, "rb") as image_file:
        key = hashlib.sha256(encryKey.encode()).digest()
        encoded_string = base64.b64decode(image_file.read())
        iv = encoded_string[:AES.block_size]
        cipher = AES.new(key, AES.MODE_CBC, iv )
        data = cipher.decrypt( encoded_string[AES.block_size:] )
        imgdata = data[:-ord(data[len(data)-1:])]

    return imgdata

def dencrypt_image_facenet(pathImage, encryKey, pathTemp):
    with open(pathImage, "rb") as image_file:
        key = hashlib.sha256(encryKey.encode()).digest()
        encoded_string = base64.b64decode(image_file.read())
        iv = encoded_string[:16]
        cipher = AES.new(key, AES.MODE_CBC, iv )
        data = cipher.decrypt( encoded_string[16:] )
        imgdata = data[:-ord(data[len(data)-1:])]
        imgWr = base64.b64decode(imgdata)

    #if os.path.exists(pathImage):
     #   os.remove(pathImage)

    with open(pathTemp, 'wb') as imageDencrypt:
        imageDencrypt.write(imgWr)

    return pathTemp