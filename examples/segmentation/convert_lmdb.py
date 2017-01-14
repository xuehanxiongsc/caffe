import getopt,sys
import numpy as np
import cv2
import lmdb
import caffe
import os

RED_CHANNEL = 2
GREEN_CHANNEL = 1
BLUE_CHANNEL = 0
TOLERANCE = 10
RESIZE_DIM = 128

def is_image(file):
    if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg")     or file.endswith(".png") or file.endswith(".JPEG"):
        return True
    return False

def get_image_files(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir) if is_image(os.path.join(dir, f))]

def red_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,RED_CHANNEL],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,GREEN_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,BLUE_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def green_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,GREEN_CHANNEL],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,BLUE_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,RED_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def blue_mask(image):
    temp,mask0 = cv2.threshold(image[:,:,BLUE_CHANNEL],255-TOLERANCE,255,cv2.THRESH_BINARY)
    temp,mask1 = cv2.threshold(image[:,:,RED_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    temp,mask2 = cv2.threshold(image[:,:,GREEN_CHANNEL],TOLERANCE,255,cv2.THRESH_BINARY_INV)
    mask0 = cv2.bitwise_and(mask0,mask1)
    mask0 = cv2.bitwise_and(mask0,mask2)
    return mask0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _gen_label_image(image):
    hair_mask = red_mask(image)
    face_mask = green_mask(image)
    shoulder_mask = blue_mask(image)
    label_image = np.zeros_like(image[:,:,0])
    label_image[hair_mask==255] = 1
    label_image[shoulder_mask==255] = 2
    label_image[face_mask==255] = 3
    kernel_width = np.maximum(int(label_image.shape[0]/100.0),5)
    kernel = np.ones((kernel_width,kernel_width),np.uint8)
    label_image = cv2.morphologyEx(label_image, cv2.MORPH_CLOSE, kernel)
    return label_image

def _resize_image_label(image,label):
    height,width = image.shape[:2]
    scale = float(RESIZE_DIM) / float(np.minimum(width,height))
    resized_image = cv2.resize(image,(0,0),fx=scale,fy=scale)
    resized_label = cv2.resize(label,(resized_image.shape[1],resized_image.shape[0]),interpolation=cv2.INTER_NEAREST)
    assert resized_image.shape[:2] == resized_label.shape[:2]
    assert np.minimum(resized_image.shape[0],resized_image.shape[1]) == RESIZE_DIM
    return resized_image,resized_label

def _change_background(bgd,image,label):
    max_image_dim = np.maximum(image.shape[0],image.shape[1])
    min_bgd_dim = np.minimum(bgd.shape[0],bgd.shape[1])
    scale = float(max_image_dim)/float(min_bgd_dim)
    if scale < 1.0:
        resized_bgd = cv2.resize(bgd,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    else:
        resized_bgd = cv2.resize(bgd,None,fx=scale,fy=scale)
    image_width = image.shape[1]
    image_height = image.shape[0]
    offset_x = np.random.randint(resized_bgd.shape[1]-image_width+1)
    offset_y = np.random.randint(resized_bgd.shape[0]-image_height+1)
    crop_bgd = resized_bgd[offset_y:(offset_y+image_height),offset_x:(offset_x+image_width),:]
    mask = (label == 0).astype(np.float)
    mask = cv2.blur(mask,(5,5))
    mask = np.expand_dims(mask,axis=2)
    image2 = crop_bgd*mask + (1-mask)*image
    return image2.astype(np.uint8)

def writeLMDB(image_files, label_files, bgd_files, lmdb_path):
    assert len(image_files) == len(label_files)
    num_examples = len(image_files)
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    write_count = 0
    for idx in xrange(num_examples):
        img = cv2.imread(image_files[idx])
        label_image = cv2.imread(label_files[idx])
        label = _gen_label_image(label_image)
        resized_image,resized_label = _resize_image_label(img,label)
        resized_label = np.expand_dims(resized_label,axis=2)
        img4ch = np.concatenate((resized_image, resized_label), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        datum = caffe.io.array_to_datum(img4ch, label=0)
        key = '%07d' % write_count
        txn.put(key, datum.SerializeToString())
        write_count = write_count + 1
        if(write_count % 1000 == 0):
            txn.commit()
            txn = env.begin(write=True)
            print 'count: %d / total count: %d' % (write_count,num_examples)
    if(write_count % 1000 != 0):
        print 'count: %d/ total count: %d' % (write_count,num_examples)
        txn.commit()
    env.close()

def usage():
    print 'convert_lmdb.py --data_dir=<data directory>'
    
def main(argv):
    root_dir = ''
    try:
        opts, args = getopt.getopt(
            argv,
            "h:d:",
            ["data_dir="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","--help"):
            usage()
            sys.exit()
        elif opt in ("-d","--data_dir"):
            root_dir = arg
    bgd_dir = os.path.join(root_dir,'segmentation_background')
    bgd_files = get_image_files(bgd_dir)
    image_dir = os.path.join(root_dir,'train/images')
    label_dir = os.path.join(root_dir,'train/labels')
    image_files = get_image_files(image_dir)
    label_files = get_image_files(label_dir)
    print 'write %s' % os.path.join(root_dir,'portrait_train')
    writeLMDB(image_files, label_files, bgd_files, 
              os.path.join(root_dir,'portrait_train'))
    # write validation data
    image_dir = os.path.join(root_dir,'val/images')
    label_dir = os.path.join(root_dir,'val/labels')
    image_files = get_image_files(image_dir)
    label_files = get_image_files(label_dir)
    print 'write %s' % os.path.join(root_dir,'portrait_val')
    writeLMDB(image_files, label_files, bgd_files, 
              os.path.join(root_dir,'portrait_val'))

if __name__ == "__main__":
    main(sys.argv[1:])
