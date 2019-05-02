import os
import numpy as np
import xml.etree.ElementTree as ElementTree
import h5py
classes = ["plate"]
voc_path=u'./data/PascalVOC/VOCdevkit'
train_set = [('2012', 'train')]
def get_boxes_for_id(voc_path, year, image_id):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    year : str
        Year of dataset containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xmin, ymin, xmax, ymax as a
        5xN array.
    """
    fname = os.path.join(voc_path, 'VOC{}/Annotations/{}.xml'.format(year,
                                                                     image_id))
    with open(fname) as in_file:
        xml_tree = ElementTree.parse(in_file)
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in classes or int(
                difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = (classes.index(label), int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text))
        boxes.extend(bbox)
    return np.array(
        boxes)  # .T  # return transpose so last dimension is variable length
def get_image_for_id(voc_path, year, image_id):
    """Get image data as uint8 array for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    year : str
        Year of dataset containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    fname = os.path.join(voc_path, 'VOC{}/JPEGImages/{}.jpg'.format(year,
                                                                    image_id))
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')

def add_to_dataset(voc_path, year, ids, images, boxes,start=0):
    """Process all given ids and adds them to given datasets."""
    for i, voc_id in enumerate(ids):
        #print(i)
        #print(voc_id)
        image_data = get_image_for_id(voc_path, year, voc_id)
        image_boxes = get_boxes_for_id(voc_path, year, voc_id)
       # images.append(image_data)
       # boxes.append(image_boxes)
        images[i+start]=image_data
        boxes[i+start]=image_boxes
        # images[i]=image_data
        # boxes[i]= image_boxes
    return i

def get_ids(voc_path, datasets):
    """Get image identifiers for corresponding list of dataset identifies.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    datasets : list of str tuples
        List of dataset identifiers in the form of (year, dataset) pairs.

    Returns
    -------
    ids : list of str
        List of all image identifiers for given datasets.
    """
    ids = []
    #for year, image_set in datasets:
    id_file_train= './data/PascalVOC/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
   # id_file_test='./data/PascalVOC/VOCdevkit/VOC2012/ImageSets/Main/test.txt'
   # list_file=[id_file_train ,id_file_test]
    #for i in id_file_train:
       # print("打印 ---------------------------------------------------i",i)
       #  print(i[0])
        # id_file = os.path.join(voc_path, 'V0C{}/ImageSets/Main/{}.txt'.format(
        #     year, image_set))
        # id_file =voc_path + '/'+'V0C{}/ImageSets/Main/{}.txt'.format(
        #     year, image_set)
        #print(str(id_file))
    # print(i)
    #print(list_file[i],'\n')
    with open(id_file_train, 'r',encoding='utf-8') as image_ids:
        ids.extend(map(str.strip, image_ids.readlines()))
          #print(map(str.strip, image_ids.readlines()))
        # print(i)
      #  print(ids)
    return ids

train_ids = get_ids(voc_path, train_set)
#print(train_ids)
# images=[]
# boxes=[]
# #for train_ids  in train_ids:
#     #get_image_for_id(voc_path, 2012, train_ids)
# i =add_to_dataset(voc_path, 2012, train_ids, images, boxes)
#print(images)
#print(i+1)
#print(images)  #uint8 array的图像数据
#print(boxes)   #array的数据

#打印数据有多少个
print(len(train_ids))
total_train_ids = len(train_ids)
fname = os.path.join(voc_path, 'pascal_voc_07_12.hdf5')
voc_h5file = h5py.File(fname, 'w')
uint8_dt = h5py.special_dtype(
    vlen=np.dtype('uint8'))  # variable length uint8
vlen_int_dt = h5py.special_dtype(
    vlen=np.dtype(int))  # variable length default int
train_group = voc_h5file.create_group('train')

# store class list for reference class ids as csv fixed-length numpy string
voc_h5file.attrs['classes'] = np.string_(str.join(',', classes))

# store images as variable length uint8 arrays
train_images = train_group.create_dataset(
    'images', shape=(total_train_ids,), dtype=uint8_dt)

# store boxes as class_id, xmin, ymin, xmax, ymax
train_boxes = train_group.create_dataset(
    'boxes', shape=(total_train_ids,), dtype=vlen_int_dt)

# process all ids and add to datasets
print('Processing Pascal V0C 2012 datasets for training set.')
last_2007 = add_to_dataset(voc_path, 2012, train_ids, train_images,
                           train_boxes)
print("last_2007",last_2007)
# print('Processing Pascal VOC 2012 training set.')


print('Closing HDF5 file.')
voc_h5file.close()
print('Done.')
