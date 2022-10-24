import json
import re
from .utils import *
from tqdm import tqdm

def textwrite(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + "\n")



def pop_range(x, y, min_length=0, max_length=1e10):
    ''' 
    remove senence that have length outside of min_length and max_length
    x is source and y is target sentence
    TODO: this runs pretty slow and needs improvement
    '''
    i = 0
    new_x, new_y = list(), list()
    for i in tqdm(range(len(x))):
        # check for length constrain
        if (min_length <= len(x[i]) <= max_length) and (min_length <= len(y[i]) <= max_length):
            # check for duplication
            if x[i] not in new_x and y[i] not in new_y: 
                new_x.append(x[i])
                new_y.append(y[i])
        else: i+=1
    
    assert len(new_x) == len(new_y)
    
    return new_x, new_y

def resub(data):
    '''
    clean special and unused characters
    '''
    
    data = re.sub('\&#?[a-z0-9]+;', '', data) # remove html entity
    # keep only english and vietnamese characters
    data = re.sub('[^a-z0-9A-Z_ \n\r\.\'?,!àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]','', data)
    data = re.sub(' +', ' ', data) # remove duplicate white space
    
    return data

def pre_processing(x, y, min_length, max_length):
    '''
    clean data using resub then pop items not in pre-defined range
    '''
    
    # resub expect string like object
    print('clean text using re.sub')
    # we use join to make continuous string for re.sub
    x = resub("\n".join(x)) 
    y = resub("\n".join(y))
    x = x.split("\n")
    y = y.split("\n")
    
    print(f"data size before x:{len(x)}, y:{len(y)}")
    # constrant
    print('remove sentence base on min max length')
    x, y = pop_range(x, y, min_length, max_length)
    assert(len(x)==len(y))
    print(f"data size after x:{len(x)}, y:{len(y)}")
    
    # print some samples
    print("Some last sentences")
    for i in range(-1,-5,-1):
        print(x[i],'|',y[i])
    print("\n")

    return x, y


def load_data(paths, min_len, sequence_len):
    #process and save data to cleaned folder
    X, Y = list(), list()
    
    for path in paths:
        print("processing path: ", path)
        x_path, y_path = path
        x = open(x_path, encoding='utf-8').read().split("\n")
        y = open(y_path, encoding='utf-8').read().split("\n")
        
        # pre-process
        x,y = pre_processing(x, y, min_length=min_len, max_length=sequence_len) 
        
        # add each file content to parent list
        X += x
        Y += y
    
    return X, Y
    
    # nine_nine_percentile = int(np.percentile([len(sen) for sen in vi],99))