import json
import re

def textwrite(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + "\n")

        
# if __name__ == "__main__":
#     textwrite('test.txt', ['cháu lên ba', 'cháu vô mẫu giáo'])

def pop_range(x, y, min_length=0, max_length=1e10):
    i = 0
    new_x, new_y = list(), list()
    for i in range(len(x)):
        if (min_length <= len(x[i]) <= max_length) and (min_length <= len(y[i]) <= max_length):
            new_x.append(x[i])
            new_y.append(y[i])
        else: i+=1
    
    assert len(new_x) == len(new_y)
    
    return new_x, new_y

def resub(data):
    data = re.sub('\&#?[a-z0-9]+;', '', data) # remove html entity
    # keep only english and vietnamese characters
    data = re.sub('[^a-z0-9A-Z_ \n\r\.\'?,!àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]','', data)
    data = re.sub(' +', ' ', data) # remove duplicate white space
    
    return data

def pre_processing(x, y, min_length, max_length):
    x, y = pop_range(x, y, min_length, max_length)
    data = '\n'.join(x+y)
    data = resub(data)
    data = data.split("\n")
    assert len(data)%2 == 0, "error happens when split data"
    x, y = data[:len(data)//2], data[len(data)//2:]

    # print some samples
    print("Some last sentences")
    for i in range(-1,-5,-1):
        print(x[i],'|',y[i])

    return x, y