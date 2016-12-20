import math
import numpy as np
import pandas as pd

def generate_spiral(number_of_points, number_of_rotations, rotation_offset=0.0):
    rotation_increment = float(number_of_rotations) / float(number_of_points)
    for data_point_index in range(0, number_of_points):
        rotation = rotation_increment * data_point_index + 0.25
        radius = rotation
        yield (math.sin((rotation + rotation_offset) * 2 * math.pi) * radius, math.cos((rotation + rotation_offset) * 2 * math.pi) * radius)

def create_dataset():
    dataset = []
    for i in generate_spiral(76,2):
        temp = list(i)
        temp.append(0)
        dataset.append(temp)   
    for i in generate_spiral(76,2,rotation_offset = 0.5):
        temp = list(i)
        temp.append(1)
        dataset.append(temp)
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    print dataset
    dataset = pd.DataFrame(dataset,columns = ['x','y','label'])
    labels = dataset.pop('label')
    return dataset.values,labels.values.reshape(152,1)

        
        
        