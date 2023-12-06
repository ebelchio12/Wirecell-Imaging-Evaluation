from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, cKDTree

missing_events = [325,327,328,329,351,376,380, 499]

def load_data(path_to_data:str, nevents:int,
              img_str="imaging", tru_str="truthDepo"):
    """
    Loads the data from the json files into a list of numpy arrays.
    Outputs a list for imaging and a list for truth. 
    
    Parameters
    ----------
    path_to_data : str
        Path to the data folder
    nevents : int
    img_str : str
        String to identify the imaging data file
    tru_str : str
        String to identify the truth data file
    
    Returns
    -------
    img_arr_list : list
        List of numpy arrays for imaging data
    tru_arr_list : list
        List of numpy arrays for truth data
    """
    
    img_arr_list = []
    tru_arr_list = []
    for event in tqdm(range(nevents)):
 
        # Ewerton (skip evetns missing in input data dir
        if event in missing_events:
          print('\nEvent {} is missing in inputdir..'.format(event))
          continue
  
        img_path = path_to_data+"/"+str(event)+"/"+str(event)+"-"+img_str+".json"
        tru_path = path_to_data+"/"+str(event)+"/"+str(event)+"-"+tru_str+".json"
        img_file = open(img_path)
        img_data = json.load(img_file)
        img_x = img_data['x']; img_y = img_data['y']; img_z = img_data['z']; img_q = img_data['q']
        img_file.close()

        tru_file = open(tru_path)
        tru_data = json.load(tru_file)
        tru_x = tru_data['x']; tru_y = tru_data['y']; tru_z = tru_data['z']; tru_q = tru_data['q']
        mask = np.where(np.array(tru_x) <0)[0]
        tru_x = np.array(tru_x)[mask]; tru_y = np.array(tru_y)[mask]; tru_z = np.array(tru_z)[mask]; tru_q = np.array(tru_q)[mask]
        tru_file.close()

        img_arr = np.column_stack([img_x, img_y, img_z, img_q])
        tru_arr = np.column_stack([tru_x, tru_y, tru_z, tru_q])
        img_arr_list.append(img_arr)
        tru_arr_list.append(tru_arr)
    return img_arr_list, tru_arr_list

def find_nearest(inner_arr:np.ndarray, outer_arr: np.ndarray ,max_dist:float):
    """
    Finds the nearest neighbor of each point in inner array to a point in outer array.
    
    Parameters
    ----------
    inner_arr : np.ndarray
    outer_arr : np.ndarray
    max_dist : float
        Maximum distance to search for nearest neighbor
        
    Returns
    -------
    inner_match_arr : np.ndarray
        Array of points that were matched succesfully.
    inner_miss_arr : np.ndarray
        Array of points with no match. 
    outer_match_arr : np.ndarray
        Array of nearest neighbors of inner_match_arr. 
    outer_miss_arr : np.ndarray
        Array of outer_arr points that are not mathched to any point in inner arr.
    """

    # create the KD trees 
    inner_pos_arr = inner_arr[:,:3]
    outer_pos_arr = outer_arr[:,:3]
    inner_tree = cKDTree(inner_pos_arr) 
    outer_tree = cKDTree(outer_pos_arr) 
    
    # get the indices of the nearest neighbors from the outer tree 
    match_idx = inner_tree.query_ball_tree(outer_tree, max_dist)

    # given the indices of the nearest neighbors within the max_dist, find the index of the actual nearest neighbor
    # store the index of the nearest neighbor (of outer) in min_indices array
    min_indices = np.zeros(len(inner_arr),dtype=int)
    for in_idx in range(len(inner_arr)): # looping over all entries in inner array
        if len(match_idx[in_idx]) == 0: # if there are no matches 
            min_indices[in_idx] = -1
            continue
        min_idx = int(np.argmin(np.linalg.norm(inner_pos_arr[in_idx]-outer_pos_arr[match_idx[in_idx]],axis=1)))
        min_indices[in_idx] = match_idx[in_idx][min_idx]
        
    # array of points that are the nearest neighbors
    outer_match_arr = outer_arr[min_indices]
    
    found_match = np.where(min_indices != -1,True,False)
    outer_match_arr = outer_match_arr[found_match]
    inner_match_arr = inner_arr[found_match]
    # array of inner points that didn't get matched to any outer point 
    inner_miss_arr = inner_arr[~found_match]
    # array of outer points that didn't get matched to any inner points
    outer_miss_arr = outer_arr[~np.isin(np.arange(len(outer_arr)),min_indices)]
    return inner_match_arr, inner_miss_arr, outer_match_arr, outer_miss_arr
