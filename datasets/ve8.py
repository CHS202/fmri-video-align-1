import torch
import jpeg4py as jpeg
import torch.utils.data as data
from torchvision import get_image_backend
from PIL import Image
import json
import os
import functools
import numpy as np
import pandas as pd
import h5py
import random

# --- add by myself ---
import torchvision.transforms as transforms

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        if value['subset'] == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader



def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        video.append(image_loader(image_path))
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def npy_loader(raw_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(raw_path, '{:06d}.npy'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        video.append(Image.fromarray(np.load(image_path)))
    return video

def gpu_loader(video_path,frame_indices,decoder):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        image_data = open(image_path, 'rb').read()
        video.append(decoder.decode(image_data))
    return video
def jpeg_loader(video_path,frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        image = jpeg.JPEG(image_path).decode()
        video.append(Image.fromarray(image))
    return video
class VE8Dataset(data.Dataset):
    def __init__(self,
                 opt,
                 video_path,
                 annotation_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 dataset_choose = 've8',
                 get_loader=get_default_video_loader,):
        self.data, self.class_names = make_dataset(opt,
            video_root_path=video_path,
            annotation_path=annotation_path,
            subset=subset,
            fps=fps,
            dataset_choose=dataset_choose
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
    def __getitem__(self, index):
        data_item = self.data[index]
        video_path = data_item['video']
        raw_path = data_item['raw']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)
        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)

        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)
        target = self.target_transform(data_item)
        visualization_item = [data_item['video_id']]

        return snippets, target, visualization_item

    def __len__(self):
        return len(self.data)



class NeuralDataset(data.Dataset):
    def __init__(self,opt,
                 neural_video_path,
                 neural_response,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,):
        self.neural_data = make_neural_dataset(opt=opt,
            video_root_path=neural_video_path,
            neural_response=neural_response,
            fps=fps,
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30

    def __getitem__(self, index):
        data_item = self.neural_data[index]
        video_path = data_item['video']
        raw_path = data_item['raw']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)


        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)

        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        neural_snippets = torch.stack(snippets, 0)

        neural_visualization_item = [data_item['video_id']]
        neural_response = data_item['neural']
        target = data_item['label'] # for LSTM

        return neural_snippets, neural_response,neural_visualization_item, target # return target for LSTM

    def __len__(self):
        return len(self.neural_data)

class BehaviorDataset(data.Dataset):
    def __init__(self,
                 neural_video_path,
                 behavior_response,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,):
        self.behavior_data = make_behavior_dataset(
            video_root_path=neural_video_path,
            behavior_response=behavior_response,
            fps=fps,
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30

    def __getitem__(self, index):
        data_item = self.behavior_data[index]
        raw_path = data_item['raw']
        video_path = data_item['video']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)


        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)

        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        neural_snippets = torch.stack(snippets, 0)

        neural_visualization_item = [data_item['video_id']]
        neural_response = data_item['behavior']

        return neural_snippets, neural_response,neural_visualization_item

    def __len__(self):
        return len(self.behavior_data)

def get_video_class(opt,subset,annotation_path):
    print('split=', opt.split)
    if opt.dataset_choose == 've8':
        split_train = np.load('train_idx_ve8.npy')
        split_test = np.load('test_idx_ve8.npy')
    elif opt.dataset_choose == 'ek6':
        split_train = np.load('train_idx_ek6.npy')
        split_test = np.load('test_idx_ek6.npy')
    elif opt.dataset_choose == 'rt':
        split_train = np.load('train_idx_rt.npy')
        split_test = np.load('test_idx_rt.npy')
        # reshape to (1, N)
        # split_train = np.expand_dims(split_train, axis=0)
        # split_test = np.expand_dims(split_test, axis=0)
        print(split_train.shape,split_test.shape)
    if subset == 'training':
        index = split_train[opt.split-1]
    elif subset == 'validation':
        index = split_test[opt.split-1]
    video_names = []
    annotations = []
    df = pd.read_csv(annotation_path)
    for i in list(index):
        video_names.append(df.loc[i-1,'Video Name and Directory'])
        annotations.append({'label':df.loc[i-1,'Video Name and Directory'].split('/')[0]})
    return video_names,annotations


def make_dataset(opt,video_root_path, annotation_path,  subset, fps=30,dataset_choose='ve8'):
    video_names, annotations = get_video_class(opt,subset,annotation_path)
    if dataset_choose == 've8':
        class_to_idx = {'Anger':0,'Anticipation':1,'Disgust':2,'Fear':3,'Joy':4,'Sadness':5,'Surprise':6,'Trust':7}
    elif dataset_choose == 'ek6':
        class_to_idx = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Joy': 3, 'Sadness': 4, 'Surprise': 5}
    elif dataset_choose == 'mafw':
        class_to_idx = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5,'Surprise':6,'Contempt':7,'Anxiety':8,'Helplessness':9,'Disappointment':10}
    elif dataset_choose == 'rt':
        class_to_idx = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    for i in range(len(video_names)):
        # t = time.time()
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_names)))
        video_path = os.path.join(video_root_path, video_names[i])
        raw_path = os.path.join(opt.video_raw_path,video_names[i])
        assert os.path.exists(video_path), video_path

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print(video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'raw': raw_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
        }
        assert len(annotations) != 0
        sample['label'] = class_to_idx[annotations[i]['label']]

        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps

        sample['frame_indices'] = list(range(1, n_frames + 1, step))
        dataset.append(sample)
        # print('data_prepare_time=', time.time() - t)
    return dataset, idx_to_class

# def make_neural_dataset(opt,video_root_path, neural_response,fps=30):
#     video_names = [str(i+1).zfill(4) for i in range(2185)]
#     video_names.remove('0859')
#     video_names.remove('0866')
#     video_names.remove('1673')
#     video_names.remove('2184')
#     if opt.random_choice == True:
#         video_names = random.sample(video_names,opt.video_num)
#     print(video_names)
#     print(len(video_names))
#     video_order = np.array(h5py.File('Neural_data/video_order_rt.mat')['video_order'])
#     dataset = []
#     for i in range(len(video_names)):
#         if i % 100 == 0:
#             print("Dataset loading [{}/{}]".format(i, len(video_names)))
#         video_path = os.path.join(video_root_path, video_names[i])
#         raw_path = os.path.join('iScience--raw/iScience',video_names[i])
#         assert os.path.exists(video_path), video_path

#         n_frames_file_path = os.path.join(video_path, 'n_frames')
#         n_frames = int(load_value_file(n_frames_file_path))
#         if n_frames <= 0:
#             print(video_path)
#             continue

#         begin_t = 1
#         end_t = n_frames
#         sample = {
#             'video': video_path,
#             'raw': raw_path,
#             'segment': [begin_t, end_t],
#             'n_frames': n_frames,
#             'video_id': video_names[i],
#         }


#         ORIGINAL_FPS = 30
#         step = ORIGINAL_FPS // fps
#         neural_index = np.where(video_order==int(video_names[i]))[1]
#         sample['frame_indices'] = list(range(1, n_frames + 1, step))
#         if opt.data_use == 'mean':
#             sample['neural']={'Subject1':neural_response[0][neural_index].squeeze(0),'Subject2':neural_response[1][neural_index].squeeze(0),'Subject3':neural_response[2][neural_index].squeeze(0),'Subject4':neural_response[3][neural_index].squeeze(0),'Subject5':neural_response[4][neural_index].squeeze(0)}
#         else:
#             sample['neural'] = {opt.data_use:neural_response[neural_index].squeeze(0)}
#         dataset.append(sample)
#     return dataset
def make_neural_dataset(opt,video_root_path, neural_response,fps=30):
    video_ids = [i+1 for i in range(992)]
    # remove video_names ranged from 1-31, 125-155, 249-279, 373-403, 497-527, 621-651, 745-775, 869-899
    # for i in range(1,32):
    #     video_ids.remove(i)
    # for i in range(125,156):
    #     video_ids.remove(i)
    # for i in range(249,280):
    #     video_ids.remove(i)
    # for i in range(373,404):
    #     video_ids.remove(i)
    # for i in range(497,528):
    #     video_ids.remove(i)
    # for i in range(621,652):
    #     video_ids.remove(i)
    # for i in range(745,776):
    #     video_ids.remove(i)
    # for i in range(869,900):
    #     video_ids.remove(i)
    # if opt.random_choice == True:
    #     video_ids = random.sample(video_ids,opt.video_num)
    # print(video_ids)
    print(len(video_ids))
    class_to_idx = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}

    import csv

    try:
        video_id_to_name = {}
        df = pd.read_csv("/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv")
        with open("/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                video_id_to_name[int(row[0])] = row[1]
    except FileNotFoundError:
        print(f"Error: Input file not found: '/mnt/d/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv'")
        return

    video_order = np.array(h5py.File(f'Neural_data/video_order_rt_{opt.split}.mat')['video_order'])
    # reshape to (1, N)
    video_order = video_order.reshape(1, -1)
    print(video_order.shape)
    dataset = []
    for i in video_order[0]:
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_order[0])))
        
        video_path = os.path.join(video_root_path, video_id_to_name[i].split('/')[1])
        raw_path = os.path.join('nRT--raw/nRT', video_id_to_name[i].split('/')[1])
        assert os.path.exists(video_path), video_path

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print(video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'raw': raw_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_id_to_name[i].split('/')[1],
        }
        sample['label'] = class_to_idx[df.loc[i-1,'Video Name and Directory'].split('/')[0]]


        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps
        neural_index = np.where(video_order==int(i))[1][0]
        # print("neural_index", neural_index)
        sample['frame_indices'] = list(range(1, n_frames + 1, step))
        if opt.data_use == 'mean':
            sample['neural']={'sub-01':neural_response[0][neural_index].squeeze(0),'sub-02':neural_response[1][neural_index].squeeze(0),'sub-03':neural_response[2][neural_index].squeeze(0),'sub-04':neural_response[3][neural_index].squeeze(0),'sub-05':neural_response[4][neural_index].squeeze(0)}
        else:
            # print("neural_response[0][neural_index].shape", neural_response[0][neural_index].shape)
            sample['neural'] = {opt.data_use:neural_response[0][neural_index]}
        dataset.append(sample)
    return dataset

def make_behavior_dataset(video_root_path, behavior_response,fps=30):
    video_names = [str(i+1).zfill(4) for i in range(2185)]
    video_names.remove('0859')
    video_names.remove('0866')
    video_names.remove('1673')
    video_names.remove('2184')

    video_order = np.array(h5py.File('Neural_data/video_order.mat')['video_order'])
    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_names)))
        video_path = os.path.join(video_root_path, video_names[i])
        raw_path = os.path.join('iScience--raw/iScience', video_names[i])
        assert os.path.exists(video_path), video_path

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print(video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'raw': raw_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i],
        }


        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps
        neural_index = np.where(video_order==int(video_names[i]))[1]
        sample['frame_indices'] = list(range(1, n_frames + 1, step))
        sample['behavior'] = behavior_response[neural_index].squeeze(0)
        dataset.append(sample)
    return dataset