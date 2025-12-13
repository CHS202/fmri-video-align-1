from datasets.ve8 import VE8Dataset,NeuralDataset,BehaviorDataset
from torch.utils.data import DataLoader

def get_ve8(opt, subset, transforms):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(opt,opt.video_path,
                      opt.annotation_path,
                      subset,
                      opt.fps,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      opt.dataset_choose)

def get_neural(opt, transforms,neural_response):
    spatial_transform, temporal_transform = transforms
    return  NeuralDataset(opt,opt.neural_video_path,neural_response,opt.fps,spatial_transform,temporal_transform)

def get_behavior(opt, transforms,behavior_response):
    spatial_transform, temporal_transform = transforms
    return  BehaviorDataset(opt.neural_video_path,behavior_response,opt.fps,spatial_transform,temporal_transform)


def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'training', transforms)

def get_neural_set(opt, spatial_transform, temporal_transform,neural_response):
    transforms = [spatial_transform, temporal_transform]
    return get_neural(opt,transforms,neural_response)

def get_behavior_set(opt, spatial_transform, temporal_transform,behavior_response):
    transforms = [spatial_transform, temporal_transform]
    return get_behavior(opt,transforms,behavior_response)

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'validation', transforms)


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'validation', transforms)


def get_data_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=False, # True
        drop_last=opt.dl
    )

def get_neural_loader(opt, dataset, shuffle):
    batch_size = opt.batch_size_neural
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=False,
        drop_last=opt.dl
    )
def get_behavior_loader(opt, dataset, shuffle):
    batch_size = opt.batch_size_behavior
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl
    )