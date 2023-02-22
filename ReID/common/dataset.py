from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

import os
from PIL import Image


def read_image(img_path):
    # Given an image path, tries to load it
    if not os.path.exists(img_path):
        raise IOError(f"{img_path} does not exist.")

    img = Image.open(img_path).convert('RGB')

    return img

def get_dataloader(args, dataset:str='motsynth'):
    target_res = (args.height, args.width)
    return get_dataloader_params(args.dataset_path, target_res, args.batch_size, args.workers, dataset=dataset)

def get_dataloader_params(dataset_path:str, target_res:tuple, batch_size:int=16, workers:int=4, max_ids_train:int=-1, max_ids_test:int=-1, normalize:bool=True, dataset:str='motsynth'):
    # Create the 3 dataloader object
    # TODO: magari creare diversi transorm per train, query e gallery, che magari differiscono
    
    if normalize:
        transform = T.Compose([
            T.Resize(target_res),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.Resize(target_res),
            T.ToTensor(),
        ])

    train_path = os.path.join(dataset_path, "train")
    queries_path = os.path.join(dataset_path, "queries")
    gallery_path = os.path.join(dataset_path, "gallery")

    if dataset == 'motsynth':
        train_dl = DataLoader(MOTSynthDataset(train_path, transform, max_ids_train), shuffle=True,
                            batch_size=batch_size, num_workers=workers,
                            pin_memory=True)
        query_dl = DataLoader(MOTSynthDataset(queries_path, transform, max_ids_test), shuffle=True,
                            batch_size=batch_size, num_workers=workers,
                            pin_memory=True)
        gallery_dl = DataLoader(MOTSynthDataset(gallery_path, transform, max_ids_test), shuffle=True,
                                batch_size=batch_size, num_workers=workers,
                                pin_memory=True)
    elif dataset == 'market1501':
        train_dl = DataLoader(Market_1501_dataset(train_path, transform, max_ids_train), shuffle=True,
                            batch_size=batch_size, num_workers=workers,
                            pin_memory=True)
        query_dl = DataLoader(Market_1501_dataset(queries_path, transform, max_ids_test), shuffle=True,
                            batch_size=batch_size, num_workers=workers,
                            pin_memory=True)
        gallery_dl = DataLoader(Market_1501_dataset(gallery_path, transform, max_ids_test), shuffle=True,
                                batch_size=batch_size, num_workers=workers,
                                pin_memory=True)
    elif dataset == 'mars':
        train_dl = None
        #MARS dataset has been processed in the same way of MOTSynth.
        query_dl = DataLoader(MOTSynthDataset(queries_path, transform, max_ids_test), shuffle=True,
                            batch_size=batch_size, num_workers=workers,
                            pin_memory=True)
        gallery_dl = DataLoader(MOTSynthDataset(gallery_path, transform, max_ids_test), shuffle=True,
                                batch_size=batch_size, num_workers=workers,
                                pin_memory=True)
    else:
        raise Exception("invalid dataset.")

    return train_dl, query_dl, gallery_dl


class CustomDataset(Dataset):
    def __init__(self, path, transform=None, max_ids=-1):
        self.transform = transform
        self.dataset = self._create_dataset(path, max_ids)

    def _create_dataset(self, basepath):
        """
        To be overrided with custom code.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid



class MOTSynthDataset(CustomDataset):

    def _create_dataset(self, basepath, max_ids=-1):
        # Loads the image list and filter it
        people_list = sorted(os.listdir(basepath))

        #if len(people_list) == 0:
        #    raise IOError("The dataset folder is empty")

        clean_people_list = [item for item in people_list if os.path.isdir(os.path.join(basepath, item))]
        # In our dataset there are a lot of folders, one for each identity.
        # Every folder starts with an unique integer number, followd by a _ and other characters.
        # We will use that (first) integer as unique ID for each identity.
        ids = {idx:int(item.split('_')[0]) for (idx, item) in enumerate(clean_people_list)}
        
        people_dirs = [os.path.join(basepath, i) for i in clean_people_list]

        dataset = []
        for idx, persondir in enumerate(people_dirs):
            if not os.path.isdir(persondir):
                continue
            id = ids[idx]
            imglist = os.listdir(persondir)
            imglist = filter(lambda i: i.endswith('.jpg'), imglist)
            person_name = os.path.basename(persondir)

            for imgname in imglist:
                imgpath = os.path.join(persondir, imgname)
                dataset.append((imgpath, id))
            if max_ids > 0 and idx+1 >= max_ids:
                return dataset

        return dataset


class Market_1501_dataset(CustomDataset):

    def _create_dataset(self, basepath, max_ids=-1):
        people_list = os.listdir(basepath)

        if len(people_list) == 0:
            raise IOError("The dataset folder is empty")

        dataset = []
        for idx, img in enumerate(people_list):
            if not img.endswith('.jpg'):
                continue

            id = int(img.split('_')[0])
            imgpath = os.path.join(basepath, img)
            dataset.append((imgpath, id))
        if max_ids > 0 and idx+1 >= max_ids:
                return dataset

        return dataset
