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


def get_dataloader(args):
    # Create the 3 dataloader object
    # TODO: magari creare diversi transorm per train, query e gallery, che magari differiscono
    transform = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # TODO: controllare se l'opzione pin_memory funziona anche senza GPU
    train_dl = DataLoader(PeopleDataset(args.train_path, transform), shuffle=True,
                          batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True)
    query_dl = DataLoader(PeopleDataset(args.query_path, transform, use_name=True), shuffle=True,
                          batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True)
    gallery_dl = DataLoader(PeopleDataset(args.gallery_path, transform, use_name=True), shuffle=True,
                            batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True)

    return train_dl, query_dl, gallery_dl


class PeopleDataset(Dataset):
    def __init__(self, path, transform=None, use_name=False):
        # use_name indica se utilizzare il nome della cartella come id,
        # oppure usare l'id numerico generato dal programma, di default usata quello numerico
        self.basepath = path
        self.transform = transform
        self.use_name = use_name

        self.dataset = self._create_dataset()

    def _create_dataset(self):
        # Loads the image list and filter it
        people_list = sorted(os.listdir(self.basepath))

        if len(people_list) == 0:
            raise IOError("The dataset folder is empty")

        people_dirs = filter(os.path.isdir, [os.path.join(self.basepath, i) for i in people_list])

        dataset = []
        for (num, persondir) in enumerate(people_dirs):
            imglist = os.listdir(persondir)
            imglist = filter(lambda i: i.endswith('.jpg'), imglist)
            person_name = os.path.basename(persondir)

            for imgname in imglist:
                imgpath = os.path.join(persondir, imgname)
                dataset.append((imgpath, num, person_name))

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, pname = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.use_name:
            return img, pname
        else:
            return img, pid

