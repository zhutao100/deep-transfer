import pathlib
import torch
from im_utils import load_img
from log_utils import get_logger
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')


class ContentStyleTripletDataset(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.synthesis = args.synthesis
        self.contentSize = args.contentSize
        self.styleSize = args.styleSize

        if args.synthesis:
            self.triplets_fn = [('texture', args.style0, args.style1)]
        else:
            content_files = self.__get_files__(args.content)
            self.triplets_fn = [(str(x), args.style0, args.style1) for x in content_files]

    def __get_files__(self, file_str):
        file_path = pathlib.Path(file_str)
        if file_path.is_dir():
            files = [x for x in file_path.iterdir() if x.is_file() and x.suffix.lower() in supported_img_formats]
        elif file_path.is_file():
            files = [file_path]
        else:
            raise RuntimeError("Content files are not accessible.")
        return files

    def __len__(self):
        return len(self.triplets_fn)

    def __getitem__(self, idx):
        triplet = self.triplets_fn[idx]

        style0 = load_img(triplet[1], self.styleSize)
        style1 = load_img(triplet[2], self.styleSize)

        if self.synthesis:
            c_c, h_c, w_c = style0.size()
            content = torch.zeros((c_c, h_c, w_c)).uniform_()
        else:
            content = load_img(triplet[0], self.contentSize)

        return {'content': content, 'contentPath': triplet[0],
                'style0': style0, 'style0Path': triplet[1],
                'style1': style1, 'style1Path': triplet[2]}
