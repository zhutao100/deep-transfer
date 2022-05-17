import pathlib
import itertools
import torch
from log_utils import get_logger
from im_utils import load_img
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')


class ContentStylePairDataset(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.synthesis = args.synthesis
        self.contentSize = args.contentSize
        self.styleSize = args.styleSize

        style_files = self.__get_files__(args.style)
        if not args.synthesis:
            content_files = self.__get_files__(args.content)
            self.pairs_fn = list(itertools.product([str(x) for x in content_files], [str(x) for x in style_files]))
        else:
            self.pairs_fn = [('texture', str(x)) for x in style_files]
        log.debug(f'Added pairs "{self.pairs_fn}" to the dataset')

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
        return len(self.pairs_fn)

    def __getitem__(self, idx):
        pair = self.pairs_fn[idx]

        style = load_img(pair[1], self.styleSize)

        if self.synthesis:
            c_c, h_c, w_c = style.size()
            content = torch.zeros((c_c, h_c, w_c)).uniform_()
        else:
            content = load_img(pair[0], self.contentSize)

        return {'content': content, 'contentPath': pair[0], 'style': style, 'stylePath': pair[1]}
