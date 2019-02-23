from fastai.vision import *
from bw_detector import detect_color_image
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
a = parser.parse_args()
torch.cuda.set_device(a.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

# make sure you're running save version. fastai is beyond picky wrt version compatibility
from fastai.utils.collect_env import *

path = Path('train')

def get_CrCb(fn): return Image(open_image(fn, convert_mode="YCbCr").data[1:,...])
def get_Y(fn): return Image(open_image(fn, convert_mode="YCbCr").data[:1,...].expand(3,-1,-1))

class MySegmentationProcessor(PreProcessor):
    "`PreProcessor` that stores the classes for segmentation."
    def __init__(self, ds:ItemList): pass
    def process(self, ds:ItemList):  pass

class MySegmentationLabelList(SegmentationLabelList):
    "`ItemList` for segmentation masks."
    _processor=MySegmentationProcessor
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func = MSELossFlat(axis=1)

    def open(self, fn): return get_CrCb(fn)
    def analyze_pred(self, pred): return pred
    def reconstruct(self, t:Tensor): return ImageSegment(t)

class MySegmentationItemList(SegmentationItemList):
    _label_cls,_square_show_res = MySegmentationLabelList,False
    def open(self,fn): return get_Y(fn)
    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()):
            _y    = (x.px[:1,...].cpu().numpy()*255).astype(np.uint8)
            _CbCr = (y.px[:,...].cpu().numpy()*255).astype(np.uint8)
            yCbCr = np.concatenate([_y, _CbCr], axis=0)
            yCbCr = np.transpose(yCbCr, (1,2,0))
            ax.imshow(PIL.Image.fromarray(yCbCr, mode='YCbCr'))
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()
    def show_xyzs(self, xs, ys, zs, imgsize:int=8, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = 'Ground truth/Grayscale Input/Colorized predictions'
        axs = subplots(len(xs), 3, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            _y    = (x.px[:1,...].cpu().numpy()*255).astype(np.uint8)
            _CbCr = (y.px[:,...].cpu().numpy()*255).astype(np.uint8)
            yCbCr = np.transpose(np.concatenate([_y, _CbCr], axis=0), (1,2,0))
            axs[i,0].imshow(PIL.Image.fromarray(yCbCr, mode='YCbCr'))
            yCbCr = np.transpose(np.concatenate([_y,_y,_y], axis=0), (1,2,0))
            axs[i,1].imshow(PIL.Image.fromarray(yCbCr, mode='RGB'))
            _CbCr = (z.px[:,...].cpu().numpy()*255).astype(np.uint8)
            yCbCr = np.transpose(np.concatenate([_y, _CbCr], axis=0), (1,2,0))
            axs[i,2].imshow(PIL.Image.fromarray(yCbCr, mode='YCbCr'))
        for ax in axs.flatten()[len(xs):]: ax.axis('off')

src = MySegmentationItemList.from_folder(path).random_split_by_pct(0.2).label_from_func(lambda x:x)

size = 224
bs = 22
data = (src.transform(get_transforms(max_lighting=None,xtra_tfms=rand_resize_crop(size)), size=size, tfm_y=True,)
        .databunch(bs=bs)
        .normalize())

data.c=2

learn = unet_learner(data, models.resnet34, blur_final=False, y_range=(0,1.)).distributed(a.local_rank)
learn.unfreeze()
learn.load('w2')
#learn = learn.to_fp16()
#lr_find(learn)
#learn.recorder.plot()

learn.fit_one_cycle(3,max_lr=1e-5)

learn.save('w2-1', return_path=True)
