from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PillarDataset(CustomDataset):

    CLASSES = ('background', 'big', 'small')

    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(PillarDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
