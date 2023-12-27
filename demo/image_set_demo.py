from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, return_result
from mmseg.core.evaluation import get_palette

import cv2 as cv
import glob
import os
import numpy as np
import re


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    if not os.path.exists(f'{args.img}/../segformer'):
        os.makedirs(f'{args.img}/../segformer')

    img_list = glob.glob(f'{args.img}/*.jpg', recursive=True)
    img_list = np.unique(np.array(img_list))
    img_list = sorted(img_list, key=lambda s: int(re.search(r'\d+', s).group()))

    for index, img in enumerate(img_list):
        print(f'{index} / {len(img_list)}')

        # test a single image
        result = inference_segmentor(model, img)

        result_img = result[0].permute(1, 2, 0).cpu().numpy() * 255

        result_img[:, :, 2] += result_img[:, :, 0]
        result_img[:, :, 1] += result_img[:, :, 0]

        result_img = np.clip(result_img, 0, 255)

        # show the results
        #result_img = return_result(model, img, result, get_palette(args.palette))

        cv.imwrite(f'{args.img}/../segformer/{index}.png', result_img)


if __name__ == '__main__':
    main()
