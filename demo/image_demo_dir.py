from argparse import ArgumentParser
import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import random

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    demo_im_names = os.listdir(args.img)
    random.shuffle(demo_im_names)
    for im_name in demo_im_names:
        if 'png' in im_name or 'jpg' in im_name:
            full_name = os.path.join(args.img, im_name)
            result = inference_segmentor(model, full_name)
            # show the results
            pl=[[220, 220, 220],[17, 142, 35], [152, 251, 152], [0, 60, 100], [70, 130, 180], [220, 20, 20]]
            show_result_pyplot(model, full_name, result,pl)


if __name__ == '__main__':
    main()
