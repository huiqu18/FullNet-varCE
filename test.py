import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import measure
from scipy import misc
from FullNet import FullNet, FCN_pooling
import utils
import time

from options import Options
from my_transforms import get_transforms


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()
   
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])

    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    tta = opt.test['tta']

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = get_transforms(opt.transform['test'])

    # load model
    model = FullNet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                    growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                    dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                    compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # pixel_accu, recall, precision, F1, dice, iou, haus, (AJI)
    num_metrics = 8 if opt.dataset == 'MultiOrgan' else 7
    avg_results = utils.AverageMeter(num_metrics)
    all_results = dict()

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        if eval_flag:
            if opt.dataset == 'MultiOrgan':
                label_path = '{:s}/{:s}.png'.format(label_dir, name)
            else:
                label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
            label_img = misc.imread(label_path)

        input = test_transform((img,))[0].unsqueeze(0)

        print('\tComputing output probability maps...')
        prob_maps = get_probmaps(input, model, opt)
        if tta:
            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf = test_transform((img_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_vf = test_transform((img_vf,))[0].unsqueeze(0)  # vertical flip input
            input_hvf = test_transform((img_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_hf = get_probmaps(input_hf, model, opt)
            prob_maps_vf = get_probmaps(input_vf, model, opt)
            prob_maps_hvf = get_probmaps(input_hvf, model, opt)

            # re flip
            prob_maps_hf = np.flip(prob_maps_hf, 2)
            prob_maps_vf = np.flip(prob_maps_vf, 1)
            prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90 = test_transform((img_r90,))[0].unsqueeze(0)
            input_r90_hf = test_transform((img_r90_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_r90_vf = test_transform((img_r90_vf,))[0].unsqueeze(0)  # vertical flip input
            input_r90_hvf = test_transform((img_r90_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_r90 = get_probmaps(input_r90, model, opt)
            prob_maps_r90_hf = get_probmaps(input_r90_hf, model, opt)
            prob_maps_r90_vf = get_probmaps(input_r90_vf, model, opt)
            prob_maps_r90_hvf = get_probmaps(input_r90_hvf, model, opt)

            # re flip
            prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
            prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                         + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8

        pred = np.argmax(prob_maps, axis=0)  # prediction
        pred_inside = pred == 1
        pred2 = morph.remove_small_objects(pred_inside, opt.post['min_area'])  # remove small object

        if 'scale' in opt.transform['test']:
            pred2 = misc.imresize(pred2.astype(np.uint8) * 255, (ori_h, ori_w), interp='bilinear')
            pred2 = (pred2 > 127.5)

        pred_labeled = measure.label(pred2)   # connected component labeling
        pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(opt.post['radius']))

        if eval_flag:
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled>0,0), np.expand_dims(label_img>0, 0))
            pixel_accu = result[0]

            if opt.dataset == 'MultiOrgan':
                result_object = utils.nuclei_accuracy_object_level(pred_labeled, label_img)
            else:
                result_object = utils.gland_accuracy_object_level(pred_labeled, label_img)

            all_results[name] = tuple([pixel_accu, *result_object])

            # update values
            avg_results.update([pixel_accu, *result_object])

        # save image
        if save_flag:
            print('\tSaving image results...')
            misc.imsave('{:s}/{:s}_prob_inside.png'.format(prob_maps_folder, name), prob_maps[1,:,:])
            misc.imsave('{:s}/{:s}_prob_contour.png'.format(prob_maps_folder, name), prob_maps[2,:,:])
            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            misc.imsave(filename, pred_colored)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    print('=> Processed all {:d} images'.format(counter))
    if eval_flag:
        print('Average of all images:\n'
              'pixel_accu: {r[0]:.4f}\n'
              'recall: {r[1]:.4f}\n'
              'precision: {r[2]:.4f}\n'
              'F1: {r[3]:.4f}\n'
              'dice: {r[4]:.4f}\n'
              'iou: {r[5]:.4f}\n'
              'haus: {r[6]:.4f}'.format(r=avg_results.avg))
        if opt.dataset == 'MultiOrgan':
            print('AJI: {r[7]:.4f}'.format(r=avg_results.avg))

        strs = img_dir.split('/')
        header = ['pixel_acc','recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        if opt.dataset == 'MultiOrgan':
            header.append('AJI')
        save_results(header, avg_results.avg, all_results,
                     '{:s}/{:s}_test_result.txt'.format(save_dir, strs[-1]))


def get_probmaps(input, model, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    main()
