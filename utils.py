
import numpy as np
import math
import random
import torch
import skimage.morphology as morph
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy import ndimage
from skimage import measure


def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((6,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value/batch_size for value in results]


def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1-pred) * (1-target))  # true negatives
    fp = np.sum(pred * (1-target))  # false postives
    fn = np.sum((1-pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn/(tn+fp+1e-10)) / 2
    iou = tp / (tp+fp+fn+1e-10)

    return [acc, iou, recall, precision, F1, performance]


def nuclei_accuracy_object_level(pred, gt):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    pred_labeled = measure.label(pred)
    gt_labeled = measure.label(gt)
    Ns = len(np.unique(pred_labeled)) - 1    # number of detected objects
    Ng = len(np.unique(gt_labeled)) - 1   # number of ground truth objects

    TP = 0.0   # true positive
    FN = 0.0   # false negative
    dice = 0.0
    haus = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(1, Ng+1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute hausdorff distance
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        # pred_copy[pred_i > 0] = 0
        pred_labeled[pred_i>0] = 0  # remove the used nucleus

    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    dice /= count
    iou /= count
    haus /= count

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    return recall, precision, F1, dice, iou, haus, AJI


def gland_accuracy_object_level(pred, gt):
    """ Compute the object-level hausdorff distance between predicted  and
    groundtruth """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    gt_labeled = morph.remove_small_objects(gt_labeled, 3)   # remove 1 or 2 pixel noise in the image
    gt_labeled = morph.label(gt_labeled, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # show_figures((pred_labeled, gt_labeled))

    # --- compute F1 --- #
    TP = 0.0  # true positive
    FP = 0.0  # false positive
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((img_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            min_haus = 1e5
            for j in range(1, Ns + 1):
                pred_j = np.where(pred_labeled == j, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_i)
            gt_ind = np.argwhere(gt_i)
            haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            min_haus = 1e5
            for i in range(1, Ng + 1):
                gt_i = np.where(gt_labeled == i, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_j)
            gt_ind = np.argwhere(gt_j)
            haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        hausdorff_s += sigma_j * haus_j

    return recall, precision, F1, (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2


def split_forward(model, input, size, overlap, outchannel=3):
    '''
    split the input image for forward process
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

