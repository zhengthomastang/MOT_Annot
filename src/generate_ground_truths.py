import os
import sys
import csv
import cv2
import argparse


def main(args):
    bbox_area_limit = 1000
    occlude_iou_limit = 0.50

    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        print('Camera: %s' % seq_name)

        path_data = os.path.join(args.data_root, seq_name)
        mtsc_baseline_path = os.path.join(path_data, 'mtsc/mtsc_tnt_mask_rcnn.txt')
        annotation_path = os.path.join(path_data, 'annotation.txt')
        gt_path = os.path.join(path_data, 'gt/gt.txt')
        path_roi = os.path.join(path_data, 'roi.jpg')

        mtsc_baseline_dict = {}
        match_dict = {}
        annotation_dict = {}
        gt_dict = {}
        roi = cv2.imread(path_roi, cv2.IMREAD_GRAYSCALE)

        # input
        with open(mtsc_baseline_path) as mtsc_baseline_file:
            mtsc_baseline_reader = csv.reader(mtsc_baseline_file, delimiter=',')
            for row in mtsc_baseline_reader:
                if int(row[1]) not in mtsc_baseline_dict:
                    mtsc_baseline_dict[int(row[1])] = {}
                if int(row[4]) * int(row[5]) > bbox_area_limit:
                    mtsc_baseline_dict[int(row[1])][int(row[0])] = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))

        # remove operation
        with open(annotation_path) as annotation_file:
            line_idx = 1
            annotation_reader = csv.reader(annotation_file, delimiter=',')
            for row in annotation_reader:
                if row[0] not in ('assign', 'insert', 'remove'):
                    print('ERROR: Unknown operation type %s at line %d' % (row[0], line_idx))
                if row[0] == 'remove':
                    if len(row) != 3:
                        print('ERROR: Wrong annotation at line %d' % line_idx)
                    if int(row[2]) not in gt_dict:
                        print('ERROR: Not assigned local ID %d at line %d' % (int(row[2]), line_idx))
                    frm_idx_min = 0
                    frm_idx_max = sys.maxsize
                    range_sign_idx = row[1].find('-')
                    if range_sign_idx == -1:
                        frm_idx_min = int(row[1])
                        frm_idx_max = int(row[1])
                    elif range_sign_idx == 0:
                        frm_idx_max = int(row[1][1:])
                    elif range_sign_idx == len(row[1]) - 1:
                        frm_idx_min = int(row[1][:-1])
                    else:
                        range_toks = row[1].split('-')
                        if len(range_toks) != 2:
                            print('ERROR: Wrong removal range at line %d' % line_idx)
                        frm_idx_min = int(range_toks[0])
                        frm_idx_max = int(range_toks[1])
                    for frm_idx in mtsc_baseline_dict[int(row[2])].keys():
                        if frm_idx >= frm_idx_min and frm_idx <= frm_idx_max:
                            del mtsc_baseline_dict[int(row[2])][frm_idx]
                line_idx += 1

        # assign operation
        with open(annotation_path) as annotation_file:
            line_idx = 1
            annotation_reader = csv.reader(annotation_file, delimiter=',')
            for row in annotation_reader:
                if row[0] == 'assign':
                    if len(row) != 5:
                        print('ERROR: Wrong annotation at line %d' % line_idx)
                    if int(row[1]) in match_dict:
                        print('ERROR: Duplicate assignment of local ID %d at line %d' % (int(row[1]), line_idx))
                    if int(row[1]) not in mtsc_baseline_dict:
                        print('ERROR: Unknown local ID %d at line %d' % (int(row[1]), line_idx))
                    match_dict[int(row[1])] = int(row[2])
                    if int(row[2]) not in annotation_dict:
                        annotation_dict[int(row[2])] = {}
                    for frm_idx in mtsc_baseline_dict[int(row[1])].keys():
                        annotation_dict[int(row[2])][frm_idx] = mtsc_baseline_dict[int(row[1])][frm_idx]
                line_idx += 1

        # insert operation
        with open(annotation_path) as annotation_file:
            line_idx = 1
            annotation_reader = csv.reader(annotation_file, delimiter=',')
            for row in annotation_reader:
                if row[0] == 'insert':
                    if len(row) != 7:
                        print('ERROR: Wrong annotation at line %d' % line_idx)
                    if int(row[2]) not in match_dict:
                        print('ERROR: Not assigned local ID %d at line %d' % (int(row[2]), line_idx))
                    global_id = match_dict[int(row[2])]
                    annotation_dict[global_id][int(row[1])] = (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
                line_idx += 1

        # linear interpolation
        for global_id in annotation_dict.keys():
            frm_indices = annotation_dict[global_id].keys()
            frm_indices.sort()
            for i in range(1, len(frm_indices)):
                frm_diff = frm_indices[i] - frm_indices[i-1]
                if frm_diff > 1:
                    bbox_prev = annotation_dict[global_id][frm_indices[i-1]]
                    bbox_next = annotation_dict[global_id][frm_indices[i]]
                    x_step = (bbox_next[0] - bbox_prev[0]) / float(frm_diff)
                    y_step = (bbox_next[1] - bbox_prev[1]) / float(frm_diff)
                    w_step = (bbox_next[2] - bbox_prev[2]) / float(frm_diff)
                    h_step = (bbox_next[3] - bbox_prev[3]) / float(frm_diff)
                    for j in range(1, frm_diff):
                        frm_idx_curr = frm_indices[i-1] + j
                        x_curr = int(round(bbox_prev[0] + x_step * j))
                        y_curr = int(round(bbox_prev[1] + y_step * j))
                        w_curr = int(round(bbox_prev[2] + w_step * j))
                        h_curr = int(round(bbox_prev[3] + h_step * j))
                        if w_curr * h_curr > bbox_area_limit:
                            annotation_dict[global_id][frm_idx_curr] = (x_curr, y_curr, w_curr, h_curr)
        
        # change the dictionary levels
        for global_id in annotation_dict.keys():
            for frm_index in annotation_dict[global_id].keys():
                if frm_index not in gt_dict:
                    gt_dict[frm_index] = {}
                if global_id in gt_dict[frm_index]:
                    print('ERROR: Duplicate global ID at frame %d' % frm_index)
                gt_dict[frm_index][global_id] = annotation_dict[global_id][frm_index]

        # handle occlusion
        occlude_id_list = []
        for frm_index in gt_dict.keys():
            global_ids = gt_dict[frm_index].keys()
            for i in range(len(global_ids) - 1):
                for j in range(i + 1, len(global_ids)):
                    bbox_a = gt_new_dict[frm_index][global_ids[i]]
                    bbox_b = gt_new_dict[frm_index][global_ids[j]]
                    x_inter_min = max(bbox_a[0], bbox_b[0])
                    y_inter_min = max(bbox_a[1], bbox_b[1])
                    x_inter_max = min(bbox_a[0] + bbox_a[2], 
                                      bbox_b[0] + bbox_b[2])
                    y_inter_max = min(bbox_a[1] + bbox_a[3], 
                                      bbox_b[1] + bbox_b[3])
                    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)
                    bbox_a_area = bbox_a[2] * bbox_a[3]
                    bbox_b_area = bbox_b[2] * bbox_b[3]
                    if inter_area / float(bbox_a_area) > occlude_iou_limit:
                        if bbox_a[1] + bbox_a[3] < bbox_b[1] + bbox_b[3]:
                            occlude_id_list.append((frm_index, global_ids[i]))
                    if inter_area / float(bbox_b_area) > occlude_iou_limit:
                        if bbox_a[1] + bbox_a[3] > bbox_b[1] + bbox_b[3]:
                            occlude_id_list.append((frm_index, global_ids[j]))
        for occlude_id in occlude_id_list:
            if occlude_id[1] in gt_dict[occlude_id[0]]:
                del gt_dict[occlude_id[0]][occlude_id[1]]

        # output
        gt_file = open(gt_path, 'w')
        for frm_index in sorted(gt_dict.keys()):
            for global_id in sorted(gt_dict[frm_index].keys()):
                bbox = gt_dict[frm_index][global_id]
                if not os.path.isfile(path_roi):
                    x_min = bbox[0]
                    y_min = bbox[1]
                    x_max = bbox[0] + bbox[2] - 1
                    y_max = bbox[1] + bbox[3] - 1
                    if roi[y_min, x_min] <= 0 or roi[y_max, x_min] <= 0 or roi[y_min, x_max] <= 0 or roi[y_max, x_max] <= 0:
                        continue
                gt_file.write('%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n' % (frm_index, global_id, bbox[0], bbox[1], bbox[2], bbox[3]))
        gt_file.close()


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Generate ground truths')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
