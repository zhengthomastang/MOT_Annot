import os
import sys
import shutil
import csv
import json
import cv2
import random
import argparse


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def write_json_to_file(file_path, json_dict, indent=4, sort_keys=True):
    with open(file_path, 'w') as output_file:
        json.dump(json_dict, output_file, sort_keys=sort_keys, indent=indent, ensure_ascii=False)


def main(args):
    if os.path.isdir(args.output_root):
        shutil.rmtree(args.output_root)

    iou_thld = 0.75
    vid_split = 440
    query_count_max = 4
    num_dist_example = 100

    veh_dict = {}
    scene_dict = {}
    bbox_num_total = 0
    bbox_num_max = 0

    split_path_dict = {}
    split_path_dict['train'] = os.path.join(args.data_root, 'train')
    split_path_dict['validation'] = os.path.join(args.data_root, 'validation')
    split_path_dict['test'] = os.path.join(args.data_root, 'test')

    for split in split_path_dict.keys():
        scene_list = os.listdir(split_path_dict[split])

        for scene in scene_list:
            scene_path = os.path.join(split_path_dict[split], scene)
            cam_list = os.listdir(scene_path)

            for cam_name in cam_list:
                print('Split: %s; Scene: %s; Camera: %s' % (split.upper(), scene, cam_name))

                cam_path = os.path.join(scene_path, cam_name)
                gt_path = os.path.join(cam_path, 'gt', 'gt_new_new.txt')

                with open(gt_path) as gt_file:
                    gt_reader = csv.reader(gt_file, delimiter=',')
                    for row in gt_reader:
                        vid = int(row[1])
                        frm_cnt = int(row[0])
                        x = int(row[2])
                        y = int(row[3])
                        w = int(row[4])
                        h = int(row[5])
                        if vid not in veh_dict:
                            veh_dict[vid] = {}
                        if vid not in scene_dict:
                            scene_dict[vid] = (split, scene)
                        if cam_name not in veh_dict[vid]:
                            veh_dict[vid][cam_name] = {}
                        if len(veh_dict[vid][cam_name].keys()) > 0:
                            frm_cnt_prev = max(veh_dict[vid][cam_name].keys())
                            bbox_prev = veh_dict[vid][cam_name][frm_cnt_prev]
                            xmin_prev = bbox_prev[0]
                            xmax_prev = bbox_prev[0] + bbox_prev[2] - 1
                            ymin_prev = bbox_prev[1]
                            ymax_prev = bbox_prev[1] + bbox_prev[3] - 1
                            xmin_curr = x
                            xmax_curr = x + w - 1
                            ymin_curr = y
                            ymax_curr = y + h - 1
                            xmin_overlap = max(xmin_prev, xmin_curr)
                            xmax_overlap = min(xmax_prev, xmax_curr)
                            ymin_overlap = max(ymin_prev, ymin_curr)
                            ymax_overlap = min(ymax_prev, ymax_curr)
                            area_overlap = 0
                            if xmax_overlap >= xmin_overlap and ymax_overlap >= ymin_overlap:
                                area_overlap = (xmax_overlap - xmin_overlap + 1) * (ymax_overlap - ymin_overlap + 1)
                            iou = float(area_overlap) / ((w * h) + (bbox_prev[2] * bbox_prev[3]) - area_overlap)
                            if iou > iou_thld:
                                continue
                        veh_dict[vid][cam_name][frm_cnt] = (x, y, w, h)
                        bbox_num_total += 1
                        if len(veh_dict[vid][cam_name].keys()) > bbox_num_max:
                            bbox_num_max = len(veh_dict[vid][cam_name].keys())

    print('Total number of vehicles: %d' % len(veh_dict.keys()))
    print('Total number of bounding boxes: %d' % bbox_num_total)
    print('Maximum number of bounding boxes: %d' % bbox_num_max)

    split_dict = {'train': [], 'test': [], 'query': []}
    train_tracks = []
    test_tracks = []
    gt_map = {}

    for vid in veh_dict.keys():
        if vid <= vid_split:
            for cam_name in veh_dict[vid].keys():
                train_tracks.append([])
                for frm_cnt in veh_dict[vid][cam_name].keys():
                    split_dict['train'].append((vid, cam_name, frm_cnt))
                    train_tracks[-1].append((vid, cam_name, frm_cnt))
        else:
            query_count = random.randint(1, query_count_max)
            cam_name_query = random.choice(veh_dict[vid].keys())
            while len(veh_dict[vid][cam_name_query].keys()) < query_count:
                cam_name_query = random.choice(veh_dict[vid].keys())
            frm_cnt_queries = set()
            frm_cnt_query = random.choice(veh_dict[vid][cam_name_query].keys())
            for i in range(query_count):
                while frm_cnt_query in frm_cnt_queries:
                    frm_cnt_query = random.choice(veh_dict[vid][cam_name_query].keys())
                split_dict['query'].append((vid, cam_name_query, frm_cnt_query))
                frm_cnt_queries.add(frm_cnt_query)
            for frm_cnt_query in frm_cnt_queries:
                gt_map[(vid, cam_name_query, frm_cnt_query)] = []
            for cam_name in veh_dict[vid].keys():
                if cam_name == cam_name_query:
                    continue
                test_tracks.append([])
                for frm_cnt in veh_dict[vid][cam_name].keys():
                    split_dict['test'].append((vid, cam_name, frm_cnt))
                    test_tracks[-1].append((vid, cam_name, frm_cnt))
                    for frm_cnt_query in frm_cnt_queries:
                        gt_map[(vid, cam_name_query, frm_cnt_query)].append((vid, cam_name, frm_cnt))

    for split in split_dict.keys():
        random.shuffle(split_dict[split])
    random.shuffle(train_tracks)
    random.shuffle(test_tracks)

    print('Total number of output images: %d' % len(split_dict['train'] + split_dict['test'] + split_dict['query']))
    print('Total number of training images: %d' % len(split_dict['train']))
    print('Total number of testing images: %d' % len(split_dict['test']))
    print('Total number of query images: %d' % len(split_dict['query']))
    print('Total number of training tracks: %d' % len(train_tracks))
    print('Total number of testing tracks: %d' % len(test_tracks))

    idx2key_map_path = os.path.join(args.output_root, 'idx2key_map.json')
    key2idx_map_path = os.path.join(args.output_root, 'key2idx_map.json')
    check_and_create(args.output_root)
    image_path_dict = {}
    name_path_dict = {}
    label_path_dict = {}
    label_gt_path_dict = {}
    for split in split_dict.keys():
        image_path_dict[split] = os.path.join(args.output_root, 'image_%s' % split)
        check_and_create(image_path_dict[split])
        name_path_dict[split] = os.path.join(args.output_root, 'name_%s.txt' % split)
        label_path_dict[split] = os.path.join(args.output_root, '%s_label.xml' % split)
        label_gt_path_dict[split] = os.path.join(args.output_root, '%s_label_gt.xml' % split)
    train_track_path = os.path.join(args.output_root, 'train_track.txt')
    test_track_path = os.path.join(args.output_root, 'test_track.txt')
    gt_index_path = os.path.join(args.output_root, 'gt_index.txt')
    dist_example_path = os.path.join(args.output_root, 'tools', 'dist_example')
    check_and_create(dist_example_path)

    idx2key_map = {}
    key2idx_map = {}
    for split in split_dict.keys():
        idx2key_map[split] = {}
        key2idx_map[split] = {}
        for i in range(len(split_dict[split])):
            instance = split_dict[split][i]
            img_key = '%04d_%s_%06d' % (instance[0], instance[1], instance[2])
            key2idx_map[split][img_key] = i + 1
            idx2key_map[split][i+1] = img_key
    write_json_to_file(idx2key_map_path, idx2key_map)
    write_json_to_file(key2idx_map_path, key2idx_map)

    for split in split_dict.keys():
        name_file = open(name_path_dict[split], 'w')
        label_file = open(label_path_dict[split], 'w')
        label_gt_file = open(label_gt_path_dict[split], 'w')
        label_file.write('<?xml version="1.0" encoding="gb2312" ?>\n')
        label_file.write('<TrainingImages Version="1.0">\n')
        label_file.write('    <Items number="%d">\n' % len(split_dict[split]))
        label_gt_file.write('<?xml version="1.0" encoding="gb2312" ?>\n')
        label_gt_file.write('<TrainingImages Version="1.0">\n')
        label_gt_file.write('    <Items number="%d">\n' % len(split_dict[split]))
        for instance in split_dict[split]:
            img_frm_path = os.path.join(args.data_root, scene_dict[instance[0]][0], scene_dict[instance[0]][1], instance[1], 'img1', '%06d.jpg' % instance[2])
            img_frm = cv2.imread(img_frm_path)
            bbox = veh_dict[instance[0]][instance[1]][instance[2]]
            img_crop = img_frm[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            img_key = '%04d_%s_%06d' % (instance[0], instance[1], instance[2])
            img_idx = key2idx_map[split][img_key]
            img_name = '%06d.jpg' % img_idx
            img_crop_path = os.path.join(image_path_dict[split], img_name)
            cv2.imwrite(img_crop_path, img_crop)
            name_file.write('%s\n' % img_name)
            label_file.write('        <Item imageName="%s" cameraID="%s" />\n' % (img_name, instance[1]))
            label_gt_file.write('        <Item imageName="%s" vehicleID="%04d" cameraID="%s" />\n' % (img_name, instance[0], instance[1]))
        name_file.close()
        label_file.write('    </Items>\n')
        label_file.write('</TrainingImages>\n')
        label_file.close()
        label_gt_file.write('    </Items>\n')
        label_gt_file.write('</TrainingImages>\n')
        label_gt_file.close()

    train_track_file = open(train_track_path, 'w')
    for track in train_tracks:
        for i in range(len(track) - 1):
            img_key = '%04d_%s_%06d' % (track[i][0], track[i][1], track[i][2])
            img_idx = key2idx_map['train'][img_key]
            train_track_file.write('%06d.jpg ' % img_idx)
        img_key = '%04d_%s_%06d' % (track[-1][0], track[-1][1], track[-1][2])
        img_idx = key2idx_map['train'][img_key]
        train_track_file.write('%06d.jpg\n' % img_idx)
    train_track_file.close()

    test_track_file = open(test_track_path, 'w')
    for track in test_tracks:
        for i in range(len(track) - 1):
            img_key = '%04d_%s_%06d' % (track[i][0], track[i][1], track[i][2])
            img_idx = key2idx_map['test'][img_key]
            test_track_file.write('%06d.jpg ' % img_idx)
        img_key = '%04d_%s_%06d' % (track[-1][0], track[-1][1], track[-1][2])
        img_idx = key2idx_map['test'][img_key]
        test_track_file.write('%06d.jpg\n' % img_idx)
    test_track_file.close()

    gt_index_file = open(gt_index_path, 'w')
    gt_num_max = 0
    for i in range(len(split_dict['query'])):
        img_idx_query = i + 1
        img_key_query = idx2key_map['query'][img_idx_query]
        split_toks = img_key_query.split('_')
        vid = int(split_toks[0])
        cam_name = split_toks[1]
        frm_cnt = int(split_toks[2])
        gt_list = gt_map[(vid, cam_name, frm_cnt)]
        for j in range(len(gt_list) - 1):
            img_key_test = '%04d_%s_%06d' % (gt_list[j][0], gt_list[j][1], gt_list[j][2])
            gt_index_file.write('%d ' % key2idx_map['test'][img_key_test])
        img_key_test = '%04d_%s_%06d' % (gt_list[-1][0], gt_list[-1][1], gt_list[-1][2])
        gt_index_file.write('%d\n' % key2idx_map['test'][img_key_test])
        if len(gt_list) > gt_num_max:
            gt_num_max = len(gt_list)
    gt_index_file.close()
    print('Maximum number of ground truths: %d' % gt_num_max)

    for i in range(len(split_dict['query'])):
        dist_example_txt_path = os.path.join(dist_example_path, '%06d.txt' % (i + 1))
        dist_example_txt_file = open(dist_example_txt_path, 'w')
        for j in range(num_dist_example):
            dist_example_txt_file.write('%d\n' % random.randint(1, len(split_dict['test'])))
        dist_example_txt_file.close()


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Generate the ReID dataset')
    parser.add_argument('--data-root', dest='data_root', default='./',
                        help='dataset root path')
    parser.add_argument('--output-root', dest='output_root', default='reid',
                        help='output root path')

    args = parser.parse_args()

    main(args)
