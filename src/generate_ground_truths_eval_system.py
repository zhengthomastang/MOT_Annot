import os
import sys
import csv
import argparse


def main(args):
    gt_dict = {}
    output_path = os.path.join(args.data_root, 'labels.txt')

    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        print('Camera: %s' % seq_name)

        gt_path = os.path.join(args.data_root, seq_name, 'gt/gt.txt')

        with open(gt_path) as gt_file:
            gt_reader = csv.reader(gt_file, delimiter=',')
            for row in gt_reader:
                if seq_name not in gt_dict:
                    gt_dict[seq_name] = {}
                if int(row[0]) not in gt_dict[seq_name]:
                    gt_dict[seq_name][int(row[0])] = {}
                gt_dict[seq_name][int(row[0])][int(row[1])] = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))

    output_file = open(output_path, 'w')
    for seq_name in sorted(gt_dict.keys()):
        for frm_idx in sorted(gt_dict[seq_name].keys()):
            for global_id in sorted(gt_dict[seq_name][frm_idx].keys()):
                bbox = gt_dict[seq_name][frm_idx][global_id]
                output_file.write('%d %d %d %d %d %d %d -1 -1\n' % (int(seq_name[1:]), global_id, frm_idx, bbox[0], bbox[1], bbox[2], bbox[3]))
    output_file.close()


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Generate labels of ground truths for the evaluation system')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
