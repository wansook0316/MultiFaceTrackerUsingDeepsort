import motmetrics as mm
import numpy as np
import os
from absl import app, flags, logging
from absl.flags import FLAGS

"""
python evaluation.py \
    --gt_file_path ./resources/gt/T-ara_gt.txt \
    --pred_file_path ./resources/gt/T-ara_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/GirlsAloud_gt.txt \
    --pred_file_path ./resources/gt/GirlsAloud_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Darling_gt.txt \
    --pred_file_path ./resources/gt/Darling_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Westlife_gt.txt \
    --pred_file_path ./resources/gt/Westlife_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/BrunoMars_gt.txt \
    --pred_file_path ./resources/gt/BrunoMars_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/HelloBubble_gt.txt \
    --pred_file_path ./resources/gt/HelloBubble_pred.txt

python evaluation.py \
    --gt_file_path ./resources/gt/Apink_gt.txt \
    --pred_file_path ./resources/gt/Apink_pred.txt
"""

flags.DEFINE_string('gt_file_path', './resources/gt/T-ara_gt.txt', 'path to gt txt')
flags.DEFINE_string('pred_file_path', './resources/gt/T-ara_pred.txt', 'path to predicted txt')

def main(args):
        
    # home = os.getcwd()
    # gt_path = os.path.join(home, "resources", "gt")

    # gt_file_path = os.path.join(gt_path, "T-ara_gt.txt")
    # pred_file_path = os.path.join(gt_path, "T-ara_pred.txt")

    f = open(FLAGS.gt_file_path, "r")
    gt = []
    while True:
        line = f.readline()
        if not line: break
        a = list(map(int, line.split()))
        gt.append(a)
    gt = np.asarray(gt)
    f.close()

    f = open(FLAGS.pred_file_path, "r")
    pred = []
    while True:
        line = f.readline()
        if not line: break
        a = list(map(int, line.split()))
        pred.append(a)
    pred = np.asarray(pred)
    f.close()

    acc = mm.MOTAccumulator(auto_id=True)
    frame_idx = 0
    count = 0
    max_index = max(max(gt[:, 0]), max(pred[:, 0]))

    while frame_idx <= max_index:
        frame_idx += 1

        gt_indexs = gt[:, 0]
        pred_indexs = pred[:, 0]

        mask1 = frame_idx == gt_indexs
        mask2 = frame_idx == pred_indexs

        # if not gt[mask1].shape[0] and not pred[mask2].shape[0]:
        #     break

        # gt_ids = sorted(list(set(gt[mask1][:, 1])))
        # pred_ids = sorted(list(set(pred[mask2][:, 1])))

        gt_ids = gt[mask1][:, 1]
        pred_ids = pred[mask2][:, 1]
        # print(gt_ids)
        # print(pred_ids)

        a = gt[mask1][:, 2:]
        b = pred[mask2][:, 2:]
        # print(mm.distances.iou_matrix(a, b, max_iou=0.5))

        f = acc.update(
            gt_ids,
            pred_ids,
            mm.distances.iou_matrix(a, b, max_iou=0.5)
        )
        # print(mm.distances.iou_matrix(a, b, max_iou=0.5))
        # print(acc.mot_events.loc[f])


    mh = mm.metrics.create()
    custom_metric = [
        "num_frames",
        "obj_frequencies",
        "pred_frequencies",
        "num_matches",
        "num_switches",
        "num_transfer",
        "num_ascend",
        "num_migrate",
        "num_false_positives",
        "num_misses",
        "num_detections",
        "num_objects",
        "num_predictions",
        "num_unique_objects",
        "track_ratios",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_fragmentations",
        "motp",
        "mota",
        "precision",
        "recall",
    ]
    summary = mh.compute_many(
        [acc, acc.mot_events],
        metrics=mm.metrics.motchallenge_metrics,
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    print(strsummary)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

