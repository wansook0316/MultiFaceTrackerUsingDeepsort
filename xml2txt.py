import xml.etree.ElementTree as elemTree
import os
import pprint
from absl import app, flags, logging
from absl.flags import FLAGS

"""
python xml2txt.py \
    --gt_path ./resources/gt/T-ara_gt.xml \
    --gt_file_path ./resources/gt/T-ara_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/GirlsAloud_gt.xml \
    --gt_file_path ./resources/gt/GirlsAloud_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Darling_gt.xml \
    --gt_file_path ./resources/gt/Darling_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Westlife_gt.xml \
    --gt_file_path ./resources/gt/Westlife_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/BrunoMars_gt.xml \
    --gt_file_path ./resources/gt/BrunoMars_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/HelloBubble_gt.xml \
    --gt_file_path ./resources/gt/HelloBubble_gt.txt

python xml2txt.py \
    --gt_path ./resources/gt/Apink_gt.xml \
    --gt_file_path ./resources/gt/Apink_gt.txt
"""

flags.DEFINE_string('gt_path', './resources/gt/T-ara_gt.xml', 'path to gt')
flags.DEFINE_string('gt_file_path', './resources/gt/T-ara_gt.txt', 'path to save converted file')


def main(args):
    tree = elemTree.parse(FLAGS.gt_path)

    root=tree.getroot()

    print(root.tag, root.attrib)
    print(root.find("Trajectory"))

    frame_list = []

    for traj in root:
        for f in traj:
            a = f.attrib
            a["frame_no"] = str(int(a["frame_no"])-1)
            a["id"] = traj.attrib["obj_id"]
            frame_list.append(a)


    frame_list = sorted(frame_list, key= lambda x: (int(x["frame_no"]), int(x["id"])))
    # pprint.pprint(frame_list)


    f = open(FLAGS.gt_file_path, 'w')

    for a in frame_list:
        f.write(a["frame_no"] + " " + a["id"] + " " + a["x"] + " " + a["y"] + " " + a["width"] + " " + a["height"] + "\n")
    # 파일 닫기
    f.close()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
