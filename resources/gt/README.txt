--------------------------------------------------------------------
Music video dataset
--------------------------------------------------------------------

This file describes the music video dataset as introduced in

[1] Shun Zhang, Yihong Gong, Jia-Bin Huang, Jongwoo Lim, Jinjun Wang, 
Narendra Ahuja and Ming-Hsuan Yang. Tracking Persons-of-Interest via 
Adaptive Discriminative Features[C]. The 14th European Conference on 
Computer Vision (ECCV), 2016.
[2] The project website: http://shunzhang.me.pn/papers/eccv2016/

The dataset contains manually annotated face trajectories from 8 music 
videos from YouTube: T-ara, Westlife, Pussycat Dolls, Apink, Darling, 
Bruno Mars, Hello Bubble and Girls Aloud (as detailed in [1,2]).

Kindly cite [1] when using the dataset, where appropriate.

--------------------------------------------------------------------
Description of the files
--------------------------------------------------------------------

The annotations for each video are stored in an XML file. 
We give an XML example below and introduce the XML format.


Example:
1. <?xml version = "1.0"?>
2. <Video fname="gt" start_frame="1" end_frame="5275">
3.   <Trajectory obj_id="1" start_frame="253" end_frame="5275">
4.       <Frame frame_no="253" x="639" y="155" width="109" height="139"></Frame>
5.       ...
6.   </Trajectory>
7. </Video>


The 1st line at the top of our example is the XML declaration 
that indicates the version of XML.
The 2nd line indicates the video information, including video name, 
start frame and end frame.
The 3rd line indicates the trajectory information, including trajectory 
identity, start frame and end frame.
The 4th line contains 5 values of per bounding box:
<frame_num>,<x-bb_left>,<y-bb_top>,<bb_width>,<bb_height>
(x-bb_left,y-bb_top) is the left-top point of the bounding box. 
<bb_width,bb_height> is the width and height of the bounding box.

-- EOF
