import cv2
import numpy as np
import time
from datetime import datetime
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


drawing = False
Box_start = []
Box_ing = ()
Box_end = []

### List 원소 중복 제거를 위해, 순서를 살려서 list 살리기
def OrderedSet(list):
    my_set = set()
    result = []
    for element in list:
        if element not in my_set:
            result.append(element)
            my_set.add(element)

    return result

### Mouse Event : 드래그 앤 드랍으로 object를 계산
def mouse_drawing(event, x, y, flags, params):
    global Box_start, Box_end, Box_ing, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            Box_start.append((x, y))
            print("Box_start is ", Box_start)
        else:
            drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            Box_ing = (x, y)
            # print(Box_ing)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing is True:
            Box_end.append((x, y))
            print("Box_end is ", Box_end)
            drawing = False
            Box_ing = ()

### Extend Box : 기존 넓이의 2.56(1.6의 제곱)배 확장. 그러지말고 화면의 비율에 맞게끔 사람의 덩치에 고려해서 3D point 조정 계산ㄱ
def extend_Box(Box_start, Box_end):
    Box_start_extended = []
    Box_end_extended = []
    if len(Box_end) == len(Box_start) and len(Box_end) != 0:
        # Box extend
        for i in range(len(Box_start)):
            x1 = min(Box_start[i][0], Box_end[i][0]); y1 = min(Box_start[i][1], Box_end[i][1])
            x2 = max(Box_start[i][0], Box_end[i][0]); y2 = max(Box_start[i][1], Box_end[i][1])

            dist_x = (x2 - x1); dist_y = (y2 - y1)

            Box_start_extended.append((int(x1 - dist_x * 0.5), int(y1 - dist_y * 0.5)))
            Box_end_extended.append((int(x2 + dist_x * 0.5), int(y2 + dist_y * 0.5)))

    return Box_start_extended, Box_end_extended

## Alarming == 2 단계 설계
def Warning_or_not(rwrists, lwrists, Box_start, Box_end):
    Warning_Box_list = []
    if len(rwrists) != 0 and (len(Box_end) == len(Box_start)):
        for rwrist in rwrists:
            for i in range(len(Box_start)):
                if (Box_start[i][0] - rwrist[0])*(rwrist[0] - Box_end[i][0]) > 0 and \
                        (Box_start[i][1] - rwrist[1])*(rwrist[1] - Box_end[i][1]) > 0:
                    Warning_Box_list.append(i)
    if len(lwrists) != 0 and (len(Box_end) == len(Box_start)):
        for lwrist in lwrists:
            for i in range(len(Box_end)):
                if (Box_start[i][0] - lwrist[0])*(lwrist[0] - Box_end[i][0]) > 0 and \
                        (Box_start[i][1] - lwrist[1])*(lwrist[1] - Box_end[i][1]) > 0:
                    Warning_Box_list.append(i)
    Warning_Box_list = set(Warning_Box_list)
    return Warning_Box_list

### Alarming == 1 단계 설계
def Notice_or_not(rwrists, lwrists, shoulders, Box_start_extended, Box_end_extended):
    Notice_Box_list = []
    if len(Box_end_extended) == len(Box_start_extended):
        if len(rwrists) != 0:
            for rwrist in rwrists:
                for i in range(len(Box_start_extended)):
                    if (Box_start_extended[i][0] - rwrist[0]) * (rwrist[0] - Box_end_extended[i][0]) > 0 and \
                            (Box_start_extended[i][1] - rwrist[1]) * (rwrist[1] - Box_end_extended[i][1]) > 0:
                        Notice_Box_list.append(i)
        if len(lwrists) != 0:
            for lwrist in lwrists:
                for i in range(len(Box_end_extended)):
                    if (Box_start_extended[i][0] - lwrist[0]) * (lwrist[0] - Box_end_extended[i][0]) > 0 and \
                            (Box_start_extended[i][1] - lwrist[1]) * (lwrist[1] - Box_end_extended[i][1]) > 0:
                        Notice_Box_list.append(i)
        #
        # if len(shoulders) != 0:
        #     for index in range(0, len(shoulders), 2):
        #         for i in range(len(Box_end_extended)):
        #             if shoulders[index][0] < shoulders[index + 1][0]:
        #                 if (Box_start_extended[i][0] < (shoulders[index][0] + shoulders[index + 1][0])/2 < Box_end_extended[i][0])\
        #                     and (Box_start_extended[i][1] < (shoulders[index][1] + shoulders[index + 1][1])/2 < Box_end_extended[i][1]):
        #                     print("등돌렸음 between", shoulders[index], "and", shoulders[index + 1], "at", datetime.now())
        #                     ### 부가적으로 1단계 알람 Caputure

    Notice_Box_list = set(Notice_Box_list)
    return Notice_Box_list

### initalize
cam = cv2.VideoCapture(0)
fps_time = 0
w, h = model_wh('0x0') ## resize == '0x0' (default)
graph_path_list = ['cmu', 'mobilenet_thin', 'mobilenet_v2_small', 'mobilenet_v2_large']
e = TfPoseEstimator(get_graph_path(graph_path_list[3]), target_size=(432, 368), trt_bool=False)

while True:
    ret_val, image = cam.read()
    rows, cols = image.shape[:2]

    ### 180도 변환, 화면 회전 변환 조정
    mat = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
    # mat = cv2.getRotationMatrix2D((cols/2, rows/2), , 1)
    image = cv2.warpAffine(image, mat, (cols, rows))

    ### 사람 만들기 : 위에서 네트워크를 불러와서 사람들의 중요 신체 부위에 포인트를 찍는다.
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    ### 이미지 뒤집어쓰기 : 사람의 뼈대를 추출한다..
    humans_RWrist_List = []
    humans_LWrist_List = []
    humans_Shoulder_List = []
    # image, humans_RWrist_List, humans_LWrist_List = TfPoseEstimator.extract_points(image, humans, imgcopy=False)
    image, total_humans_point = TfPoseEstimator.human_extract_points(image, humans, imgcopy=False)
    # print(total_humans_point)
    # print(len(total_humans_point))
    for human_point in total_humans_point:
        if "RWrist" in human_point:
            humans_RWrist_List.append(human_point["RWrist"])
        if "LWrist" in human_point:
            humans_LWrist_List.append(human_point["LWrist"])
        if "RShoulder" in human_point and "LShoulder" in human_point:  # 애초에 한 사람에게서 같은 짝의 어깨를 추출
            humans_Shoulder_List.append(human_point["LShoulder"])  # 오른쪽이 짝수번 째
            humans_Shoulder_List.append(human_point["RShoulder"])  # 왼쪽이 홀수번 쨰

    humans_LWrist_List = OrderedSet(humans_LWrist_List)
    humans_RWrist_List = OrderedSet(humans_RWrist_List)
    humans_Shoulder_List = OrderedSet(humans_Shoulder_List) # 항상 len이 2개씩 저장

    # Draw Boxes : 현재는 동일 화면에 띄운다.
    cv2.namedWindow('tf-pose-estimation result')
    cv2.setMouseCallback('tf-pose-estimation result', mouse_drawing)

    Box_start_extended, Box_end_extended = extend_Box(Box_start, Box_end)

    if len(Box_start) != 0 and len(Box_ing) != 0:
        cv2.rectangle(image, Box_start[-1], Box_ing, (0, 255, 0), 5)
    if len(Box_start) != 0 and len(Box_end_extended) != 0:
        for i in range(len(Box_start) - 1):
            cv2.rectangle(image, Box_start[i], Box_end[i], (255, 0, 0), 5)
            cv2.rectangle(image, Box_start_extended[i], Box_end_extended[i], (255, 0, 0), 3)

        if len(Box_end_extended) == len(Box_start):
            cv2.rectangle(image, Box_start[-1], Box_end[-1], (255, 0, 0), 5)
            cv2.rectangle(image, Box_start_extended[-1], Box_end_extended[-1], (255, 0, 0), 3)

    ### Alarming 0 : ㄱㅊ, 1 : 주의, 2 : 경고
    alarm = 0
    Warning_Box_list = Warning_or_not(humans_RWrist_List, humans_LWrist_List, Box_start, Box_end)
    Noticed_Box_list = Notice_or_not(humans_RWrist_List, humans_LWrist_List, humans_Shoulder_List,
                                     Box_start_extended, Box_end_extended)

    if len(Warning_Box_list) > 0:  # 2단계 경고 신호
        for Warning_at in Warning_Box_list:
            print("2단계 Detected!!! at ", Warning_at + 1, " box : ", "왼쪽 : ", humans_LWrist_List, "오른쪽 : ",
                  humans_RWrist_List, 'at : ', datetime.now())
            cv2.rectangle(image, Box_start[Warning_at], Box_end[Warning_at], (0, 0, 255), 5)
            alarm = 2

    if len(Noticed_Box_list) > 0: # 1단계 경고 신호
        for Noticed_at in Noticed_Box_list:
            print("1단계 Detected!!! at ", Noticed_at + 1, " box : ", "왼쪽 : ", humans_LWrist_List, "오른쪽 : ",
                  humans_RWrist_List, 'at : ', datetime.now())
            cv2.rectangle(image, Box_start_extended[Noticed_at], Box_end_extended[Noticed_at], (0, 0, 255), 3)

    ### UI 띄우기
    cv2.putText(image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)

    # Reset
    fps_time = time.time()

    ### 'key' listener : 'Pop : 'b'', 'Escape : 'Esc'
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif len(Box_start) + len(Box_end) >= 2 and k == 98:
        Box_start.pop()
        Box_end.pop()
        print(Box_start)
        print(Box_end)

cam.release()
cv2.destroyAllWindows()
