import cv2
import numpy as np
import math

def main():
    for i in range(30, 32):
        """
        1. 이미지 캡처(이미지 읽기)
        """
        imgpath = "S{:02d}.jpg".format(i)
        img_color = cv2.imread(imgpath, 1)
        windowname = "S{:02d}".format(i)
        cv2.namedWindow(windowname)
        
        """
        2. 이미지 사이즈 설정(4:3 고정)
        """
        img_resize = cv2.resize(img_color, (400,300), interpolation=cv2.INTER_AREA)
        
        """
        3. GrayScale로 변환
        """
        img_gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
        
        """
        4. 영상 변환을 위해 각각의 이미지 복사
        """
        img_copy = img_resize.copy()
        gray_copy = img_gray.copy()
         
        """
        5. 영상 잡음 제거 - bilateral 필터링
        """
        blur = cv2.bilateralFilter(gray_copy, 5, 75, 75)
        
        """
        6. 히스토그램 균일화
        """        
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        hist_img = clahe.apply(blur)
        
        """
        7. Mopolozy연산을 통한 잡음 제거
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(hist_img, cv2.MORPH_OPEN, kernel, iterations = 40)
        
        """
        8. subtract image
        """
        sub_img = cv2.subtract(gray_copy, opening)
        
        """
        9. OTSU 이진 Thresholding
        """
        ret, thr = cv2.threshold(sub_img, 0, 255, cv2.THRESH_OTSU)
        
        """
        10. Canny Edge Detection
        """
        canny = cv2.Canny(thr, 250, 255)
       
        """
        11. Deliation연산 - strengthen edges
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(canny, kernel, iterations=1)
        
        """
        12. Find contours
        """
        cimg, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        """
        13. Contour의 영역 넓이 기준으로 내림차순 정렬 후 상위 30개에서 번호판 탐색
        """
        contours = sorted(contours, key = cv2.contourArea, reverse=True)[:30]
       
        """
        14. Contour points의 수를 4개로 줄여 사각형 line 검출
            - elipsion = 0.3
            - 넓이 범위 제한(너무 큰 영역, 너무 작은 영역 감지할 시)
            - 검출된 영역의 좌우 세로 길이, 상하 가로 길이 비교 차이값 계산하여 비정상적인 사각형 필터링
        """
        
        final = img_resize
        plate = None
        
        for c in contours:
            arc = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * arc, True)
            
            #point가 4개일 때(사각형일 때)
            if len(approx) == 4:
                area = cv2.contourArea(c)
              
                #넓이 범위 필터링
                if area > 10000 or area < 1000:
                    continue
                
                else:
                    plate = approx
                    
                    #영역 좌표 정렬
                    point1 = np.float32([plate[0][0], plate[1][0], plate[2][0], plate[3][0]])
                    point1 = point1[point1[:, 0].argsort()[::-1]]
                    
                    if(point1[0][1] > point1[1][1]):
                        point1[0][0], point1[1][0] = point1[1][0], point1[0][0]
                        point1[0][1], point1[1][1] = point1[1][1], point1[0][1]
                        
                    if(point1[2][1] > point1[3][1]):
                        point1[2][0], point1[3][0] = point1[3][0], point1[2][0]
                        point1[2][1], point1[3][1] = point1[3][1], point1[2][1]
                        
                    #2-3번 거리(왼쪽 세로)
                    r23 = point1[3][0] - point1[2][0]
                    c23 = point1[3][1] - point1[2][1]
                    d23 = math.sqrt(r23**2 + c23**2)
                    #0-1번 거리(오른쪽 세로)
                    r01 = point1[0][0] - point1[1][0]
                    c01 = point1[0][1] - point1[1][1]
                    d01 = math.sqrt(r01**2 + c01**2)
                    #2-0번 거리(위쪽 가로)
                    r02 = point1[0][0] - point1[2][0]
                    c02 = point1[0][1] - point1[2][1]
                    d02 = math.sqrt(r02**2 + c02**2)
                    #3-1번 거리(아래쪽 가로)
                    r13 = point1[1][0] - point1[3][0]
                    c13 = point1[1][1] - point1[3][1]
                    d13 = math.sqrt(r13**2 + c13**2)
                    
                    diff1 = abs(d23 - d01)
                    diff2 = abs(d02 - d13)
                    
                    if diff1 > 15 or diff2 > 15:
                        plate = None
                        continue

                    final = cv2.drawContours(final, [plate], -1, (0, 255, 0), 2)   #대상 contour
                    break
    
        """
        15. Perspective Transformation
            - 인식 못 할 경우: 400x100 검은 영상에 "Can't find any plates" 문구 띄우기
            - 인식 성공할 경우: 인식된 영역의 가로, 세로 중 최대 길이 선정, 해당 길이의 3배에 해당하는 새로운 좌표 생성
                                해당 좌표로 perspective transformation 후 output이라는 새로운 영상 생성
        """
        if plate is None:
            output = np.zeros((100, 400, 1), np.uint8)
            cv2.putText(output, "Can't find any plates", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        else:
            row = 3 * int(max(d02, d13))#가로
            col = 3 * int(max(d23, d01))#세로
            
            #변환 좌표
            point2 = np.float32([[row, 0], [row, col], [0, 0], [0, col]])
            
            #영역 꼭지점 표시
            cv2.circle(final, (point1[0][0], point1[0][1]), 5, (255,0,0), -1)
            cv2.circle(final, (point1[1][0], point1[1][1]), 5, (0,255,0), -1)
            cv2.circle(final, (point1[2][0], point1[2][1]), 5, (0,0,255), -1)
            cv2.circle(final, (point1[3][0], point1[3][1]), 5, (255,255,255), -1)
            
            #Perspective transformation
            P = cv2.getPerspectiveTransform(point1, point2)
            output = cv2.warpPerspective(img_copy, P, (row, col))
            
        """
        16. 결과 출력
            - 기존 원본 이미지에 검출 영역 표시
            - 새로운 window(License Plate)에 번호판 출력
        """
        cv2.imshow('License Plate', output)
        cv2.imshow(windowname, final)
        
        while(True):
            K = cv2.waitKey(1)
            if K == 13:
                break     
            elif K == 27:
                cv2.destroyAllWindows() 
                return 0
            
        cv2.destroyAllWindows() 
    
if __name__ == "__main__":
    main()