# -*- coding: utf-8 -*-
"""

@author: Karthik Iyer
"""

import cv2
import numpy as np
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")

#source_image = cv2.imread("images/jason.jpg")
#source_image = cv2.imread("images/RitiknShidu.jpg")
source_image = cv2.imread("images/hri.jpg")

source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("source_image",source_image)


#destination_image = cv2.imread("images/brucewills.jpg")
destination_image = cv2.imread("images/karthik.jpg")
#destination_image = cv2.imread("images/ShidunRitik.jpg")
#destination_image = cv2.imread("image/modi.jpg")
destination_image_grayscale = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("destination_image",destination_image)

source_image_canvas = np.zeros_like(source_image_grayscale)

height, width, no_of_channels = destination_image.shape

destination_image_canvas = np.zeros((height,width,no_of_channels),np.uint8)

def index_from_array(numpyarray):
    index = None
    for n in numpyarray[0]:
        index = n
        break
    return index

source_faces = frontal_face_detector(source_image_grayscale)


for source_face in source_faces:
    
    source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
    source_face_landmark_points = []
    
    
    
    for landmark_no in range(0,68):
        x_point = source_face_landmarks.part(landmark_no).x
        y_point = source_face_landmarks.part(landmark_no).y
        source_face_landmark_points.append((x_point, y_point))
        
        # cv2.circle(source_image,(x_point,y_point),2,(255,255,0),-1)
        # cv2.putText(source_image, str(landmark_no), (x_point,y_point), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255))
        # cv2.imshow("1: landmark points of source",source_image)

    
    source_face_landmark_points_array = np.array(source_face_landmark_points,np.int32)
    
    source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)
    
    # cv2.polylines(source_image, [source_face_convexhull], True, (255,0,0),1)
    # cv2.imshow("2: convex hull of source image face",source_image)
    
    
    cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)
    #cv2.imshow("3: draw the convexhull polygon over canvas", source_image_canvas)
    
    
    source_face_image = cv2.bitwise_and(source_image,source_image,mask=source_image_canvas)
    #cv2.imshow("4: place mask over source image", source_face_image)

    bounding_rectangle = cv2.boundingRect(source_face_convexhull)

    
    subdivisions = cv2.Subdiv2D(bounding_rectangle)
    
    subdivisions.insert(source_face_landmark_points)
    
    triangles_vector = subdivisions.getTriangleList()

    triangles_array = np.array(triangles_vector,dtype=np.int32)
    
    #print(triangles_array)
    source_triangle_index_points_list = []
    
    for triangle in triangles_array:
        index_point1 = (triangle[0], triangle[1])
        index_point2 = (triangle[2], triangle[3])
        index_point3 = (triangle[4], triangle[5])
        
        # line_color = (255,0,0)
        # cv2.line(source_face_image, index_point1, index_point2, line_color, 1 )
        # cv2.line(source_face_image, index_point2, index_point3, line_color, 1 )
        # cv2.line(source_face_image, index_point3, index_point1, line_color, 1 )
        
        # cv2.imshow("5: Drawing all Delaunay triangles in source image ", source_face_image)

        index_point1 = np.where((source_face_landmark_points_array == index_point1).all(axis=1))
        index_point1 = index_from_array(index_point1)
        index_point2 = np.where((source_face_landmark_points_array == index_point2).all(axis=1))
        index_point2 = index_from_array(index_point2)
        index_point3 = np.where((source_face_landmark_points_array == index_point3).all(axis=1))
        index_point3 = index_from_array(index_point3)
        
        triangle = [index_point1, index_point2, index_point3]
        source_triangle_index_points_list.append(triangle)
        
#print(triangle_index_points_list)
   

destination_faces = frontal_face_detector(destination_image_grayscale)

for destination_face in destination_faces:
    #predictor takes human face as input and returns the list of facial landmarks
    destination_face_landmarks = frontal_face_predictor(destination_image_grayscale, destination_face)
    destination_face_landmark_points = []
    
    for landmark_no in range(0,68):
        x_point = destination_face_landmarks.part(landmark_no).x
        y_point = destination_face_landmarks.part(landmark_no).y
        destination_face_landmark_points.append((x_point, y_point))
        #just for demo
        # cv2.circle(destination_image,(x_point,y_point),2,(255,255,0),-1)
        # cv2.putText(destination_image, str(landmark_no), (x_point,y_point), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,255,255))
        # cv2.imshow("1: landmark points of destination",destination_image)

    
    destination_face_landmark_points_array = np.array(destination_face_landmark_points,np.int32)
    
    destination_face_convexhull = cv2.convexHull(destination_face_landmark_points_array)
    
    # cv2.polylines(destination_image, [destination_face_convexhull], True, (255,0,0),1)
    # cv2.imshow("2: convex hull of destination image face",destination_image)
    

for i, triangle_index_points in enumerate(source_triangle_index_points_list):
    

    source_triangle_point1 = source_face_landmark_points[triangle_index_points[0]]
    source_triangle_point2 = source_face_landmark_points[triangle_index_points[1]]
    source_triangle_point3 = source_face_landmark_points[triangle_index_points[2]]
    
    source_triangle = np.array([source_triangle_point1,source_triangle_point2,source_triangle_point3], np.int32)
    
    source_rectangle = cv2.boundingRect(source_triangle)
    (x,y,w,h) = source_rectangle
    cropped_source_rectangle = source_image[y:y+h, x:x+w]
    
    source_triangle_points = np.array([[source_triangle_point1[0] - x, source_triangle_point1[1] - y],
                                        [source_triangle_point2[0] - x, source_triangle_point2[1] - y],
                                        [source_triangle_point3[0] - x, source_triangle_point3[1]- y]], np.int32)
                                      
    # if i==10:
    #     #display triangle lines in white, rectangle in red
    #     cv2.line(source_image,source_triangle_point1,source_triangle_point2, (255,255,255))
    #     cv2.line(source_image,source_triangle_point2,source_triangle_point3, (255,255,255))
    #     cv2.line(source_image,source_triangle_point3,source_triangle_point1, (255,255,255))
    #     cv2.imshow('8.1 Source Triangle Lines', source_image)
    #     cv2.rectangle(source_image, (x,y), (x+w,y+h), (0,0,255))
    #     cv2.imshow('8.2 Source Rectangle Lines', source_image)
    #     cv2.imshow('8.3 Cropped Source Rectangle', cropped_source_rectangle)


    destination_triangle_point1 = destination_face_landmark_points[triangle_index_points[0]]
    destination_triangle_point2 = destination_face_landmark_points[triangle_index_points[1]]
    destination_triangle_point3 = destination_face_landmark_points[triangle_index_points[2]]
    destination_triangle = np.array([destination_triangle_point1, destination_triangle_point2, destination_triangle_point3], np.int32)
    
    destination_rectangle = cv2.boundingRect(destination_triangle)
    (x, y, w, h) = destination_rectangle
    
    
    cropped_destination_rectangle = source_image[h,w]
    cropped_destination_rectangle_mask = np.zeros((h, w), np.uint8)
    
    destination_triangle_points = np.array([[destination_triangle_point1[0] - x, destination_triangle_point1[1] - y],
                        [destination_triangle_point2[0] - x, destination_triangle_point2[1] - y],
                        [destination_triangle_point3[0] - x, destination_triangle_point3[1] - y]], np.int32)

    
    cv2.fillConvexPoly(cropped_destination_rectangle_mask, destination_triangle_points, 255)

    
    # if i==10:
    #     cv2.line(destination_image,destination_triangle_point1,destination_triangle_point2, (255,255,255), 1)
    #     cv2.line(destination_image,destination_triangle_point2,destination_triangle_point3, (255,255,255), 1)
    #     cv2.line(destination_image,destination_triangle_point3,destination_triangle_point1, (255,255,255), 1)
    #     cv2.imshow("9.1: Destination Triangle Lines", destination_image)        
    #     cv2.rectangle(destination_image,(x,y),(x+w,y+h), (0,0,255), 1)
    #     cv2.imshow("9.2: Destination rectangle Lines", destination_image)
    #     cv2.imshow("9.3: Destination filled rectangle mask", cropped_destination_rectangle_mask)

    
    source_triangle_points = np.float32(source_triangle_points)
    destination_triangle_points = np.float32(destination_triangle_points)
    
    Matrix = cv2.getAffineTransform(source_triangle_points, destination_triangle_points)
    
    warped_triangle = cv2.warpAffine(cropped_source_rectangle, Matrix, (w,h))
    # if i==10:
    #     cv2.imshow("10.1: warped source triangle wrt the destination triangle points",warped_triangle)
    #placing destination rectangle mask over the warped triangle
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_destination_rectangle_mask)
    #for demo, select triangle 10
    # if i==10:
    #     cv2.imshow("10.2: warped source triangle with the mask",warped_triangle)

    new_dest_face_canvas_area = destination_image_canvas[y: y+h, x: x+w]
    
    new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)
    
    _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    
    wraped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask = mask_created_triangle)
    
    new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, wraped_triangle)
    
    destination_image_canvas[y: y+h, x: x+w] = new_dest_face_canvas_area
    #for demo, select triangle 10
#     if i==10:
#         cv2.imshow("11: pasting the triangle at destination canvas",destination_image_canvas)    
    
# cv2.imshow("12: the completed destination canvas", destination_image_canvas)    
    

final_destination_canvas = np.zeros_like(destination_image_grayscale)
#cv2.imshow("13.1: the final destination canvas", final_destination_canvas)     

final_destination_face_mask = cv2.fillConvexPoly(final_destination_canvas, destination_face_convexhull, 255)
#cv2.imshow("13.2: the final destination face mask", final_destination_face_mask)     
    
final_destination_canvas = cv2.bitwise_not(final_destination_face_mask)    
#cv2.imshow("13.3: the inverted final destination face mask", final_destination_face_mask)      
    
destination_face_masked = cv2.bitwise_and(destination_image, destination_image, mask=final_destination_canvas)  
#cv2.imshow("13.4: the destination_face_masked", destination_face_masked)

destination_with_face = cv2.add(destination_face_masked,destination_image_canvas)      
#cv2.imshow("13.5: the destination_with_face", destination_with_face)  
    

(x,y,w,h) = cv2.boundingRect(destination_face_convexhull)  
destination_face_center_point = (int((x+x+w)/2), int((y+y+h)/2))

seamlesscloned_face = cv2.seamlessClone(destination_with_face, destination_image, final_destination_face_mask, destination_face_center_point, cv2.NORMAL_CLONE)

cv2.imshow("14: seamlesscloned_face", seamlesscloned_face)  

cv2.waitKey(0)
cv2.destroyAllWindows()