import cv2
import numpy as np
import os

ROW = 16
COLUMN = 49
# locate the 3 corner wells of the chip labelled by users using blue circles (bottom-left, bottom-right and top-left).
def FindLabels(labels,r,c):

    bl_region = np.copy(labels[int(r / 2):r,0:int(c / 2)])
    br_region = np.copy(labels[int(r / 2):r,int(c / 2):c])
    tl_region = np.copy(labels[0:int(r / 2),0:int(c / 2)])

    #cv2.imshow('bl',bl_region)
    #cv2.imshow('br',br_region)
    #cv2.imshow('tl',tl_region)
    #cv2.waitKey(0)

    bl_well = cv2.HoughCircles(bl_region, cv2.HOUGH_GRADIENT, 1, 200, \
                             param1=150, param2=10)
    br_well = cv2.HoughCircles(br_region, cv2.HOUGH_GRADIENT, 1, 200, \
                             param1=150, param2=10)
    tl_well = cv2.HoughCircles(tl_region, cv2.HOUGH_GRADIENT, 1, 200, \
                             param1=150, param2=10)
    #print (bl_well[0][0])
    bl_center = (bl_well[0][0][0],int (r / 2) + bl_well[0][0][1])
    br_center = (int (c / 2) + br_well[0][0][0],int(r / 2) + br_well[0][0][1])
    tl_center = (tl_well[0][0][0],tl_well[0][0][1])
    radius = (bl_well[0][0][2] + br_well[0][0][2] + tl_well[0][0][2]) / 3

    # for i in bl_well[0,:]:
    #     cv2.circle(bl_region,(i[0],i[1]),i[2],127,1)
    # for i in br_well[0,:]:
    #     cv2.circle(br_region,(i[0],i[1]),i[2],127,1)
    # for i in tl_well[0,:]:
    #     cv2.circle(tl_region,(i[0],i[1]),i[2],127,1)

    # print (radius)
    return (bl_center,br_center,tl_center,int(radius))


outputfile = open('result.txt','w')
outputfile.write("Img\tRow\tIntensity\tSTD\n")

for imgfile in os.listdir('.'):
    if imgfile.endswith('.jpg') and imgfile.startswith('grid') is False:

        print (imgfile)
        img0 = cv2.imread(imgfile)
        r,c,_ = img0.shape
        img0 = cv2.resize(img0,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        r1 = int(r / 2)
        c1 = int(c / 2)
        #cv2.namedWindow('wells',cv2.WINDOW_NORMAL)

        #blue labels made by users
        labels = img0[:,:,0]
        #red fluorescent signal
        img = img0[:,:,2]

        #_,thresh = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
        _,labels = cv2.threshold(labels,127,255,cv2.THRESH_BINARY)
        #cv2.imshow('labels',img0)

        bl,br,tl,radius = FindLabels(labels,r1,c1)
        template = np.zeros((2 * radius,2 * radius),np.uint8)
        cv2.circle(template,(radius,radius),radius,255,-1)


        #print (bl,br,tl,radius)

        # 49 * 16 chip
        x_dist_row = (br[0] - bl[0]) / (COLUMN - 1)
        y_dist_row = (br[1] - bl[1]) / (COLUMN - 1)
        x_dist_col = (bl[0] - tl[0]) / (ROW - 1)
        y_dist_col = (bl[1] - tl[1]) / (ROW - 1)

        intensity = []

        # calculate based on the 48 * (2 * 8) region
        for i in range(ROW):
            intensity.append([])
            #print (intensity)
            anchor = (int(tl[0] + i * x_dist_col),int(tl[1] + i * y_dist_col))
            cv2.putText(img0,str(i),anchor,cv2.FONT_HERSHEY_COMPLEX,4,(255,0,0),3)
            for j in range(COLUMN - 2):
                x = int(anchor[0] + (j + 1) * x_dist_row)
                y = int(anchor[1] + (j + 1) * y_dist_row)

                roi = np.copy(img[y-radius-5:y+radius+5,x-radius-5:x+radius+5])
                _,roi_thresh = cv2.threshold(roi,cv2.mean(roi)[0]-3,255,cv2.THRESH_BINARY)
                matching = cv2.matchTemplate(roi_thresh,template,cv2.TM_SQDIFF_NORMED)
                _,_,(x0,y0),_ = cv2.minMaxLoc(matching)
                mask = np.zeros((2 * radius + 10,2 * radius + 10),np.uint8)
                cv2.circle(mask,(x0+radius,y0+radius),radius,255,-1)
                cv2.circle(img0,(x-5+x0,y-5+y0),radius,(255,0,0),3)

                #cv2.imshow('roi',roi_thresh)
                #cv2.imshow('mask',mask)
                #cv2.waitKey(0)

                intensity[i].append(cv2.mean(roi,mask=mask)[0])
            # anchor2 = (anchor[0] + x_dist_col,anchor[1] + y_dist_col)
            # for j in range(47):
            #     x = int(anchor2[0] + (j + 1) * x_dist_row)
            #     y = int(anchor2[1] + (j + 1) * y_dist_row)
            #
            #     roi = np.copy(img[y-radius-5:y+radius+5,x-radius-5:x+radius+5])
            #     _,roi_thresh = cv2.threshold(roi,cv2.mean(roi)[0]-3,255,cv2.THRESH_BINARY)
            #     matching = cv2.matchTemplate(roi_thresh,template,cv2.TM_SQDIFF_NORMED)
            #     _,_,(x0,y0),_ = cv2.minMaxLoc(matching)
            #     mask = np.zeros((2 * radius + 10,2 * radius + 10),np.uint8)
            #     cv2.circle(mask,(x0+radius,y0+radius),radius,255,-1)
            #     cv2.circle(img0,(x-5+x0,y-5+y0),radius,(255,0,0),3)
            #
            #     intensity[i].append(cv2.mean(roi,mask=mask)[0])

        for i in range(ROW):
            #print (np.mean(intensity[i]),np.std(intensity[i]))
            outputfile.write("%s\t%i\t%.2f\t%.2f\n" % (imgfile,i,np.mean(intensity[i]),np.std(intensity[i])))

        #cv2.imshow('wells',img0)
        #cv2.resizeWindow('wells',(2000,int(r/(c/2000))))
        cv2.imwrite('grid' + imgfile,img0)
        #cv2.waitKey(0)

outputfile.close()
#cv2.waitKey(0)
