#start up for the raspberry pi and motor
import os
os.environ["PIGPIO_ADDR"] = "169.254.176.114" # ip address of Raspberry Pi for Henry's laptop
# os.environ["PIGPIO_ADDR"] = "192.168.137.41" # ip address of Raspberry Pi for Celeste's big computer
os.environ["GPIOZERO_PIN_FACTORY"] = "pigpio"
from gpiozero import DigitalOutputDevice
from time import sleep 

#start up for computer vision algorithms
from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler 
import imutils
import cv2
import numpy as np
import time

#start up for the AL5D ---------------------------------------------------------------------------------
import serial
ssc32 = serial.Serial('COM5', 9600, timeout = 1) #change to COM3
    
spd = 1000 # motor speed
grip_spd = 2000 # gripper speed

# controls servo motor for single servo control
def cmd(joint, encoder_val, speed):
    joint_list = {
        'base': 0,
        'shoulder': 1,
        'elbow': 2,
        'wrist': 3,
        'wrist_r': 5,
        'gripper': 7
        }
    servo = joint_list.get(joint)
    ssc32.write("#%i P%i S%i \r".encode() %(servo,encoder_val,speed))

def gripper(position):
    if position == 'open':
        cmd('gripper', 1000, grip_spd)
    elif position == 'close':
        cmd('gripper', 1650, grip_spd)

# Defines HOME position of arm
def home():
    gripper('open')
    cmd('base', 1410, spd)
    cmd('shoulder', 1390, spd)
    cmd('elbow', 1410, spd)
    cmd('wrist', 1550, spd)
    cmd('wrist_r', 1610, spd)

#### ENTRY ###

# Defines pre-ENTRY position of arm 
def intermediate_entry(): 
    ssc32.write("#0 P2030 Sspd #1 P1390 Sspd #2 P1400 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines ENTRY position of puck
# Note: Wrist should move faster than rest
# shoulder = 1100, elbow = 1650
def entry():
    ssc32.write("#0 P2030 Sspd #1 P1100 Sspd #2 P1630 Sspd #3 P2050 Sspd #5 P1610 Sspd T800 \r".encode())

### PROCESSING ###

# Defines pre-PROCESSING position of arm
def intermediate_process():
    ssc32.write("#0 P780 Sspd #1 P1550 Sspd #2 P1500 Sspd #3 P1450 Sspd #5 P1610 Sspd T800 \r".encode())
def intermediate_process_2():
    ssc32.write("#0 P780 Sspd #1 P1550 Sspd #2 P1500 Sspd #3 P2000 T200 #5 P1610 Sspd T800 \r".encode())

# Defines PROCESSING position of arm
# s1350, e1900
def process():
    ssc32.write("#0 P780 Sspd #1 P1375 Sspd #2 P1830 Sspd #3 P1940 Sspd #5 P1610 Sspd T800 \r".encode())

### CHUTES ###

# Defines position of pre-CHUTE red
def intermediate_red_chute():
    ssc32.write("#0 P1160 Sspd #1 P1390 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of CHUTE red
def red_chute():
    ssc32.write("#0 P1160 Sspd #1 P1250 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of pre-CHUTE green
def intermediate_green_chute():
    ssc32.write("#0 P1315 Sspd #1 P1390 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of CHUTE green
def green_chute():
    ssc32.write("#0 P1315 Sspd #1 P1250 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of pre-CHUTE blue
def intermediate_blue_chute():
    ssc32.write("#0 P1490 Sspd #1 P1390 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of CHUTE blue
def blue_chute():
    ssc32.write("#0 P1490 Sspd #1 P1250 Sspd #2 P1410 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())

# Defines position of pre-CHUTE defect
def intermediate_defect_chute():
    ssc32.write("#0 P1680 Sspd #1 P1390 Sspd #2 P1430 Sspd #3 P1500 Sspd #5 P1610 Sspd T600 \r".encode())

# Defines position of CHUTE defect
def defect_chute():
    ssc32.write("#0 P1680 Sspd #1 P1250 Sspd #2 P1430 Sspd #3 P1500 Sspd #5 P1610 Sspd T500 \r".encode())
# ------------------------------------------------------------------------------------------------------------

#start up webcams
video = cv2.VideoCapture(1) #process 
video2 = cv2.VideoCapture(0) # entrancepy

#initilze variables for color defect detection
red_detector = 0
blue_detector = 0
green_detector = 0
defect_counter = 0

#variable which tells the robot where to move 1 = red 2 = blue 3 = green 4 = defect
robot_move = 0
robot_text = " "
stat_text = "Number of Pucks Sorted"
red_text = "Red: "
green_text = "Green: "
blue_text = "Blue: "
defect_text = "Defective: "

# initilaze processing station rotation of motor
full_rotation = 0

#initilaze entrance chute presence and color detection
new_puck_entered = 0
black = 0

#statistic variabless
red = 0
blue = 0
green = 0
defect = 0

#motor control
stepPulse = DigitalOutputDevice(17) # use pin 17 for GPIO
EN = DigitalOutputDevice(27,active_high=False) # use pin 27 to control enable pin of stepper motor driver (active low)
stepHz = 350 # stepper motor step frequency

def motor(): # stepper motor function
    EN.on()
    for i in range(0,100): # rotates 180 degrees
        stepPulse.on()
        stepPulse.off()
        sleep(1/stepHz)
    EN.off() 

def statistics():
    blank_image = np.zeros(shape=[290, 420, 3], dtype=np.uint8)
    cv2.putText(blank_image, stat_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(blank_image, red_text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    cv2.putText(blank_image, "{}".format(red), (80,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    cv2.putText(blank_image, green_text, (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.putText(blank_image, "{}".format(green), (110,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    cv2.putText(blank_image, blue_text, (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    cv2.putText(blank_image, "{}".format(blue), (90,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    cv2.putText(blank_image, defect_text, (10,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (169,169,169))
    cv2.putText(blank_image, "{}".format(defect), (165,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (169,169,169))
    cv2.imshow("Statistics", blank_image)
    cv2.waitKey(1)  

while True:

    _, original = video2.read()
    _, frame = video.read()
    
    #reset variables
    red_detector = 0
    blue_detector = 0
    green_detector = 0
    defect_counter = 0
    robot_move = 0
    full_rotation = 0

    home()
    gripper('open')

    crop = original[120:220, 250:450] #[y, x]

    img_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # color masks -----------------------------------------------------------------------------------------------------
    mask_red1 = cv2.inRange(img_hsv, (0,100,100), (10,255,255)) # upper red
    mask_red2 = cv2.inRange(img_hsv, (160,100,100), (179,255,255)) # lower red
    mask_blue = cv2.inRange(img_hsv, (100, 150, 0), (140,255,255)) # blue
    mask_green = cv2.inRange(img_hsv, (40, 40, 40), (70, 255, 255)) # green
    # ---------------------------------------------------------------------------------------------------------------------
    
    # Merge the masks to single mask
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_bg = cv2.bitwise_or(mask_blue, mask_green)
    mask = cv2.bitwise_or(mask_red, mask_bg)

    img_rgb = cv2.bitwise_and(crop, crop, mask = mask)

    median = cv2.medianBlur(img_rgb, 15)
    # obtain pixel BGR value in cropped
    color = median[60,25] #[y,x]

    if not os.path.exists('entrance.png'):
        cv2.imwrite('entrance.png', median)
    
    capture = cv2.imread('entrance.png')

    color_single = (int(color[0]) + int(color[1]) + int(color[2]))/3

    cv2.imshow("Entrance Chute", original)
    cv2.imshow("Colour Detection", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break    
        
    if color_single != black:
        new_puck_entered = 1
        statistics()

    if new_puck_entered == 1:

        new_puck_entered = 0; # set variable back to 0
        robot_move = 0;

        #robot will go grab puck
        sleep(0.5)
        intermediate_entry()
        sleep(0.5)
        
        entry()
        sleep(0.8)
        gripper('close')
        sleep(0.4)
        cmd('shoulder', 1200, spd) # pull back when lifting puck to prevent collision with ramp
        intermediate_entry();

        #robot pucks puck on process station
        sleep(0.5)
        intermediate_process()
        sleep(0.8)
        process()
        sleep(1)
        gripper('open')
        sleep(0.4)
        cmd('shoulder', 1450, 1200) # pull back when opening to prevent collision with the puck
        intermediate_process()
        sleep(0.5)
        
        while full_rotation < 2:

            for x in range(2):
                _, frame = video.read()
                _, original = video2.read()
               # sleep(0.1)

                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # color masks ---------------------------------------------------------------------------------------------------------
                mask_red1 = cv2.inRange(img_hsv, (0,115,115), (40,255,255)) # upper red
                mask_red2 = cv2.inRange(img_hsv, (160,115,115), (180,255,255)) # lower red               
                mask_blue = cv2.inRange(img_hsv, (100, 100, 40), (160,255,255)) # blue
                mask_green = cv2.inRange(img_hsv, (40, 80, 40), (84, 255, 255)) # green
                #------------------------------------------------------------------------------------------------------------------------
                mask_red = cv2.bitwise_or(mask_red1, mask_red2) # merge the upper and lower red mask

                img_red = cv2.bitwise_and(frame, frame, mask = mask_red)
                img_green = cv2.bitwise_and(frame, frame, mask = mask_green)
                img_blue = cv2.bitwise_and(frame, frame, mask = mask_blue)

                median_red = cv2.medianBlur(img_red, 15)
                gray_r = cv2.cvtColor(median_red,cv2.COLOR_BGR2GRAY)
                lab_r = cv2.cvtColor(median_red, cv2.COLOR_BGR2LAB)
                thresh_r = cv2.threshold(gray_r, 60, 255, cv2.THRESH_BINARY)[1]

                median_green = cv2.medianBlur(img_green, 15)
                gray_g = cv2.cvtColor(median_green,cv2.COLOR_BGR2GRAY)
                lab_g = cv2.cvtColor(median_green, cv2.COLOR_BGR2LAB)
                thresh_g = cv2.threshold(gray_g, 60, 255, cv2.THRESH_BINARY)[1]

                median_blue = cv2.medianBlur(img_blue, 15)
                gray_b = cv2.cvtColor(median_blue,cv2.COLOR_BGR2GRAY)
                lab_b = cv2.cvtColor(median_blue, cv2.COLOR_BGR2LAB)
                thresh_b = cv2.threshold(gray_b, 60, 255, cv2.THRESH_BINARY)[1]

                contours_r = cv2.findContours(thresh_r.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_g = cv2.findContours(thresh_g.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_b = cv2.findContours(thresh_b.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contours_r = imutils.grab_contours(contours_r)
                contours_g = imutils.grab_contours(contours_g)
                contours_b = imutils.grab_contours(contours_b)

                # initialize the sahpe detector and color labeler
                cl = ColorLabeler()

                for c in contours_r:
                        if cv2.contourArea(c) > 0:
                                M = cv2.moments(c)
                                red_label = cl.label(lab_r, c) # label color of object
                                c = c.astype("int") # change to type int
                                text = "{}".format(red_label)

                                cv2.drawContours(median_red, [c], -1, (0, 255, 0), 2)
                                cv2.drawContours(median_red, [c], -1, (0,255,0), 3)

                                if red_label == "red":  
                                        red_detector = 1
                                        robot_move = 1
                                        robot_text = "red"
                                        if blue_detector == 1 or green_detector == 1:
                                                defect_counter += 1
                                                blue_detector = 0
                                                green_detector = 0
                for d in contours_b:
                        if cv2.contourArea(d) > 0:
                                M = cv2.moments(d)
                                blue_label = cl.label(lab_b, d)
                                d = d.astype("int")
                                text = "{}".format(blue_label)

                                cv2.drawContours(median_blue, [d], -1, (0, 255, 0), 2)
                                cv2.drawContours(median_blue, [d], -1, (0,255,0), 3)

                                if blue_label == "blue":
                                        blue_detector = 1
                                        robot_move = 2
                                        robot_text = "blue"
                                        if red_detector == 1 or green_detector == 1:
                                                defect_counter += 1
                                                red_detector = 0
                                                green_detector = 0
                for e in contours_g:
                        if cv2.contourArea(e) > 0:
                                M = cv2.moments(e)
                                green_label = cl.label(lab_g, e)
                                e = e.astype("int")
                                text = "{}".format(green_label)

                                cv2.drawContours(median_green, [e], -1, (0, 255, 0), 2)
                                cv2.drawContours(median_green, [e], -1, (0,255,0), 3)

                                if green_label == "green":
                                        green_detector = 1
                                        robot_move = 3
                                        robot_text = "green"
                                        if blue_detector == 1 or red_detector == 1:
                                                defect_counter += 1
                                                red_detector = 0
                                                blue_detector = 0
                                                
                if defect_counter >= 4:
                    robot_move = 4
                    robot_text = "defect"
                    defect_counter = 0
                    full_rotation = 2   
               
                cv2.putText(frame, robot_text, (500,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
                cv2.imshow("Colour Detection", frame)
                cv2.imshow("Entrance Chute",original)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break     

            if full_rotation ==0:
                motor() #rotate motor by 90 degrees
            full_rotation +=1

        process()
        sleep(1)
        gripper('close')
        sleep(0.4)
        intermediate_process_2()
        sleep(0.1)
        intermediate_process()
        sleep(0.5)

        if robot_move == 1:
            intermediate_red_chute()
            sleep(0.5)
            red_chute()
            sleep(0.5)
            gripper('open')
            intermediate_red_chute()
            red +=1

        elif robot_move == 2:
            intermediate_blue_chute()
            sleep(0.5)
            blue_chute()
            sleep(0.5)
            gripper('open')
            intermediate_blue_chute()
            blue +=1

        elif robot_move == 3:
            intermediate_green_chute()
            sleep(0.5)
            green_chute()
            sleep(0.5)
            gripper('open')
            intermediate_green_chute()
            green +=1

        else: #defect
            intermediate_defect_chute()
            sleep(0.6)
            defect_chute()
            sleep(0.5)
            gripper('open')
            intermediate_defect_chute()
            defect +=1

        statistics()

video.release()
cv2.destroyAllWindows()
