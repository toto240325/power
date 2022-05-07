#
# after a lot of trial and errors :
# 3 ways to use tesseract :
# 1. calling the shell tesseract command via subprocess.run
# 2. calling pytesseract on the original image
# 3. calling pytesseract on the original image which has been extended with a row of white pixel on the top and on the bottom
# and then we can apply those 3 methods on either the original image or on an optimised image
# (for instance after having "eroded" the characters)
# the last one (method 3 on eroded) seems to work the most often, but not always

# cd /opt/db
# sqlite3 /opt/db/mydatabase.db
# select time,text from events where categ='power_day' and time<"2022-04-21";
# select text from events where categ="power_night" and text like '7%';
# select time, text from events where categ="power_day" and text = "7.9"; # and time = "2022-04-22 15:10:44";
# select id, time, text from events where categ="power_night" and text > "61000" and time > "2022-04-24 13:00";
# update events set text = "65861.4"  where id=300646;
# select * from events where id=300769;

# # import modules and packages are searched for in sys.path :
# import sys
# type(sys.path)
# for path in sys.path:
#   print(path)

# Import packages
import os
import sys
import platform
import time
import datetime
import subprocess
import logging

import numpy as np
import pytesseract
import cv2
import shlex

from event import create_event
from event import read_where
import utils
import params
# from audioop import add

calib_day_x = params.calib_day_x
calib_day_y = params.calib_day_y
calib_day_width = params.calib_day_width
calib_day_height = params.calib_day_height

calib_night_x = params.calib_night_x
calib_night_y = params.calib_night_y
calib_night_width = params.calib_night_width
calib_night_height = params.calib_night_height

calib_day_dec_x = params.calib_day_dec_x
calib_day_dec_y = params.calib_day_dec_y
calib_day_dec_width = params.calib_day_dec_width
calib_day_dec_height = params.calib_day_dec_height

calib_night_dec_x = params.calib_night_dec_x
calib_night_dec_y = params.calib_night_dec_y
calib_night_dec_width = params.calib_night_dec_width
calib_night_dec_height = params.calib_night_dec_height

# export DISPLAY=localhost:xx.0

os.environ["DISPLAY"] = "localhost:10.0"
print(os.environ["DISPLAY"] +
      " (don't forget to run an Xterm on your laptop and set DISPLAY to the right value (for Ubuntu2 !))")


def set_calibration(img, x, y, width, height):
    """"
    allows to move a rectangle on top of a given image and returns the x,y coordinates of the top left corner 
    of the rectangle
    """
    dist = 10  # distance (in pixels) to move the rectangle with each move
    mode = "P"   # P: arrows change position    S: arrows change size
    window_name = "with rectangle"
    flags = cv2.WINDOW_NORMAL
    # flags = cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(window_name, flags)
    while True:
        img2 = np.copy(img)
        mytext = f'({x},{y}) width:{width} height:{height} - dist (+/-) : {dist} - Mode:{mode}'
        cv2.putText(img=img2, text=mytext, org=(
            50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

        cv2.rectangle(img2, (x, y), (x + width, y + height), (255, 0, 0), 2)
        cv2.imshow(window_name, img2)
        k = cv2.waitKey(0)

        # print(k)

        mySystem = platform.system()
        if mySystem == "Windows":
            esc = 27
            up = 2490368
            down = 2621440
            left = 2424832
            right = 2555904
            plus = 43
            minus = 45
        if mySystem == "Linux":
            esc = 27
            up = 82
            down = 84
            left = 81
            right = 83
            plus = 43
            minus = 45

        if k == esc:
            break
        elif k == -1:  # normally -1 returned,so don't print it
            continue
        elif k == up:
            if mode == "P": y -= dist
            else: height -= dist
        elif k == down:
            if mode == "P": y += dist
            else: height += dist
        elif k == left:
            if mode == "P": x -= dist
            else: width -= dist
        elif k == right:
            if mode == "P": x += dist
            else: width += dist
        elif k == plus:
            dist += 1
        elif k == minus:
            dist -= 1
        elif k == ord("m"):
            mode = "P" if mode == "V" else "V"
        else:
            print(k)  # else print its value

    cv2.destroyAllWindows()

    return x, y, width, height



# def set_calibration_power(img, x, y, width, height):
#     """"
#     allows to move a rectangle on top of a given image and returns the x,y coordinates of the top left corner 
#     of the rectangle
#     """
#     dist = 3  # distance (in pixels) to move the rectangle with each move
#     while True:
#         img2 = np.copy(img)
#         cv2.putText(img=img2, text=f'Hello {y} - dist : {dist}', org=(
#             50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 255, 0), thickness=1)

#         cv2.rectangle(img2, (x, y), (x + width, y + height), (255, 0, 0), 2)
#         cv2.imshow("with rectangle", img2)
#         k = cv2.waitKeyEx(0)

#         # print(k)

#         myenv = "Windows"
#         if myenv == "Windows":
#             esc = 27
#             up = 2490368
#             down = 2621440
#             left = 2424832
#             right = 2555904
#             plus = 43
#             minus = 45
#         if myenv == "Linux":
#             esc = 27
#             up = 82
#             down = 84
#             left = 81
#             right = 83
#             # plus = 43
#             # minus = 45

#         if k == esc:
#             break
#         elif k == -1:  # normally -1 returned,so don't print it
#             continue
#         elif k == up:
#             y -= dist
#         elif k == down:
#             y += dist
#         elif k == left:
#             x -= dist
#         elif k == right:
#             x += dist
#         elif k == plus:
#             dist += 1
#         elif k == minus:
#             dist -= 1
#         else:
#             print(k)  # else print its value

#     return x, y



# def get_cam_footage(basename):
#     """
#     get 1 second of video from the chalet Webcam and put it in <basename>.h264
#     """
#     process = subprocess.run(
#         ['openRTSP', '-d', '1', '-V', '-F',
#             f'{basename}-', 'rtsp://admin:123456@192.168.0.91/'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         universal_newlines=True)
#     # print("rc = ", process.returncode)
#     # print("result = ", process.stdout)
#     # err = process.stderr
#     # # print("err = ", process.stderr)
#     # $ cp chalet-video-H264-1 a.h264
#     # $ vlc a.h264

#     # os.rename(f'{basename}-video-H264-1', f'{basename}.h264')
#     # os.remove(f'{basename}-audio-PCMA-2')

#     video_file = f'{basename}-video-H264-1'
#     if os.path.isfile(video_file):
#         os.rename(video_file, f'{basename}.h264')
    
#     audio_file = f'{basename}-audio-PCMA-2'
#     if os.path.isfile(audio_file):
#         os.remove(audio_file)



# def get_snapshot(basename):
#     """
#     extract a snapshot from <basename>.h264 and put it in <basename>.jpg
#     """
#     try_again = True
#     i = 0
#     max_iteration = 10
#     while try_again and i <= max_iteration:
#         # extra a picture from that video
#         process = subprocess.run(
#             ['ffmpeg', '-y', '-i', f'{basename}.h264',
#                 '-frames:v', '1', f'{basename}.jpg'],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True)
#         my_stdout = process.stdout
#         err = process.stderr
#         # print("----args = ", process.args)
#         # print("----rc = ", process.returncode)
#         # print("----stdout = ", my_stdout)
#         # print("----err = ", err)
        
#         # try_again = (
#         #     (err.find("Output file is empty") != -1) 
#         #     or
#         #     (err.find("Conversion failed!") != -1)
#         # )
#         # continue until a jpg is produced (which doesn't happen if the .h264 file is corrupted or empty)
#         try_again = (not os.path.isfile(f'{basename}.jpg'))

#         i += 1
#         time.sleep(1)

#     logging.info(f"nb_iteration : {i}")
#     if i > max_iteration:
#         logging.info("!!!!!!!! couldn't extract snapshot from footage !!!!!")
#     else:
#         os.rename(f'{basename}.h264', f'{basename}.bak.h264')

#     return i <= max_iteration


def get_cam_footage(basename):
    """
    get 1 second of video from the chalet Webcam and put it in <basename>.h264
    """
    process = subprocess.run(
        ['openRTSP', '-d', '1', '-V', '-F',
            f'{basename}-', 'rtsp://admin:123456@192.168.0.91/'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # # print("err = ", process.stderr)
    # $ cp chalet-video-H264-1 a.h264
    # $ vlc a.h264

    audio_file = f'{basename}-audio-PCMA-2'
    if os.path.isfile(audio_file):
        os.remove(audio_file)

    video_file = f'{basename}-video-H264-1'
    if os.path.isfile(video_file):
        footage_filename = f'{basename}.h264'
        os.rename(video_file, footage_filename)
    else:
        footage_filename = None
    return footage_filename


def get_snapshot_old(footage_filename):
    """
    extract a snapshot from <basename>.h264 and put it in <basename>.jpg
    """

    if footage_filename == None:
        return None
    try_again = True
    i = 1
    max_iteration = 1
    basename_ext = os.path.basename(footage_filename)
    basename, ext = os.path.splitext(basename_ext)
    while try_again and i <= max_iteration:
        # extra a picture from that video
        process = subprocess.run(
            ['ffmpeg', '-y', '-i', f'{basename}.h264',
                '-frames:v', '1', f'{basename}.jpg'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
        my_stdout = process.stdout
        err = process.stderr
        # print("----args = ", process.args)
        # print("----rc = ", process.returncode)
        # print("----stdout = ", my_stdout)
        # print("----err = ", err)

        # try_again = (
        #     (err.find("Output file is empty") != -1)
        #     or
        #     (err.find("Conversion failed!") != -1)
        # )
        # continue until a jpg is produced (which doesn't happen if the .h264 file is corrupted or empty)
        try_again = (not os.path.isfile(f'{basename}.jpg'))

        if try_again: time.sleep(1)
        i += 1

    logging.info(f"nb_iteration : {i}")
    if i > max_iteration:
        logging.error("!!!!!!!! couldn't extract snapshot from footage !!!!!")
    else:
        os.rename(f'{basename}.h264', f'{basename}.bak.h264')

    if i <= max_iteration:
        extracted_img_filename = f'{basename}.jpg'
    else:
        extracted_img_filename = None

    return extracted_img_filename


def get_snapshot(footage_filename):
    """
    extract a snapshot from <basename>.h264 and put it in <basename>.jpg
    """

    if footage_filename == None:
        return None
    basename_ext = os.path.basename(footage_filename)
    basename, ext = os.path.splitext(basename_ext)
    # extract a picture from that video
    process = subprocess.run(
        ['ffmpeg', '-y', '-i', f'{basename}.h264',
            '-frames:v', '1', f'{basename}.jpg'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    my_stdout = process.stdout
    err = process.stderr
    # print("----args = ", process.args)
    # print("----rc = ", process.returncode)
    # print("----stdout = ", my_stdout)
    # print("----err = ", err)

    if os.path.isfile(f'{basename}.jpg'):
        os.rename(f'{basename}.h264', f'{basename}.bak.h264')
        extracted_img_filename = f'{basename}.jpg'
    else:
        extracted_img_filename = None

    return extracted_img_filename


def cropped_digits_img(filename):
    global interactive

    # read the snapshot
    img = cv2.imread(filename)
    # logging.info(img.shape) # Print image shape
    #if interactive: cv2.imshow("original", img)

    # convert to grey only
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("greyed", img)

    # invert image (black becomes white)
    img = (255-img)
    # if interactive: cv2.imshow("greyed inverted", img)  #; cv2.waitKey(0)

    # calib_x = 805
    # calib_width = 170
    # calib_day_y = 445
    # calib_day_height = 47
    # calib_night_y = calib_day_y+85
    # calib_night_height = 40
    # calib_dec_start = 172
    # calib_dec_width = 20
    # calib_night_decimal_height = 40-4

    # -------------
    # day figures
    # Crop the image to focus on the digits
    #img_day = img[445:492, 805:975]
    img_day = img[calib_day_y:calib_day_y +
                  calib_day_height, calib_day_x:calib_day_x+calib_day_width]
    # if interactive: cv2.imshow("cropped img_day", img_day) ; #cv2.waitKey(0)

    # # testing the best threshold
    # img_bck = np.copy(img_day)
    # for t in range(80, 180, 20):
    #     img_day = np.copy(img_bck)
    #     _, img_day = cv2.threshold(img_day, t, 255, cv2.THRESH_BINARY)
    #     if interactive: cv2.imshow(f"threshed {t}", img_day)
    # img_bck = np.copy(img_day)
    # cv2.waitKey(0)

    best_threshold = 100

    # thresholding to get a black/white picture
    _, img_day = cv2.threshold(img_day, best_threshold, 255, cv2.THRESH_BINARY)

    # -------------
    # day_decimal
    # Crop the image to focus on the digits
    img_day_decimal = img[calib_day_dec_y:calib_day_dec_y+calib_day_dec_height,
                          calib_day_dec_x:calib_day_dec_x+calib_day_dec_width]
    # if interactive: cv2.imshow("cropped img_day_dec", img_day_dec_decimal) ; cv2.waitKey(0)

    # thresholding to get a black/white picture
    _, img_day_decimal = cv2.threshold(img_day_decimal, best_threshold, 255, cv2.THRESH_BINARY)
    # if interactive: cv2.imshow("threshed day", img_day_decimal); cv2.waitKey(0)

    # -------------
    # night figures
    # Crop the image to focus on the digits
    #img_night = img[530:570, 805:975]
    img_night = img[calib_night_y:calib_night_y +
                    calib_night_height, calib_night_x:calib_night_x+calib_night_width]
    # if interactive: cv2.imshow("cropped", img_night)  #; cv2.waitKey(0)

    # thresholding to get a black/white picture
    _, img_night = cv2.threshold(img_night, best_threshold, 255, cv2.THRESH_BINARY)
    # if interactive: cv2.imshow("threshed img_night", img_night)  #; cv2.waitKey(0)

    # -------------
    # night_decimal
    # Crop the image to focus on the digits
    img_night_decimal = img[calib_night_dec_y:calib_night_dec_y+calib_night_dec_height,
                            calib_night_dec_x:calib_night_dec_x+calib_night_dec_width]
    # if interactive: cv2.imshow("cropped img_night", img_night_decimal) ; cv2.waitKey(0)

    # thresholding to get a black/white picture
    _, img_night_decimal = cv2.threshold(
        img_night_decimal, best_threshold, 255, cv2.THRESH_BINARY)
    # if interactive: cv2.imshow("threshed night", img_night_decimal); cv2.waitKey(0)

    # -------------

    return img_day, img_night, img_day_decimal, img_night_decimal


def get_digits(img_name, img, options_list):
    # reads digits from picture
    # if interactive: cv2.imshow("cropped digits", img)
    temp_filename = img_name+".jpg"
    temp_output_filename = "tmp_output.txt"
    cv2.imwrite(temp_filename, img)

    # print("shell tesseract options: ", options_list)
    process = subprocess.run(
        #['tesseract', '-c', 'page_separator=', temp_filename, temp_output_filename] + options_list,
        ['tesseract', '-c', 'page_separator=',
            temp_filename, 'stdout'] + options_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    # print("args = ", process.args)
    # print("rc = ", process.returncode)
    # print("result = ", process.stdout)
    # err = process.stderr
    # print("err = ", process.stderr)
    return process.stdout.strip()


def check_digits(st):
    """
    checks if string st contains 3 digits, then a space, then 3 digits, and 
    return the corresponding int values (day and night) if that's the case, or None otherwise
    """
    st = st.strip()
    day = None
    night = None
    if len(st) == 7 and st[0:3].isnumeric and st[-3:].isnumeric:
        day = int(st[0:3])/100.0
        night = int(st[-3:])
    return day, night


def last_validated_value(categ):
    """
    get from the DB the last value of category categ
    """
    now1 = datetime.datetime.now()
    nowStr = now1.strftime('%Y-%m-%d %H:%M:%S')
    res = read_where(categ, 1, "")
    error = res["error"]
    short_error_msg = ""
    long_error_msg = ""
    validated_value = None

    if (error != ""):
        short_error_msg = "events server unresponsive !!!"
        long_error_msg(f"!!!! Error : could not read the last {categ} event - {error}")

    # check that date is OK
    if short_error_msg == "":
        event = res["events"][0]
        last_event_date = event["time"]
        try:
            last_event_Datetime = datetime.datetime.strptime(
                last_event_date, '%Y-%m-%d %H:%M:%S')
        except Exception as error:
            short_error_msg = "event date is not a valid date !!!"
            long_error_msg = f"date of last {categ} is not a date : {last_event_date} ! - {str(error)}"
            # logging.error(long_error_msg)

    # then check if the value we got is a valid float
    if short_error_msg == "":
        if event["text"].isnumeric:
            validated_value = float(event["text"])
        else:
            validated_value = None
            short_error_msg = "{categ} value is not a numeric value - {nowStr}"
            long_error_msg = "{categ} value is not a numeric value"

    if short_error_msg != "":
        user_name = params.mailer
        passwd = params.mailer_pw
        from_email = params.from_email
        to_email = params.to_email
        subject = short_error_msg + "- " + nowStr
        body = long_error_msg
        htmlbody = None
        myfilename = None
        utils.mySend(user_name, passwd, from_email,
                          to_email, subject, body, htmlbody, myfilename)

    return validated_value


def get_best_result(candidate_results, img, kind, optional_non_decimal_part):
    """
    results is an array of [label_str, result_str] (ex: ["tesseract optimised","743 423"])
    - create a list with only the valid results (format must be "999 999", after having removed any dot ("."))
    - if there are no result
        store the problematic image for later analysis in issues/noresult-<datetime>
        return None,None
    - if there are several valid results :
        - if all of them are the same :
            return day,night
        - if there are different results :
            store the problematic image for later analysis in issues/ambiguous-<datetime>
            return first day,night in the list

    - kind is the kind of image at stake : "day", "night", "day_decimal", "night_decimal"
    """

    x = datetime.datetime.now()
    now_str = x.strftime("%Y-%m-%d_%H-%M-%S")

    # create a list of valid results only
    valid_results = []

    # manual_mode = True (in parameters.py) if the figures are reset/checked manually after an interruption of the normal series

    for c in candidate_results:
        if interactive: print(f'{c[0]:35}: {c[1]}')
        st = c[1].strip().replace(" ", "")
        st = st.strip().replace(".", "")
        if len(st) >= 1 and st != "." and st.isnumeric:
            number = int(st)
            # check the read figures make sense (sometimes a "7" is read as a "1" by tesseract)
            if kind == "day":
                # first get the last validated measure (the strong assumption is that we store only validated values in the DB !!)
                last_validated_val = last_validated_value("power_day")
                # if number > 71000 and number < 72000:
                if last_validated_val != None:
                    truncated = int(last_validated_val)
                    if (params.manual_mode and number >= params.manual_day and number <= params.manual_day+1) or (number >= int(last_validated_val)-1 and number <= last_validated_val+2):
                        valid_results.append(int(st))
            elif kind == "night":
                # first get the last validated measure (the strong assumption is that we store only validated values in the DB !!)
                last_validated_val = last_validated_value("power_night")
                # if number > 65000 and number < 67000:
                if last_validated_val != None:
                    if (params.manual_mode and number >= params.manual_night and number <= params.manual_night+1) or (number >= int(last_validated_val)-1 and number <= last_validated_val+2):
                        valid_results.append(int(st))
            elif kind == "day_decimal":
                # # first get the last validated measure (the strong assumption is that we store only validated values in the DB !!)
                last_validated_val = last_validated_value("power_day")
                # prev_decimal_part = round((last_validated_val % 1) * 10)
                if number >= 0 and number <= 9:
                    if optional_non_decimal_part != None:
                        candidate_full_value = optional_non_decimal_part + number/10
                        if candidate_full_value >= last_validated_val:
                            valid_results.append(int(st))
            elif kind == "night_decimal":
                # # first get the last validated measure (the strong assumption is that we store only validated values in the DB !!)
                last_validated_val = last_validated_value("power_night")
                # prev_decimal_part = round((last_validated_val % 1) * 10)
                if number >= 0 and number <= 9:
                    if optional_non_decimal_part != None:
                        candidate_full_value = optional_non_decimal_part + number/10
                        if candidate_full_value >= last_validated_val:
                            valid_results.append(int(st))

    # remove duplicates from list of valid results
    valid_results = list(dict.fromkeys(valid_results))

    issues_path = "issues/"
    if not os.path.isdir(issues_path):
        os.mkdir(issues_path)

    if len(valid_results) == 0:
        # no valid results; store image for later analysis
        best_candidate = None
        # print("No valid results !")
        # store image for later analysis :
        filename = issues_path + "noresult_decimal_" + now_str + ".jpg"
        cv2.imwrite(filename, img)
    else:
        # at least one valid result; first one is kept, unless we find another one which is closer to the last_validated_val
        best_candidate = valid_results[0]
        # if there were more than 1 valid result
        if len(valid_results) > 1:
            prev_delta = abs(best_candidate - last_validated_val)
            all_candidates = ""
            for candidate in valid_results:
                # find the delta between this candidate and the previously stored value in the DB
                delta = abs(candidate - last_validated_val)
                if delta < prev_delta:
                    best_candidate = candidate
                    prev_delta = delta
                    
                # accumulate in all_candidates a string with all the candidate values, for later analysis
                if all_candidates == "":
                    all_candidates = str(candidate)
                else:
                    all_candidates = all_candidates + "_" + str(candidate)
            logging.info(f'more than 1 valid result : {all_candidates}')
            # store image for later analysis :
            filename = issues_path + "ambiguous_" + all_candidates + "_" + now_str + ".jpg"
            cv2.imwrite(filename, img)

    return best_candidate


def optimise_img(img):
    """
    optimise the passed image by various methods, for instance by eroding the borders of the characters
    """

    # invert the image (because the erosion method I know work with white chars on black background images)
    # I am sure it can be made much better ;-)

    img = 255 - img
    # if interactive: cv2.imshow("inverted image", img)

    kernel = np.ones((5, 5), np.uint8)

    # kernel = np.array( [
    #     [ 0, 0, 0, 0, 0 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 1, 1, 1, 1, 1 ],
    #     [ 0, 0, 0, 0, 0 ]
    #     ],np.uint8)

    img = cv2.erode(img, kernel, iterations=2)
    # cv2.imwrite("base_eroded.jpg", img)
    # if interactive: cv2.imshow("eroded", img)

    # invert the image again, since done at the beginning
    img = 255 - img

    return img


def explain_tesseract(img, title, options_str, candidate_results):
    """
    explains the tesseract way of analysing this image by having boxes drawn around the characters
    """
    r, c = img.shape
    nb_lines = 40
    additional_lines = np.full((nb_lines, c), 255, dtype=np.uint8)

    # adding a blank rectangle above the image
    img2 = np.append(img, additional_lines, axis=0)
    img2 = np.append(additional_lines, img2, axis=0)

    # if interactive: cv2.imshow("img extended", img2)

    hImg, wImg = img.shape
    # print("pytesseract options", options_str)
    myres = pytesseract.image_to_string(img, config=options_str).strip()
    candidate_results.append([f'{title} (orig size)', myres])
    #if interactive: print("pytesseract (orig): ", myres)
    myres2 = pytesseract.image_to_string(img2, config=options_str).strip()
    candidate_results.append([f'{title} (extended)', myres2])
    #if interactive: print("pytesseract (extended): ", myres2)

    boxes = pytesseract.image_to_boxes(img, config=options_str)

    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        # print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img2, (x, hImg-y+nb_lines),
                      (w, hImg-h+nb_lines), (0, 255, 0), 2)
        cv2.putText(img2, b[0], (x, hImg-y+25+nb_lines),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # if interactive: cv2.imshow('Image with boxes', img2)
    # cv2.waitKey(0)


def write_gray_to_file(img_name, img):
    """
    takes a gray image and write it to disk
    """
    img_to_save = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(img_name + ".jpg", img_to_save)


def collect_candidate_results(img, kind, basename):
    """
    extract from img, of given kind ("day", "night", etc) all the possible candidates as 
    read string of numerical digits
    """
    # NB : shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_str = "--psm 13 -c tessedit_char_whitelist='.0123456789 '"
    #options_str="--psm 6 -c tessedit_char_whitelist='.0123456789 '"
    # shlex.split('tesseract -c page_separator="" cropped_chalet.jpg stdout --psm 13')
    options_list = shlex.split(options_str)

    candidate_results = []
    img_name = basename+'_' + kind + '_cropped'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # # read it again to check
    # img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if interactive: cv2.imshow("cropped digits", img); cv2.waitKey

    # extract the figures from this plain image
    res1 = get_digits(img_name, img, options_list)
    candidate_results.append([kind + " tess. not optimised", res1])
    # if interactive: print("tesseract not optimised : ",res1)
    # if interactive: cv2.imshow("not optimised", img)
    explain_tesseract(img, kind + " pytess. not optimised",
                      options_str, candidate_results)

    # try to optimise the image
    img = optimise_img(img)
    img_name = basename+'_' + kind + '_optimised'
    #if interactive: cv2.imshow(img_name, img)
    # save a copy of this plain image for later analysis
    write_gray_to_file(img_name, img)

    # extract the figures from this optimised image
    res2 = get_digits(img_name, img, options_list)
    candidate_results.append([kind + " tess. optimised", res2])
    # if interactive: print("tesseract  optimised : ",res1)
    # if interactive: cv2.imshow("optimised", img)
    explain_tesseract(img, kind + " pytess. optimised",
                      options_str, candidate_results)
    return candidate_results


def display_candidate_results(candidate_results):
    for c in candidate_results:
        if interactive:
            print(f'{c[0]:35}: {c[1]}')


def check_power():
    global candidate_results
    global interactive

    basename = "power_base"

    successful = False
    i = 1
    max_iteration = 10
    while not successful and i <= max_iteration:
        footage_filename = get_cam_footage("tmp_"+basename)
        if footage_filename != None:
            successful = get_snapshot("tmp_"+basename)
        else:
            logging.error(f'iter {i} : failed to get footage')
        if not successful:
            logging.error(f'iter {i} : failed to get snapshot out of footage')
        i+=1

    if successful:
        debug = False
        if debug:
            filename = "threshed_chalet1.jpg"
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            filename = "tmp_"+basename+'.jpg'
            filename_bak = "tmp_"+basename+'.bak.jpg'
            img_day, img_night, img_day_decimal, img_night_decimal = cropped_digits_img(
                filename)
            os.rename(filename, filename_bak)

        if interactive:
            print("")

        # ---- day ----------
        candidate_results = collect_candidate_results(img_day, "day", basename)
        if interactive:
            display_candidate_results(candidate_results)
        day = get_best_result(candidate_results, img_day, "day", None)
        if interactive:
            print("")

        # ---- day decimal part ----------
        candidate_results = collect_candidate_results(
            img_day_decimal, "day_decimal", basename)
        if interactive:
            display_candidate_results(candidate_results)
        # it makes sense to try and identify the decimal part only if a non-decimal part has been found
        if day != None:
            day_decimal = get_best_result(
                candidate_results, img_day_decimal, "day_decimal", day)
            # if interactive: print("day_decimal : ", day_decimal)

        if interactive:
            print("----------------------")

        # ---- night ----------
        candidate_results = collect_candidate_results(
            img_night, "night", basename)
        if interactive:
            display_candidate_results(candidate_results)
        night = get_best_result(candidate_results, img_night, "night", None)
        if interactive:
            print("")

        # ---- night decimal part ----------
        candidate_results = collect_candidate_results(
            img_night_decimal, "night_decimal", basename)
        if interactive:
            display_candidate_results(candidate_results)
        if night != None:
            # it makes sense to try and identify the decimal part only if a non-decimal part has been found
            night_decimal = get_best_result(
                candidate_results, img_night_decimal, "night_decimal", night)
            # if interactive: print("night_decimal : ", night_decimal)

        if interactive:
            print("")

        # --------------------------

        if day != None:
            if day_decimal != None:
                day = day + day_decimal/10
            create_event("power_day", str(day))

        if night != None:
            if night_decimal != None:
                night = night + night_decimal/10
            create_event("power_night", str(night))

        # if night != None:
        #     create_event("power_night",str(night))

        if interactive:
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        day, night = None, None

    return day, night


def calibration_power_day():

    global calib_day_x, calib_day_y, calib_day_width, calib_day_height
    global calib_night_x, calib_night_y, calib_night_width, calib_night_height
    global calib_day_dec_x, calib_day_dec_y, calib_day_dec_width, calib_day_dec_height
    global calib_night_dec_x, calib_night_dec_y, calib_night_dec_width, calib_night_dec_height

    basename = "power"
    footage_filename = get_cam_footage(basename)
    img_filename = get_snapshot(footage_filename)
    img = None
    if img_filename != None and os.path.isfile(img_filename):
        img = cv2.imread(img_filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if isinstance(img,np.ndarray) and img.any() != None:

        basename = "power_day_base"
        calib_day_x, calib_day_y, calib_day_width, calib_day_height = set_calibration(
            img, calib_day_x, calib_day_y, calib_day_width, calib_day_height)
        utils.replace_param("params.py", "calib_day_x", calib_day_x)
        utils.replace_param("params.py", "calib_day_y", calib_day_y)
        utils.replace_param("params.py", "calib_day_width", calib_day_width)
        utils.replace_param("params.py", "calib_day_height", calib_day_height)
        logging.info(
            f'day : x:{calib_day_x}, y:{calib_day_y}, width:{calib_day_width}, height:{calib_day_height}')

        basename = "power_night_base"
        calib_night_x, calib_night_y, calib_night_width, calib_night_height = set_calibration(
            img, calib_night_x, calib_night_y, calib_night_width, calib_night_height)
        utils.replace_param("params.py", "calib_night_x", calib_night_x)
        utils.replace_param("params.py", "calib_night_y", calib_night_y)
        utils.replace_param("params.py", "calib_night_width", calib_night_width)
        utils.replace_param("params.py", "calib_night_height", calib_night_height)
        logging.info(
            f'night : x:{calib_night_x}, y:{calib_night_y}, width:{calib_night_width}, height:{calib_night_height}')

        basename = "power_day_dec_base"
        calib_day_dec_x, calib_day_dec_y, calib_day_dec_width, calib_day_dec_height = set_calibration(
            img, calib_day_dec_x, calib_day_dec_y, calib_day_dec_width, calib_day_dec_height)
        utils.replace_param("params.py", "calib_day_dec_x", calib_day_dec_x)
        utils.replace_param("params.py", "calib_day_dec_y", calib_day_dec_y)
        utils.replace_param("params.py", "calib_day_dec_width", calib_day_dec_width)
        utils.replace_param("params.py", "calib_day_dec_height", calib_day_dec_height)
        logging.info(
            f'day_dec : x:{calib_day_dec_x}, y:{calib_day_dec_y}, width:{calib_day_dec_width}, height:{calib_day_dec_height}')

        basename = "power_night_dec_base"
        calib_night_dec_x, calib_night_dec_y, calib_night_dec_width, calib_night_dec_height = set_calibration(
            img, calib_night_dec_x, calib_night_dec_y, calib_night_dec_width, calib_night_dec_height)
        utils.replace_param("params.py", "calib_night_dec_x", calib_night_dec_x)
        utils.replace_param("params.py", "calib_night_dec_y", calib_night_dec_y)
        utils.replace_param("params.py", "calib_night_dec_width", calib_night_dec_width)
        utils.replace_param("params.py", "calib_night_dec_height", calib_night_dec_height)
        logging.info(
            f'night_dec : x:{calib_night_dec_x}, y:{calib_night_dec_y}, width:{calib_night_dec_width}, height:{calib_night_dec_height}')

    else:
        logging.error("Cannot calibrate because didn't get an image")

    # calib_day_x, calib_day_y = set_calibration(
    #     img, calib_day_x, calib_day_y, calib_day_width, calib_day_height)

    # logging.info(
    #     f'x:{calib_day_x}, y:{calib_day_y}, width:{calib_day_width}, height:{calib_day_height}')


def print_usage():
    print("Usage : ")
    print(" python power.py         : get the power figures and display them on stdout")
    # print(" python power.py where [[[ categ ] nb ] date_from ]     : prints the most recent ps4 powers")  
    print(" python power.py calib   : recalibrate the cropping of the image")  
    print(" python power.py anythingelse : print this usage")


def main():
    utils.init_logger('INFO')
    logging.info("------------------------------------------------------------")
    logging.info("Starting power")

    nb_args = len(sys.argv)
    logging.info(f'Number of arguments: {nb_args} arguments.')
    logging.info(f'Argument List: {str(sys.argv)}')
    if nb_args == 2:
        arg1 = sys.argv[1]
        logging.info(f"arg1 = {arg1}")
        if arg1 == "calib":
            calibration_power_day()
        else:
            print_usage()

    day, night = check_power()
    if day != None or night != None:
        logging.info(f'day : {day} - night : {night}')
    else:
        logging.info("Couldn't read valid figures !")

    logging.info("Ending power")
    utils.shutdown_logger()


interactive = False
calibration = False

if __name__ == '__main__':
    import getpass
    username = getpass.getuser()
    interactive = (username == "toto")
    calibration = True
    main()
