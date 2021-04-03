import cv2
from google.cloud import vision
import io
import numpy
import json
from PIL import Image, ImageDraw
import datetime
from os import walk, path
import sys


GOOGLE_VISION_API_KEY_PATH = "C:/Users/HP/.google-cloud/my-key.json"


def get_directory_path_from_command_argument():
    if len(sys.argv) < 2:
        print("argument list is empty")
        return ""

    directory_path = sys.argv[1]
    if not path.isdir(directory_path):
        print("provided argument is not a valid directory")
        return ""

    print("directory_path: " + directory_path)
    return directory_path


def histogram_equalization(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite("o_01_histogram_equalized.png", equalized_img)
    return equalized_img


def calculate_chroma(rgb_img):
    rows, cols = rgb_img.shape[:2]
    color_detected_img = numpy.zeros((rows, cols, 1), numpy.uint8)

    for r in range(rows):
        for c in range(cols):
            pixel = rgb_img[r, c]
            min_color = min(pixel[0], pixel[1], pixel[2])
            max_color = max(pixel[0], pixel[1], pixel[2])
            color_detected_img[r, c] = max_color - min_color

    # cv2.imwrite("o_02_color_detected.png", color_detected_img)
    return color_detected_img


def otsu_thresholding(grayscale_img):
    ret, threshold_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_OTSU)

    # cv2.imwrite("o_03_threshold.png", threshold_img)
    return threshold_img


def detect_highlight(input_img_path):
    rgb_img = cv2.imread(input_img_path)
    equalized_img = histogram_equalization(rgb_img)
    color_detected_img = calculate_chroma(equalized_img)
    threshold_img = otsu_thresholding(color_detected_img)

    print("highlight mask is ready")
    return threshold_img


def run_google_text_detection(input_img_path):
    with io.open(input_img_path, 'rb') as img_file:
        input_img_file = img_file.read()

    input_img = vision.Image(content=input_img_file)
    client = vision.ImageAnnotatorClient.from_service_account_json(GOOGLE_VISION_API_KEY_PATH)
    return client.document_text_detection(image=input_img)


def get_google_api_response(input_img_path):
    google_api_response = run_google_text_detection(input_img_path)

    # google_api_response_json = vision.AnnotateImageResponse.to_json(google_api_response)
    # google_api_response_dump_path = "o_04_google_api_response.txt"
    # with open(google_api_response_dump_path, 'w') as json_file:
    #     json.dump(google_api_response_json, json_file)

    # with open(google_api_response_dump_path, 'r') as json_file:
    #     google_api_response_json = json.load(json_file)
    # print(google_api_response_json)

    # google_api_response = vision.AnnotateImageResponse.from_json(google_api_response_json)
    print("google text detection result received")

    return google_api_response


def get_all_word_objects(google_api_response):
    google_word_objects = []

    for text_annotation in google_api_response.text_annotations:
        if ' ' not in text_annotation.description:
            google_word_objects.append(text_annotation)

    return google_word_objects


def is_word_highlighted(highlight_mask, word_bounding_poly):
    word_mask = numpy.zeros((highlight_mask.shape[0], highlight_mask.shape[1]))
    cv2.fillConvexPoly(word_mask, word_bounding_poly, 255)

    word_part_of_highlight_mask = highlight_mask[word_mask > 0]
    if word_part_of_highlight_mask.size < 1:
        return False

    only_highlight_part_count = numpy.count_nonzero(word_part_of_highlight_mask > 0)
    only_highlight_part_ratio = only_highlight_part_count / word_part_of_highlight_mask.size

    # if at least 30% of the word is detected to be highlighted,
    # then consider the whole word to be highlighted
    if only_highlight_part_ratio > 0.3:
        return True
    else:
        return False


def get_highlighted_word_objects(highlight_mask, google_word_objects):
    highlighted_google_word_objects = []

    for google_word_object in google_word_objects:
        poly = google_word_object.bounding_poly.vertices
        word_bounding_poly = numpy.array([[poly[0].x, poly[0].y],
                                          [poly[1].x, poly[1].y],
                                          [poly[2].x, poly[2].y],
                                          [poly[3].x, poly[3].y]])

        if is_word_highlighted(highlight_mask, word_bounding_poly):
            highlighted_google_word_objects.append(google_word_object)

    print("filtered highlighted word objects")
    return highlighted_google_word_objects


def visualize_detected_word_boundaries(google_word_objects, input_img_path, word_marked_img_path):
    input_img = Image.open(input_img_path)
    draw = ImageDraw.Draw(input_img)

    for google_word_object in google_word_objects:
        poly = google_word_object.bounding_poly.vertices
        draw.polygon([poly[0].x, poly[0].y,
                      poly[1].x, poly[1].y,
                      poly[2].x, poly[2].y,
                      poly[3].x, poly[3].y], None, 'red')

    # input_img.show()
    input_img.save(word_marked_img_path)


def dump_text(google_word_objects, dump_file_path):
    text = ""
    for google_word_object in google_word_objects:
        text += google_word_object.description + " "

    with open(dump_file_path, 'w', encoding="utf-8") as dump_file:
        dump_file.write(text)


def show_result(highlighted_google_word_objects, input_img_path):
    word_marked_img_path = "o_06_highlighted_word_marked.png"
    visualize_detected_word_boundaries(highlighted_google_word_objects, input_img_path, word_marked_img_path)

    highlighted_text_file_path = "o_07_highlighted_text.txt"
    dump_text(highlighted_google_word_objects, highlighted_text_file_path)

    print("result is shown")


def main():
    input_img_dir_path = get_directory_path_from_command_argument()
    if input_img_dir_path == "":
        return

    _, _, files = next(walk(input_img_dir_path))

    for file in files:
        input_img_path = input_img_dir_path + "/" + file
        print("# processing- " + input_img_path)

        highlight_mask = detect_highlight(input_img_path)

        google_api_response = get_google_api_response(input_img_path)
        google_word_objects = get_all_word_objects(google_api_response)
        # visualize_detected_word_boundaries(google_word_objects, input_img_path, "o_05_all_word_marked.png")

        highlighted_google_word_objects = get_highlighted_word_objects(highlight_mask, google_word_objects)
        # show_result(highlighted_google_word_objects, input_img_path)
        dump_text(highlighted_google_word_objects, input_img_path + ".txt")

    print("## task complete!")


if __name__ == '__main__':
    t1 = datetime.datetime.now()
    main()
    t2 = datetime.datetime.now()
    dt = t2 - t1
    print("elapsed time: ", dt.seconds, "s")
