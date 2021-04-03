import cv2
from google.cloud import vision
import io
import numpy
import json
from PIL import Image, ImageDraw
import datetime


GOOGLE_VISION_API_KEY_PATH = "C:/Users/HP/.google-cloud/my-key.json"


def histogram_equalization(input_img_path, histogram_equalized_img_path):
    rgb_img = cv2.imread(input_img_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(histogram_equalized_img_path, equalized_img)


def calculate_chroma(input_img_path, color_detected_img_path):
    input_img = cv2.imread(input_img_path)

    for column in input_img:
        for pixel in column:
            min_color = min(pixel[0], pixel[1], pixel[2])
            max_color = max(pixel[0], pixel[1], pixel[2])
            pixel[0] = max_color - min_color
            pixel[1] = pixel[0]
            pixel[2] = pixel[0]

    cv2.imwrite(color_detected_img_path, input_img)


def otsu_thresholding(input_img_path, threshold_img_path):
    input_img = cv2.imread(input_img_path, 0)

    ret, threshold_img = cv2.threshold(input_img, 0, 255, cv2.THRESH_OTSU)

    cv2.imwrite(threshold_img_path, threshold_img)


def detect_highlight(input_img_path):
    histogram_equalized_img_path = "o_01_histogram_equalized.png"
    histogram_equalization(input_img_path, histogram_equalized_img_path)
    color_detected_img_path = "o_02_color_detected.png"
    calculate_chroma(histogram_equalized_img_path, color_detected_img_path)
    threshold_img_path = "o_03_threshold.png"
    otsu_thresholding(color_detected_img_path, threshold_img_path)

    print("highlight mask is ready")
    return threshold_img_path


def run_google_text_detection(input_img_path):
    with io.open(input_img_path, 'rb') as img_file:
        input_img_file = img_file.read()

    input_img = vision.Image(content=input_img_file)
    client = vision.ImageAnnotatorClient.from_service_account_json(GOOGLE_VISION_API_KEY_PATH)
    return client.document_text_detection(image=input_img)


def get_google_api_response(input_img_path):
    google_api_response = run_google_text_detection(input_img_path)

    google_api_response_json = vision.AnnotateImageResponse.to_json(google_api_response)
    google_api_response_dump_path = "o_04_google_api_response.txt"
    with open(google_api_response_dump_path, 'w') as json_file:
        json.dump(google_api_response_json, json_file)

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


def get_highlighted_word_objects(highlight_mask_path, google_word_objects):
    highlight_mask = cv2.imread(highlight_mask_path)

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
    # input_img_path = "highlighted-bengali-sample.jpg"
    input_img_path = "highlighted-english-sample.jpg"

    highlight_mask_path = detect_highlight(input_img_path)

    google_api_response = get_google_api_response(input_img_path)
    google_word_objects = get_all_word_objects(google_api_response)
    visualize_detected_word_boundaries(google_word_objects, input_img_path, "o_05_all_word_marked.png")

    highlighted_google_word_objects = get_highlighted_word_objects(highlight_mask_path, google_word_objects)

    show_result(highlighted_google_word_objects, input_img_path)

    print("process successful!")


if __name__ == '__main__':
    t1 = datetime.datetime.now()
    main()
    t2 = datetime.datetime.now()
    dt = t2 - t1
    print("elapsed time: ", dt.seconds, "s")
