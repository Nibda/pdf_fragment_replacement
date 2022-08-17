import os
import time
import fitz
import cv2 as cv
import numpy as np

start_prog = time.time()

# credentials
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS = "Червень01-15.06.2022" + os.sep
TEMPLATES = "Templates"+ os.sep
RESULT_DIR = DOCUMENTS + os.sep + "RESULTS" + os.sep
# bound source pattern with alternative
patterns_img = {('pattern1.png', 'alter_1img.png'), ('pattern2.png', 'alter_2img.png'), ('pattern3.png', 'alter_3img.png')}
patterns_text = {('pattern1.png', 'alter_1txt.png'), ('pattern2.png', 'alter_2txt.png'), ('pattern3.png', 'alter_3txt.png')}
print('Process: ', end='')


def get_file() -> str:
    '''Crowling by dir and return filename without path
    Return: filenames -> generator'''
    for file in os.listdir(DOCUMENTS):
        if file.endswith(".pdf"):
            yield file


count = 0
warning = []
for pdf_file in get_file():
    doc = fitz.open(DOCUMENTS + pdf_file)
    # get picture from pdf
    page = doc.load_page(0)
    pil_image = page.get_pixmap(dpi=200)
    byte_img = pil_image.pil_tobytes('png')
    np_image = np.frombuffer(byte_img, np.uint8)
    doc_img = cv.imdecode(np_image, 0)
    # determines inner format of document by size of file 
    if  os.path.getsize(DOCUMENTS + pdf_file) < 300000:
        pattern_files = patterns_text
    else:
        pattern_files = patterns_img

    # replace templates
    for filenames in pattern_files:
        tik = "|"
        # get template
        template = cv.imread(TEMPLATES + filenames[0], 0)
        w, h = template.shape[::-1]
        res = cv.matchTemplate(doc_img, template, cv.TM_CCOEFF_NORMED)
        # check if the template matches (0.8 is threshold)
        if np.amax(res) < 0.8:
            warning.append(f'Template \"{filenames[0]}\" missmatched with source: \"{doc.name}\"\n')
            pdf_file = "$" + pdf_file
            tik = "0"
            break
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        
        # mark found fragment by rectangle
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(doc_img, top_left, bottom_right, 255, -1)

        # get alter
        alter = cv.imread(TEMPLATES + filenames[1], 0)
        # define POI
        h_top = top_left[1]
        h_bottom = h_top + alter.shape[::-1][1]
        w_left = top_left[0]
        w_right = w_left + alter.shape[::-1][0]
        # insert alterimage in в POI
        doc_img[h_top : h_bottom, w_left : w_right] = alter

    count += 1
    #save results
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    res, im_png = cv.imencode('.png', doc_img)
    bytes_img = im_png.tobytes()
    doc.delete_page(0)
    doc.new_page()
    page = doc.load_page(0)
    page.insert_image(page.rect, stream=bytes_img)
    doc.save(RESULT_DIR + pdf_file)
    print(tik, end='')

print('\nTime: ', round(time.time() - start_prog), 'sec')
print('Total files: ', count)
print('Problem moments: ', len(warning))
for i in warning:
    print(i)
input("=== Presss ENTER to exit the program ===")