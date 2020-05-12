from __future__ import print_function

import os
import cv2
import numpy as np
# import argparse

# from keras.applications.imagenet_utils import preprocess_input
# from keras.models import load_model

from utils import RotNetDataGenerator, crop_largest_rectangle, angle_error, rotate


# def process_images(model, input_path, output_path,
#                    batch_size=64, crop=True):
#     extensions = ['.jpg', '.jpeg', '.bmp', '.png']

#     output_is_image = False
#     if os.path.isfile(input_path):
#         image_paths = [input_path]
#         if os.path.splitext(output_path)[1].lower() in extensions:
#             output_is_image = True
#             output_filename = output_path
#             output_path = os.path.dirname(output_filename)
#     else:
#         image_paths = [os.path.join(input_path, f)
#                        for f in os.listdir(input_path)
#                        if os.path.splitext(f)[1].lower() in extensions]
#         if os.path.splitext(output_path)[1].lower() in extensions:
#             print('Output must be a directory!')

#     grayscale_image_paths = []
#     for path in image_paths:
#         print(path)
#         print(os.path.dirname(path))
#         print(os.path.basename(path))
#         #grayscale_dir = os.path.join(os.path.dirname(path), '/_grayscale/', os.path.basename(path))
#         im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
#         # grayscale_filename = os.path.join(grayscale_dir, os.path.splitext(output_path)[0])
#         # grayscale_filename = os.path.join(grayscale_filename, os.path.splitext(output_path)[1])
#         grayscale_dir = os.path.dirname(path) + '_grayscale' #, os.path.basename(path))
#         if not os.path.exists(grayscale_dir):
#             os.makedirs(grayscale_dir)
#         # grayscale_filename = os.path.join(os.path.dirname(path), '_grayscale') #, os.path.basename(path))
#         # grayscale_filename = os.path.join(grayscale_filename, os.path.basename(path))
#         grayscale_filename = os.path.join(grayscale_dir, os.path.basename(path))
#         rotated_image = rotate(image, -predicted_angle)
#         cv2.imwrite(grayscale_filename, im_gray)
#         grayscale_image_paths.append(grayscale_filename)

#     print(grayscale_image_paths)

#     predictions = model.predict_generator(
#         RotNetDataGenerator(
#             #image_paths,
#             grayscale_image_paths,
#             input_shape=(224, 224, 3),
#             batch_size=64,
#             one_hot=True,
#             preprocess_func=preprocess_input,
#             rotate=False,
#             crop_largest_rect=True,
#             crop_center=True
#         ),
#         val_samples=len(grayscale_image_paths)
#         #al_samples=len(image_paths)
#     )

#     predicted_angles = np.argmax(predictions, axis=1)

#     if output_path == '':
#         output_path = '.'

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     for path, predicted_angle in zip(image_paths, predicted_angles):
#         image = cv2.imread(path)
#         rotated_image = rotate(image, -predicted_angle)
#         if crop:
#             size = (image.shape[0], image.shape[1])
#             rotated_image = crop_largest_rectangle(rotated_image, -predicted_angle, *size)
#         if not output_is_image:
#             output_filename = os.path.join(output_path, os.path.basename(path))
#         cv2.imwrite(output_filename, rotated_image)


if __name__ == '__main__':
    input_files = [ ('20200208_football_throw02_R02-[NISwGSP][2D][BLEND_LINEAR].png', -26),
                    ('20200208_football_throw02_R03-[NISwGSP][2D][BLEND_LINEAR].png', 0),
                    ('20200208_football_throw02_R04-[NISwGSP][2D][BLEND_LINEAR].png', -50),
                    ('20200208_football_throw02_R05-[NISwGSP][2D][BLEND_LINEAR].png', -27),
                    ('20200208_football_throw02_R06-[NISwGSP][2D][BLEND_LINEAR].png', -26),
                    ('20200208_football_throw02_R07-[NISwGSP][2D][BLEND_LINEAR].png', -45),
                    ('20200208_football_throw02_R08-[NISwGSP][2D][BLEND_LINEAR].png', -28),
     ]

    # rotation = 27

    input_dir = './images/20200208_football_throw02_ALL'
    out_dir = input_dir + '_out_train'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    


    for input_file, rotation in input_files:
        path = os.path.join(input_dir, input_file)
        output_filename = os.path.join(out_dir, input_file)

        image = cv2.imread(path)
        rotated_image = rotate(image, -rotation)
        cv2.imwrite(output_filename, rotated_image)

