import os
import cv2

def pre_processing(image_path, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # image list
    images = os.listdir(image_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for i, img in enumerate(images):
        img_dir = os.path.join(image_path, img)
        image = cv2.imread(img_dir, 0)
        # add dilated operation
        image_dilated = cv2.erode(image, kernel)

        # cv2.imshow('src_image', image)
        # cv2.imshow('dilate_image', image_dilated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # save image
        cv2.imwrite(save_path + '/' + img, image_dilated)

        if (i + 1) % 100 == 0:
            print('{} have saved successfully'.format(i + 1))
    print('_' * 50)
    print('All images have saved successfully!')

if __name__ == '__main__':
    image_path = 'E:/datasets/denoising/processing_data/blur_cliped_test'
    save_path = 'E:/datasets/denoising/processing_data/blur_cliped_test_dilate'
    pre_processing(image_path, save_path)
