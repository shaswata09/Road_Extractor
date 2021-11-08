import cv2

def open_image(path):
    img = cv2.imread(path)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = "/home/shaswata/PycharmProjects/AI-Robotics-Project/data/satellite_images/Starkville_Satellite_Image3.png"
    open_image(img_path)