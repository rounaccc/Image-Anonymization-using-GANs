import torch
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont


def draw_image_with_boxes(filename, results, model_name):
    image = Image.open(filename)
    draw = ImageDraw.Draw(image)

    for result in results:
        x, y, width, height = result['box']
        confidence = result['confidence']
        confidence = confidence * 100

        draw.rectangle([x, y, x + width, y + height], outline='red', width=2)

        text = f"{confidence:.2f}"
        draw.text((x, y), text, fill='white')

    image.show(title=f"Model: {model_name}")

def compare_models(pixels,filename=""):
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    mtcnn_accuracy = sum([face['confidence'] for face in faces]) / len(faces) if faces else 0

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(pixels)
    person_results = [obj for obj in results.xyxy[0] if model.names[int(obj[5])] == 'person']
    yolov5_accuracy = sum([obj[4].item() * 100 for obj in person_results]) / len(person_results) if person_results else 0

    mtcnn_accuracy =  mtcnn_accuracy * 100

    if mtcnn_accuracy > yolov5_accuracy:
        model_name='MTCNN'
        print(f"MTCNN has higher accuracy ({mtcnn_accuracy:.2f}%) for face detection compared to YOLOv5 which has ({yolov5_accuracy:.2f}%)")
        # draw_image_with_boxes(filename, faces, 'MTCNN')
    else:
        model_name='YOLOv5'
        print(f"YOLOv5 has higher accuracy ({yolov5_accuracy:.2f}%) for person detection compared to MTCNN which has ({mtcnn_accuracy:.2f}%)")
        results.show()

    return mtcnn_accuracy, yolov5_accuracy,model_name

if __name__=='__main__':
    filename="Photo on 21-03-24 at 15.48.jpg"
    pixels = pyplot.imread(filename)
    mtcnn_accuracy, yolov5_accuracy = compare_models(filename=filename,pixels=pixels)