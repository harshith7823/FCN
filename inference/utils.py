
import os
import torch
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_inference_samples(output_dir, testloader, model, test_folder):
    image_outputs = gen_test_output(2, testloader, model, test_folder)
    for name, image in image_outputs:
        plt.imsave(os.path.join(output_dir, name), image)        

def resize_label(image_path, label):
    image = io.imread(image_path)
    label = transform.resize(label, image.shape)
    output = cv2.addWeighted(image, 0.6, label, 0.4, 0, dtype = 0)
    return output

def gen_test_output(n_class, testloader, model, test_folder):
    model.eval();
    with torch.no_grad():
        for i, data in enumerate(testloader):
            sample = data
            images = sample['image']
            images = images.float()
            images = Variable(images.to(device))

            output = model(images)
            output = torch.sigmoid(output)
            output = output.cpu()
            N, c, h, w = output.shape
            pred = np.squeeze(output.detach().cpu().numpy(), axis=0)

            pred = pred.transpose((1, 2, 0))
            pred = pred.argmax(axis=2)
            pred = (pred > 0.5)

            pred = pred.reshape(*pred.shape, 1)
            pred = np.concatenate((pred, np.invert(pred)), axis=2).astype('float')
            pred = np.concatenate((pred, np.zeros((*pred[:,:,0].shape, 1))), axis=2).astype('float')

            pred[pred == 1.0] = 127.0
            test_paths = get_test_paths(test_folder)
            output = resize_label(os.path.join(test_folder, test_paths[i]), pred)
            output = output/127.0
            output = np.clip(output, 0.0, 1.0)
            yield test_paths[i], output


def mean_pixel_accuracy(pixel_correct, pixel_labeled):
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    return mean_pixel_accuracy 

def compute_IoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))])) 

def pixel_accuracy(y_pred, y_true, N_CLASSES):
    y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES,img_rows,img_cols]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[N_CLASSES,img_rows,img_cols]),axis=0)
    y_pred = y_pred * (y_true>0)
    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)


