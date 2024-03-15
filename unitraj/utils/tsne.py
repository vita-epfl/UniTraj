import argparse
import random

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm


# from animals_dataset import AnimalsDataset, collate_skip_empty, colors_per_class
# from resnet import ResNet101


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset, batch, num_images):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    model = ResNet101(pretrained=True)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = AnimalsDataset(dataset, num_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels, image_paths


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, c):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    # color = colors_per_class[label]
    # if c =='red':
    #     c=[0,0,255]
    # elif c=='green':
    #     c=[0,255,0]
    # elif c=='blue':
    #     c=[255,0,0]
    c *= 255
    # exchange the first and the last channel in order to convert
    # it from RGB to BGR
    c[0], c[-1] = c[-1], c[0]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=c, thickness=4)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, c_list, plot_size=10000, max_image_size=500):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    plt.clf()
    # plt.rcParams["figure.figsize"] = [10, 10]
    fig = plt.figure(dpi=1000)
    ax = fig.add_subplot()
    ax.axis('off')
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    cnt = 0
    for image_path, x, y in tqdm(
            zip(images, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, c_list[cnt])

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
        cnt += 1

    ax.imshow(tsne_plot[:, :, ::-1])

    return plt


def visualize_tsne_points(Y, c_list):
    # initialize matplotlib plot
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = Y[:, 0]
    ty = Y[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure(dpi=1000)
    ax = fig.add_subplot(111)

    ax.scatter(tx, ty, s=1, c=c_list)

    color = sns.color_palette("colorblind")
    red_patch = mpatches.Patch(color=color[2], label='Waymo')
    blue_patch = mpatches.Patch(color=color[0], label='nuPlan')
    green_patch = mpatches.Patch(color=color[3], label='Argoverse2')

    ax.legend(handles=[red_patch, blue_patch, green_patch])

    ax.axis('off')
    # finally, show the plot
    return ax


def visualize_tsne(tsne, images, labels, plot_size=1000, max_image_size=500):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, plot_size=plot_size, max_image_size=max_image_size)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/raw-img')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    args = parser.parse_args()

    fix_random_seeds()

    features, labels, image_paths = get_features(
        dataset=args.path,
        batch=args.batch,
        num_images=args.num_images
    )

    tsne = TSNE(n_components=2).fit_transform(features)

    visualize_tsne(tsne, image_paths, labels)


if __name__ == '__main__':
    main()
