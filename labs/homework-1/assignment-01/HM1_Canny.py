import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    # Normalized gradient direction to [0, 180)
    grad_dir = (grad_dir/np.pi*180) % 180

    # Creating a Direction Mask
    mask_east = (grad_dir >= 157.5) | (grad_dir < 22.5)
    mask_northeast = (grad_dir >= 22.5) & (grad_dir < 67.5)
    mask_north = (grad_dir >= 67.5) & (grad_dir < 112.5)
    mask_northwest = (grad_dir >= 112.5) & (grad_dir < 157.5)

    # Get the gradient magnitude of neighboring pixels
    shift_east = np.roll(grad_mag, shift=-1, axis=1)
    shift_west = np.roll(grad_mag, shift=1, axis=1)
    shift_north = np.roll(grad_mag, shift=-1, axis=0)
    shift_south = np.roll(grad_mag, shift=1, axis=0)
    shift_northeast = np.roll(shift_north, shift=-1, axis=1)
    shift_southwest = np.roll(shift_south, shift=1, axis=1)
    shift_northwest = np.roll(shift_north, shift=1, axis=1)
    shift_southeast = np.roll(shift_south, shift=-1, axis=1)

    NMS_output = np.zeros_like(grad_mag)
    NMS_output[mask_east & (grad_mag >= shift_east) & (grad_mag >= shift_west)] = grad_mag[mask_east &
                                                                                           (grad_mag >= shift_east) & (grad_mag >= shift_west)]
    NMS_output[mask_northeast & (grad_mag >= shift_northeast) & (grad_mag >= shift_southwest)] = grad_mag[mask_northeast &
                                                                                                          (grad_mag >= shift_northeast) & (grad_mag >= shift_southwest)]
    NMS_output[mask_north & (grad_mag >= shift_north) & (grad_mag >= shift_south)] = grad_mag[mask_north &
                                                                                              (grad_mag >= shift_north) & (grad_mag >= shift_south)]
    NMS_output[mask_northwest & (grad_mag >= shift_northwest) & (grad_mag >= shift_southeast)] = grad_mag[mask_northwest &
                                                                                                          (grad_mag >= shift_northwest) & (grad_mag >= shift_southeast)]

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    # you can adjust the parameters to fit your own implementation
    low_ratio = 0.07
    high_ratio = 0.20

    # Define the high and low thresholds
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    # Identify strong and weak edge pixels
    strong_edges_mask = (img >= high_threshold)
    weak_edges_mask = ((img >= low_threshold) & (img < high_threshold))

    # Label connected components of the strong edges
    from scipy.ndimage import label
    labeled_strong_edges, num_features = label(strong_edges_mask, structure=np.ones((3, 3)))

    # If a weak edge pixel is connected to a strong edge pixel, it's considered part of an edge
    output_mask = np.copy(strong_edges_mask)
    for i in range(1, num_features + 1):
        # Find weak edges connected to the current strong edge connected-set
        connected_weak_edges = (labeled_strong_edges == i) & weak_edges_mask
        # Mark them as edge pixels in the output_mask
        output_mask += connected_weak_edges

    # Ensure the output is binary
    output_mask = (output_mask > 0)

    # If output = output_mask * img, the picture will be more blurred
    output = output_mask

    return output


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("Lenna.png")/255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img*255)
