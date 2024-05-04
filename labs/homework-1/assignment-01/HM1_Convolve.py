import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type == "zeroPadding":
        padding_img = np.zeros((img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        return padding_img
    elif type == "replicatePadding":
        padding_img = np.zeros((img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        padding_img[0:padding_size, padding_size:padding_size+img.shape[1]] = img[0, :]
        padding_img[padding_size+img.shape[0]:, padding_size:padding_size+img.shape[1]] = img[-1, :]
        padding_img[:, 0:padding_size] = padding_img[:, padding_size:padding_size+1]
        padding_img[:, padding_size+img.shape[1]:] = padding_img[:, padding_size+img.shape[1]-1:padding_size+img.shape[1]]
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)

        toeplitz_matrix: 36*64, each line's non-zero values are a convolution kernel
        padding_img.flatten(): 64*1, the image after zero padding
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")

    # build the Toeplitz matrix and compute convolution
    img_idx = np.arange(0, 64).reshape(8, 8)
    img_kernel_0_idx = img_idx[0:6, 0:6].flatten()  # index of kernel value 0, for all 36 lines of Toeplitz matrix
    img_kernel_1_idx = img_idx[0:6, 1:7].flatten()  # index of kernel value 1, for all 36 lines of Toeplitz matrix
    img_kernel_2_idx = img_idx[0:6, 2:8].flatten()  # index of kernel value 2, for all 36 lines of Toeplitz matrix
    img_kernel_3_idx = img_idx[1:7, 0:6].flatten()  # index of kernel value 3, for all 36 lines of Toeplitz matrix
    img_kernel_4_idx = img_idx[1:7, 1:7].flatten()  # index of kernel value 4, for all 36 lines of Toeplitz matrix
    img_kernel_5_idx = img_idx[1:7, 2:8].flatten()  # index of kernel value 5, for all 36 lines of Toeplitz matrix
    img_kernel_6_idx = img_idx[2:8, 0:6].flatten()  # index of kernel value 6, for all 36 lines of Toeplitz matrix
    img_kernel_7_idx = img_idx[2:8, 1:7].flatten()  # index of kernel value 7, for all 36 lines of Toeplitz matrix
    img_kernel_8_idx = img_idx[2:8, 2:8].flatten()  # index of kernel value 8, for all 36 lines of Toeplitz matrix

    toeplitz_matrix = np.zeros((36, 64))
    toeplitz_matrix[np.arange(0, 36), img_kernel_0_idx] = kernel[0, 0]
    toeplitz_matrix[np.arange(0, 36), img_kernel_1_idx] = kernel[0, 1]
    toeplitz_matrix[np.arange(0, 36), img_kernel_2_idx] = kernel[0, 2]
    toeplitz_matrix[np.arange(0, 36), img_kernel_3_idx] = kernel[1, 0]
    toeplitz_matrix[np.arange(0, 36), img_kernel_4_idx] = kernel[1, 1]
    toeplitz_matrix[np.arange(0, 36), img_kernel_5_idx] = kernel[1, 2]
    toeplitz_matrix[np.arange(0, 36), img_kernel_6_idx] = kernel[2, 0]
    toeplitz_matrix[np.arange(0, 36), img_kernel_7_idx] = kernel[2, 1]
    toeplitz_matrix[np.arange(0, 36), img_kernel_8_idx] = kernel[2, 2]

    output = toeplitz_matrix @ padding_img.flatten()
    return output.reshape(6, 6)


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float) unknown shape
            kernel: array(float) , shape = (k, k)
        Outputs:
            output: array(float)

        img_matrix: ((Nx-2)*(Ny-2), 9), each line is the valuse of one sliding window of the image (no padding for the original image)
        kernel.flatten(): (9, 1), the kernel after flatten
    """
    Nx, Ny = img.shape
    k = kernel.shape[0]
    img_idx = np.arange(0, Nx*Ny).reshape(Nx, Ny)
    '''use fancy indexing to build the sliding-window convolution'''
    '''I spent two hours on this question; and this method is really too FANCY  ~ʕ•ᴥ•ʔ~  '''
    x_start, y_start, k_range = np.arange(0, Nx-k+1), np.arange(0, Ny-k+1), np.arange(0, k)
    x_idx = x_start[:, None, None, None]+k_range[None, None, :, None]
    y_idx = y_start[None, :, None, None]+k_range[None, None, None, :]
    img_matrix_idx = img_idx[x_idx, y_idx]  # broadcast x_idx and y_idx to a non-singleton 4-dim tensor
    img_matrix_idx = img_matrix_idx.reshape((Nx-k+1)*(Ny-k+1), k*k)

    img = img.flatten()
    img_matrix = img[img_matrix_idx]

    output = img_matrix @ kernel.flatten()
    return output.reshape((Nx-k+1, Ny-k+1))


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    # write_img("result/HM1_input_img.png", input_img*255)
    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)
