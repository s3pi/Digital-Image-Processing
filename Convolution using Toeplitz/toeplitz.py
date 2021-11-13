import cv2
import numpy as np

def matrix_to_vector(input):
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row
        
    return output_vector

def vector_to_matrix(input, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output


def toeplitz(c, r):
    c = np.asarray(c).ravel()
    r = np.asarray(r).ravel()
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    return vals[indx]


I = cv2.imread('image2.jpg',0)
I = cv2.resize(I,(128,128))
F = np.array([[0, 0.5,0], [0.5,1,0.5],[0,0.5,0]])

# I = np.array([[1, 2, 3], [4, 5, 6]])
# F = np.array([[10, 20], [30, 40]])


# Calculate the final output size 
# (m_1 + m_2 -1) x (n_1 + n_2 -1)

I_row_num, I_col_num = I.shape
F_row_num, F_col_num = F.shape
output_row_num = I_row_num + F_row_num - 1
output_col_num = I_col_num + F_col_num - 1

#                       Zero-pad the filter matrix to make it the same size as the output

F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),(0, output_col_num - F_col_num)),'constant', constant_values=0)

#                      Toeplitz matrix for each row of the zero-padded filter 
#number of columns of these generated Toeplitz matrices should be same as the number of columns of the input (I) matrix

toeplitz_list = []
for i in range(F_zero_padded.shape[0]-1, -1, -1):
    c = F_zero_padded[i, :]
    r = np.r_[c[0], np.zeros(I_col_num-1)]
    toeplitz_m = toeplitz(c,r)
    toeplitz_list.append(toeplitz_m)

#                        Create doubly blocked toeplitz matrix


c = range(1, F_zero_padded.shape[0]+1)
r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
doubly_indices = toeplitz(c, r)

toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
h = toeplitz_shape[0]*doubly_indices.shape[0]
w = toeplitz_shape[1]*doubly_indices.shape[1]
doubly_blocked_shape = [h, w]
doubly_blocked = np.zeros(doubly_blocked_shape)

b_h, b_w = toeplitz_shape 
for i in range(doubly_indices.shape[0]):
    for j in range(doubly_indices.shape[1]):
        start_i = i * b_h
        start_j = j * b_w
        end_i = start_i + b_h
        end_j = start_j + b_w
        doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

#       Vectorise the input matrix
vectorized_I = matrix_to_vector(I)

#                              Multiply doubly blocked toeplitz matrix with vectorized input signal

result_vector = np.matmul(doubly_blocked, vectorized_I)

#                         reshape the result to a matrix form

out_shape = [output_row_num, output_col_num]
my_output = vector_to_matrix(result_vector, out_shape)

##########################################################################################################
#             FOR COMPARISON

from scipy import signal

lib_output = signal.convolve2d(I, F, "full")

if(my_output.all() == lib_output.all()):
    print('yaaaaaaaaaaaaaaaaaaaaay')
########################################################################################################