import numpy as np
import itertools


np.random.seed(0)


def main():
    # 4.14.1
    print('4.14.1')
    # Generator matrix for Hamming code
    G = np.array([
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ])

    # 4.14.2
    print('4.14.2')
    print(np.dot(G, np.array([1, 0, 0, 1])) % 2)

    # 4.14.3: find R s.t. np.dot(R, G) % 2 == np.eye(4)
    print('4.14.3')
    R = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ])
    for vec in itertools.product([0, 1], repeat=4):
        # Check whether decoding works well.
        message = np.array(vec)
        code = np.dot(G, message) % 2
        decoded_message = np.dot(R, code) % 2
        print(message, code, decoded_message)

    # 4.14.4
    print('4.14.4')
    H = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])
    print(np.dot(H, G) % 2)  # == np.zeros((3, 4))

    # 4.14.5
    print('4.14.5')
    def find_error(syndrome):
        n_column = H.shape[1]
        for column in range(n_column):
            if np.all(H[:, column] == syndrome):
                error = np.zeros(n_column, dtype=np.int)
                error[column] = 1
                return error
        return np.zeros(n_column, dtype=np.int)

    # 4.14.6
    print('4.14.6')
    tilde_c = np.array([1, 0, 1, 1, 0, 1, 1])
    syndrome = np.dot(H, tilde_c) % 2
    error = find_error(syndrome)
    error_fixed_tilde_c = (tilde_c + error) % 2
    print(tilde_c, error_fixed_tilde_c)

    # 4.14.7
    print('4.14.7')
    def find_error_matrix(S):
        return np.apply_along_axis(find_error, 0, S)
    S = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 1],
    ])
    print(find_error_matrix(S))

    # 4.14.8
    print('4.14.8')
    def str2bits(input_str):
        bits = [1 << i for i in range(8)]  # 1 byte = 8 bits
        return [1 if ord(char) & b > 0 else 0 for char in input_str for b in bits]
    def bits2str(input_bits):
        bits = [1 << i for i in range(8)]
        return ''.join(chr(
            sum(b * ib for b, ib in zip(bits, input_bits[i:i+8]))
        ) for i in range(0, len(input_bits), 8))
    s = ''.join([chr(i) for i in range(256)])
    ss = bits2str(str2bits(s))
    print(s)
    print(ss)

    # 4.14.9
    print('4.14.9')
    def bits2mat(bits, n_rows=4, trans=False):
        n_cols = len(bits) // n_rows
        mat = np.zeros((n_rows, n_cols), dtype=np.int)
        for i in range(n_rows):
            for j in range(n_cols):
                if bits[n_rows * j + i] > 0:
                    mat[i, j] = 1
        if trans:
            mat = mat.T
        return mat
    def mat2bits(mat, trans=False):
        n_rows, n_cols = mat.shape
        if trans:
            return [mat[i, j] for i in range(n_rows) for j in range(n_cols)]
        else:
            return [mat[i, j] for j in range(n_cols) for i in range(n_rows)]
    s = ''.join([chr(i) for i in range(256)])
    ss = bits2str(mat2bits(bits2mat(str2bits(s))))
    print(s)
    print(ss)

    # 4.14.10
    print('4.14.10')
    message = 'I’m trying to free your mind, Neo. But I can only show you the door. You’re the one that has to walk through it.'
    P = bits2mat(str2bits(message))

    # 4.14.11
    print('4.14.11')
    error_rate = 0.02
    E = np.random.choice([0, 1], P.shape, [1 - error_rate, error_rate])
    P_TILDE = (P + E) % 2
    decoded_message = bits2str(mat2bits(P_TILDE))
    print(decoded_message)  # non-readable message

    # 4.14.12
    print('4.14.12')
    C = np.dot(G, P) % 2

    # 4.14.13
    print('4.14.13')
    error_rate = 0.02
    C_TILDE = (C + np.random.choice([0, 1], C.shape, p=[1 - error_rate, error_rate])) % 2
    decoded_message = bits2str(mat2bits(np.dot(R, C_TILDE) % 2))
    print(decoded_message)  # non-readable message

    # 4.14.14
    print('4.14.14')
    def correct(mat):
        return (mat + find_error_matrix(np.dot(H, mat) % 2)) % 2

    # 4.14.15
    print('4.14.15')
    C_CORRECTED = correct(C_TILDE)
    decoded_message = bits2str(mat2bits(np.dot(R, C_CORRECTED) % 2))
    print(decoded_message)

    # 4.14.16
    print('4.14.16')
    for error_rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
        C_TILDE = (C + np.random.choice([0, 1], C.shape, p=[1 - error_rate, error_rate])) % 2
        C_CORRECTED = correct(C_TILDE)
        decoded_message = bits2str(mat2bits(np.dot(R, C_CORRECTED) % 2))
        print('[Result for error rate {}]: '.format(error_rate) + decoded_message)


if __name__ == '__main__':
    main()