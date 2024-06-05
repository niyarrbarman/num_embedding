import numpy as np
from numpy.fft import fft, ifft

class NumberEmbedding:
    def __init__(self, length=4096):
        self.length = length

    def digit_to_complex(self, digit):
        angle = np.pi * (digit / 10)
        return np.cos(angle) + 1j * np.sin(angle)

    def number_to_complex_list(self, number):
        str_number = str(number)
        if '.' in str_number:
            integer_part, decimal_part = str_number.split('.')
            integer_complex_list = [self.digit_to_complex(int(d)) for d in integer_part]
            decimal_complex_list = [self.digit_to_complex(int(d)) for d in decimal_part]
            return integer_complex_list, decimal_complex_list
        else:
            integer_complex_list = [self.digit_to_complex(int(d)) for d in str_number]
            return integer_complex_list, []

    def apply_fft(self, complex_list):
        return fft(complex_list)

    def encode(self, number):
        sign = 1 if number >= 0 else -1
        number = abs(number)
        integer_part, decimal_part = self.number_to_complex_list(number)
        
        int_length = len(integer_part)
        dec_length = len(decimal_part)
        
        complex_list = integer_part + decimal_part
        
        fft_coeffs = self.apply_fft(complex_list)
        
        length_info = [int_length, dec_length]
        
        embedding = np.concatenate([[sign], length_info, fft_coeffs.real, fft_coeffs.imag])
        
        if len(embedding) < self.length:
            embedding = np.pad(embedding, (0, self.length - len(embedding)), mode='constant')
        elif len(embedding) > self.length:
            embedding = embedding[:self.length]
        
        return embedding

    def decode(self, embedding):
        sign = embedding[0]
        
        int_length, dec_length = map(int, embedding[1:3])
        
        real_part = embedding[3:3 + int_length + dec_length]
        imag_part = embedding[3 + int_length + dec_length:3 + 2 * (int_length + dec_length)]
        fft_coeffs = real_part + 1j * imag_part
        
        recovered_complex_list = ifft(fft_coeffs)
        
        def complex_to_digit(c):
            angle = np.angle(c)
            digit = int(round((angle / np.pi) * 10)) % 10
            return digit
        
        recovered_digits = [complex_to_digit(c) for c in recovered_complex_list]
        
        if dec_length > 0:
            recovered_number = float(''.join(map(str, recovered_digits[:int_length]))) + \
                               float('0.' + ''.join(map(str, recovered_digits[int_length:])))
        else:
            recovered_number = int(''.join(map(str, recovered_digits)))
        
        recovered_number *= sign
        
        return recovered_number

# Test the NumberEmbedding class
number = -54.454
number_embedding = NumberEmbedding()

print('Input Number:', number)
embedding = number_embedding.encode(number)
print("Embedding:", embedding)
print("Embedding shape:", embedding.shape)

recovered_number = number_embedding.decode(embedding)
print("Decoded Number:", recovered_number)
