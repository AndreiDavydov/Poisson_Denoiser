import pywt
import numpy as np


class WaveletTransform():
    '''
    Class for using the particular wavelet transformation. After doing first transformation, it will save shapes of the image. 
    So, to work with the images of different size you should make another instance of this class.
    '''
    def __init__(self, wt_filt='haar', wt_scale=1, size=256):
        super(WaveletTransform, self).__init__()

        self.wt_filt = wt_filt
        self.wt_scale = wt_scale
        

        self.size = size # specified just to be able to compute the CG. (must be 256)
        self.shapes = []
        self.approx_coefs = None # here the coefs deleted by mask will be saved

    
    def wt_shapes_acqusition(self, coefs):
        '''
        The function creates a list of shapes of coeff-arrays at each scale.
        '''
        for i, coef_block in enumerate(coefs):
            if i == 0: # LL block
                self.shapes.append(coef_block.shape)
                
            else: # triplet for wavelet coefs.
                self.shapes += [(coef_block[0].shape, coef_block[1].shape, coef_block[2].shape)]     


    def W(self, img):
        '''
        Direct discrete 2-D Wavelet transformation.
        Returns the list of multiscale coefficients.
        '''
        mult_coefs = pywt.wavedec2(img, self.wt_filt, level=self.wt_scale)
        
        if len(self.shapes) == 0:   
            self.wt_shapes_acqusition(mult_coefs)
        
        return mult_coefs
    

    def W_inv(self, coefs):
        '''
        Inverse Discrete Wavelet transformation.
        Input types can be: np.array (img-like) | list of coefs | vector.
        If input is img-like - the dimensions must be equal and divided by 2.
        '''
        
        mult_coefs = coefs.copy()

        if isinstance(mult_coefs, np.ndarray): # img-like or vector case
            mult_coefs = self.as_coefs(mult_coefs)
        
        reconstructed = pywt.waverec2(mult_coefs, self.wt_filt)
        return reconstructed 

    
    def as_coefs(self, coefs):
        '''
        Tranform coefficients from img-like or vector-like input to the list of coefs.
        '''

        mult_coefs = coefs.copy()
        

        assert(isinstance(mult_coefs, np.ndarray))
        coefs = []

        if len(mult_coefs.shape) == 1: # vector-like case

            for i, block_shapes in enumerate(self.shapes):
                if i == 0:
                    len_coefs = block_shapes[0] * block_shapes[1]
                    LL_coefs = mult_coefs[:len_coefs].reshape(block_shapes[0], block_shapes[1])
                    coefs.append(LL_coefs)

                    mult_coefs = mult_coefs[len_coefs:]

                else:
                    coefs_per_block = []
                    for block_shape in block_shapes:
                        len_coefs = block_shape[0] * block_shape[1]
                        wt_coefs = mult_coefs[:len_coefs].reshape(block_shape[0], block_shape[1])
                        coefs_per_block.append(wt_coefs)

                        mult_coefs = mult_coefs[len_coefs:]

                    coefs.append(tuple(coefs_per_block))

            return coefs


        else: # img-like case

            reversed_shapes = self.shapes[::-1]
            
            for block_shapes in reversed_shapes:
                block_shape = block_shapes[0]

                if isinstance(block_shape, tuple):
                    block_shape = block_shape[0] # each wavelet block has same shape.

                    HVD = (mult_coefs[ :block_shape, block_shape: ], 
                           mult_coefs[ block_shape:, :block_shape ], 
                           mult_coefs[ block_shape:, block_shape: ])

                    coefs.append(HVD)
                    mult_coefs = mult_coefs[:block_shape, :block_shape]

                else:
                    coefs.append(mult_coefs)

            coefs = coefs[::-1]

        return coefs


    def as_vector(self, coefs):
        '''
        The input is either img-like object or a list of coefs.
        '''

        mult_coefs = coefs.copy()

        if isinstance(mult_coefs, np.ndarray): # must be an img-like case
            mult_coefs = self.as_coefs(mult_coefs)

        vector = mult_coefs[0].flatten()
        mult_coefs = mult_coefs[1:] # only wavelet blocks remained.

        for block in mult_coefs:
            for i in range(3):
                vector = np.concatenate((vector, block[i].flatten()))

        return vector

    
    def as_image(self, coefs):
        '''
        The input is a list of wavelet coefs with triplets for detailed coefs. 
        If it is a vector it will be transformed to the list.
        
        Returns image-like object (if possible).
        '''
        
        mult_coefs = coefs.copy()

        if isinstance(mult_coefs, np.ndarray): # vector-like case
            if len(mult_coefs.shape) == 1:
                mult_coefs = self.as_coefs(mult_coefs)
        
        try:
            block = mult_coefs[0]
            for i in range(1, len(mult_coefs)):
                (cH, cV, cD) = mult_coefs[i]
                block = np.block([[block, cH],[cV, cD]])

        except ValueError:
            print ('ValueError: Dimensions mismatch. Such WT cannot be viewed as a 2D image.')
        
        else:
            return block    
        

    def masked_coefs(self, coefs):
        '''
        Computes the masked wavelet transform given full list of wavelet coefficients. 
        (as if the binary mask was applied to the coefs for leaving only scaling (wavelet coefficients)).
        '''

        masked_mult_coefs = coefs.copy()

        if not isinstance(masked_mult_coefs, list):
            masked_mult_coefs = self.as_coefs(masked_mult_coefs) # a hope that the input is vector-like or img-like.

        self.approx_coefs = masked_mult_coefs[0].copy() # save deleted coefs.
        masked_mult_coefs[0] = np.zeros_like(masked_mult_coefs[0])

        return masked_mult_coefs


    def Wdir(self, img):
        '''
        Computes the masked WT (mask is applied only on scaling coefs). 
        Returns the vectorized format of coefs.
        '''
        return self.as_vector(self.masked_coefs(self.W(img)))

    def Wconj(self, vec_coefs):
        '''
        Computes the Conjugate to "Wdir" operator. By default, vec_coefs is given in a vectorized format.
        '''
        return self.W_inv(self.masked_coefs(self.as_coefs(vec_coefs)))


    def IsConjugateRight(self, eps=1e-5):
        '''
        Simple check for whether the Wavelet Conjugate Tranformation implemented here is correct.
        As direct WT the function "Wdirect" is used, as conjugate WT the "Wconj" is used.
        '''
        np.random.seed(5)

        x = np.random.rand(2048,2048)

        W_x = self.Wdir(x)
        y = np.random.randn(len(W_x))

        left = (W_x * y).sum()

        Wconj_y = self.Wconj(y)
        right = (Wconj_y * x).sum()

        print(np.abs(left - right) < eps)
