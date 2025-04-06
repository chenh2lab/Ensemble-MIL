import shutil
import cv2
import os
import numpy as np
from PIL import Image

def normalizeStaining(img, saveNorm=None, saveHE=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        saveNorm: output path for normalized patches (if None, don't save them)
        saveHE: output path for H, E patches (if None, don't save them)
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveNorm is not None:
        Image.fromarray(Inorm).save(saveNorm+'.png')
    
    if saveHE is not None:
        Image.fromarray(H).save(saveHE+'_H.png')
        Image.fromarray(E).save(saveHE+'_E.png')

    return Inorm, H, E

def varianceLaplacian(image, ksize=13):
    '''
    Input: image array (gray scale)
    Output: variance of Laplacian
    '''
    return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize).var()

raw_root = '/staging/biology/u1307362/TCGA-COAD_raw_tumor_patch/vgg19_bn/TCGA-HE-89K'
norm_root = '/staging/biology/u1307362/TCGA-COAD_norm_tumor_patch/vgg19_bn/TCGA-HE-89K'
blur_root = '/staging/biology/u1307362/TCGA-COAD_blur_tumor_patch/vgg19_bn/TCGA-HE-89K'

for svs in os.listdir(raw_root):
    os.mkdir(os.path.join(norm_root, svs))
    os.mkdir(os.path.join(blur_root, svs))
    for tile in os.listdir(os.path.join(raw_root, svs)):
        try:
            img = np.array(Image.open(os.path.join(raw_root, svs, tile)))
            normalizeStaining(img = img, saveNorm = os.path.join(norm_root, svs, tile[:-4]))
            norm_img = cv2.imread(os.path.join(norm_root, svs, tile))
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
            fm = varianceLaplacian(norm_img)
            if fm < 1e14 or fm > 1e15:
                shutil.move(os.path.join(norm_root, svs, tile), os.path.join(blur_root, svs))
        except:
            print('Normalization Error', f'{svs}', ',', tile)
