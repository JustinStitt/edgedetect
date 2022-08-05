from PIL import Image
import numpy as np
import sys

VKERNEL = np.array([
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
                ])
HKERNEL = np.array([
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
                ])
KERNEL_SIZE = 3

def kernelPass(image: np.ndarray, 
               kernels: list[np.ndarray]):
    assert(len(image) and len(image[0])) # non empty image
    rows = len(image)
    cols = len(image[0])
    assert(rows >= KERNEL_SIZE and cols >= KERNEL_SIZE) # kernel fits
    out = np.zeros((rows-2, cols-2, 3), int) # adjust for padding 
    for r in range(1, rows-2):
        for c in range(1, cols-2):
            lookingat = image[r-1:r+2, c-1:c+2, 1]
            vsum = (lookingat*VKERNEL).sum()/4
            hsum = (lookingat*HKERNEL).sum()/4
            score = (vsum**2 + hsum**2)**0.5
            out[r, c] = np.array([score]*3) * np.array([.15, .15, .7])
    return out

def main():
    im_path = sys.argv[1]
    try:
        image = Image.open(im_path)
    except:
        print(f"Couldn't open image: {im_path}...")
    image_asarray = np.array(image)
    # add padding to image so that kernel doesn't need to do bound checking
    image_asarray = np.pad(image_asarray, 1)
    result = kernelPass(image_asarray, [VKERNEL, HKERNEL])
    # print(result)
    image_result = Image.fromarray(result.astype(np.uint8))
    image_result.save(f'out.jpg')

if __name__ == '__main__':
    main()
