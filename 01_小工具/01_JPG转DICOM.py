import pydicom
from PIL import Image
import numpy as np


jpg_image = Image.open(r"D:\桌面\2.jpg")
ds = pydicom.dcmread(r"D:\桌面\1.dcm")


if jpg_image.mode == 'L':
    np_image = np.array(jpg_image.getdata(), dtype=np.uint8)
    ds.Rows = jpg_image.height
    ds.Columns = jpg_image.width
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.SamplesPerPixel = 1
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np_image.tobytes()
    ds.save_as('result_gray.dcm')

elif jpg_image.mode == 'RGB':
    np_image = np.array(jpg_image.getdata(), dtype=np.uint8)[:, :3]
    ds.Rows = jpg_image.height
    ds.Columns = jpg_image.width
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np_image.tobytes()
    ds.save_as('result_rgb.dcm')
