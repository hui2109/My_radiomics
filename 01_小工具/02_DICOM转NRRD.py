import SimpleITK as STik

# filepath = r"D:\桌面\新建文件夹"

# dcms_name = STik.ImageSeriesReader.GetGDCMSeriesFileNames(filepath)
# print(dcms_name)
# print(type(dcms_name))
# dcms_read = STik.ImageSeriesReader()
# dcms_read.SetFileNames(dcms_name)
# dcms_series = dcms_read.Execute()
#
# STik.WriteImage(dcms_series, r'D:\1.nrrd')

# dcms_name = (r'D:\桌面\新建文件夹\example.dcm',)
dcms_read = STik.ImageSeriesReader()
dcms_read.SetFileNames((r'D:\桌面\新建文件夹\example.dcm',))
dcms_series = dcms_read.Execute()
STik.WriteImage(dcms_series, r'D:\1.nrrd')
