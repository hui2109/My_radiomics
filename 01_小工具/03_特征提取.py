import six
from radiomics import featureextractor

imageName = '../00_MaterialBox/brain1_image.nrrd'
maskName = '../00_MaterialBox/brain1_label.nrrd'
extractor = featureextractor.RadiomicsFeatureExtractor()
result = extractor.execute(imageName, maskName)
for key, val in six.iteritems(result):
    print("\t%s: %s" % (key, val))
