from apriltag_calibrate.configparase.TagBundle import TagBundle

def test_TagBundle():
    tag=TagBundle()
    tag.load('bundle_result.yaml')
    for key, value in tag.tag_points.items():
        print(key, value)
    
test_TagBundle()