
# Set head pose model
MODEL=model/cls_image_sharp_v1.0.0.trt

# Set test data
IMAGES_PATH=images/testing/

# Set test data
CONFIG_PATH=model/config.ini


libs/linux/gpu/performance_testing_GPU $MODEL $IMAGES_PATH $CONFIG_PATH 1




